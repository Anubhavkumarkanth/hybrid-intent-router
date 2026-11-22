"""Fine-tune the small intent model on ENGLISH train only, with our own PyTorch loop.

Loop: DataLoader -> forward -> cross-entropy -> backward -> clip grads -> AdamW step -> LR schedule.
After every epoch: accuracy on English dev. Keep the best checkpoint; stop early if dev
accuracy does not improve for `patience` epochs.

Usage:
    python -m router.train                 # full run (settings from PLAN.md)
    python -m router.train --smoke 30      # 30-step timing run, no checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from router.data import load_labels, load_split
from router.model import IntentClassifier, predict_logits, save_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "small"
RESULTS = ROOT / "results"

DEFAULTS = dict(
    encoder="google/muril-base-cased",
    max_len=64,
    batch_size=32,
    lr=2e-5,
    head_lr=1e-3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    epochs=5,
    patience=2,
    grad_clip=1.0,
    dropout=0.1,
    seed=42,
    amp=True,
    freeze_embeddings=True,  # word-embedding matrix frozen: fits 4 GB GPU, as in the MASSIVE paper
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.texts = [r["text"] for r in rows]
        self.labels = [r["label"] for r in rows]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

    return collate


def linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    """LR goes 0 -> peak over warmup, then linearly down to 0 at the end."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def evaluate_acc(model, tokenizer, rows, device, max_len) -> float:
    logits = predict_logits(model, tokenizer, [r["text"] for r in rows], device=device, max_len=max_len)
    preds = logits.argmax(-1).numpy()
    return float((preds == np.array([r["label"] for r in rows])).mean())


def train(cfg: dict, smoke_steps: int = 0) -> dict:
    from transformers import AutoTokenizer

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = load_labels()
    train_rows = load_split("en", "train")
    dev_rows = load_split("en", "dev")

    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])
    model = IntentClassifier(cfg["encoder"], len(labels), dropout=cfg["dropout"]).to(device)

    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(TextDataset(train_rows), batch_size=cfg["batch_size"], shuffle=True, generator=g,
                        collate_fn=make_collate(tokenizer, cfg["max_len"]))

    if cfg["freeze_embeddings"]:
        # Freeze only the big word-embedding matrix (~150M params for MuRIL's 197k vocab).
        # Saves GPU memory and keeps the multilingual word vectors unchanged.
        model.encoder.get_input_embeddings().weight.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"parameters: total {n_total / 1e6:.1f}M, trainable {n_train / 1e6:.1f}M")

    # No weight decay on biases and LayerNorm weights (standard practice).
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    named = [(n, p) for n, p in model.encoder.named_parameters() if p.requires_grad]
    enc_decay = [p for n, p in named if not any(k in n for k in no_decay)]
    enc_nodecay = [p for n, p in named if any(k in n for k in no_decay)]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_decay, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]},
            {"params": enc_nodecay, "lr": cfg["lr"], "weight_decay": 0.0},
            {"params": model.head.parameters(), "lr": cfg["head_lr"], "weight_decay": cfg["weight_decay"]},
        ]
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = linear_warmup_decay(optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = cfg["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    print(f"device={device} train={len(train_rows)} dev={len(dev_rows)} steps/epoch={steps_per_epoch}")
    history, best_acc, best_epoch, bad_epochs = [], -1.0, -1, 0
    t_start = time.perf_counter()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        run_loss, n_seen, t_epoch = 0.0, 0, time.perf_counter()
        for input_ids, attn, y in loader:
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attn)
                loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total_steps} loss {run_loss / n_seen:.4f}")
            if smoke_steps and step >= smoke_steps:
                sec = time.perf_counter() - t_start
                est = sec / step * total_steps
                res = {"device": device, "encoder": cfg["encoder"], "steps": step, "seconds": round(sec, 1),
                       "sec_per_step": round(sec / step, 3), "steps_per_epoch": steps_per_epoch,
                       "est_total_train_minutes": round(est / 60, 1), "loss": round(run_loss / n_seen, 4)}
                print(json.dumps(res))
                return res
        train_loss = run_loss / n_seen
        dev_acc = evaluate_acc(model, tokenizer, dev_rows, device, cfg["max_len"])
        epoch_sec = time.perf_counter() - t_epoch
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "dev_acc": round(dev_acc, 4),
                        "seconds": round(epoch_sec, 1)})
        print(f"epoch {epoch}: train_loss {train_loss:.4f} en_dev_acc {dev_acc:.4f} ({epoch_sec:.0f}s)")
        if dev_acc > best_acc:
            best_acc, best_epoch, bad_epochs = dev_acc, epoch, 0
            save_model(model, tokenizer, MODEL_DIR, extra={"max_len": cfg["max_len"], "best_epoch": epoch,
                                                           "en_dev_acc": dev_acc})
            print(f"  saved best checkpoint (epoch {epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping: no dev improvement for {bad_epochs} epochs")
                break

    summary = {
        "config": cfg,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "history": history,
        "params_total_m": round(n_total / 1e6, 1),
        "params_trainable_m": round(n_train / 1e6, 1),
        "best_epoch": best_epoch,
        "best_en_dev_acc": round(best_acc, 4),
        "train_minutes": round((time.perf_counter() - t_start) / 60, 1),
        "math_check_total_steps": total_steps,
        "math_check_warmup_steps": int(cfg["warmup_ratio"] * total_steps),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "training_log.json").write_text(json.dumps(summary, indent=2))
    plot_curve(history)
    return summary


def plot_curve(history: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax1.plot(ep, [h["train_loss"] for h in history], "o-", color="#2a6fdb", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#2a6fdb")
    ax2 = ax1.twinx()
    ax2.plot(ep, [100 * h["dev_acc"] for h in history], "s-", color="#e07b00", label="English dev accuracy")
    ax2.set_ylabel("English dev accuracy (%)", color="#e07b00")
    ax1.set_xticks(ep)
    ax1.set_title("Training curve (English train only)")
    fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))
    fig.tight_layout()
    out = RESULTS / "charts" / "training_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run N steps and estimate total time")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    args = vars(ap.parse_args())
    smoke = args.pop("smoke")
    res = train(args, smoke_steps=smoke)
    if smoke:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "smoke_run.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
