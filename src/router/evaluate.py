"""Evaluate the trained small model.

Writes:
- results/small_model.json: accuracy on full test (en/hi/kn), on test_sub and dev_sub,
  CPU latency for one message (median/p95), model size on disk.
- results/logits/{name}.npz: per-item logits + gold labels + ids for
  en_dev (full English dev, for temperature scaling) and
  {lang}_dev_sub / {lang}_test_sub (for routing).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from router.data import load_split, load_subset
from router.model import load_model, predict_logits

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "small"
RESULTS = ROOT / "results"
LOGITS_DIR = RESULTS / "logits"
LANGS = ("en", "hi", "kn")
LATENCY_THREADS = 4  # same per-worker thread count used for serving (deploy/gunicorn.conf.py)


def save_logits(name: str, rows: list[dict], logits: torch.Tensor) -> None:
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LOGITS_DIR / f"{name}.npz",
        ids=np.array([r["id"] for r in rows]),
        labels=np.array([r["label"] for r in rows]),
        logits=logits.numpy().astype(np.float32),
    )


def load_logits(name: str) -> dict:
    d = np.load(LOGITS_DIR / f"{name}.npz")
    return {"ids": d["ids"], "labels": d["labels"], "logits": d["logits"]}


def accuracy(logits: torch.Tensor, rows: list[dict]) -> float:
    return float((logits.argmax(-1).numpy() == np.array([r["label"] for r in rows])).mean())


def measure_cpu_latency(model_dir: Path, texts: list[str], n: int = 200, warmup: int = 20) -> dict:
    torch.set_num_threads(LATENCY_THREADS)
    model, tok, meta = load_model(model_dir, device="cpu")
    max_len = meta.get("max_len", 64)
    times = []
    with torch.inference_mode():
        for i in range(warmup + n):
            text = texts[i % len(texts)]
            t0 = time.perf_counter()
            enc = tok([text], truncation=True, max_length=max_len, return_tensors="pt")
            logits = model(enc["input_ids"], enc["attention_mask"])
            torch.softmax(logits, -1)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "threads": LATENCY_THREADS,
        "n": n,
        "warmup": warmup,
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[int(0.95 * len(times)) - 1], 2),
        "mean_ms": round(statistics.fmean(times), 2),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_model(MODEL_DIR, device=device)
    max_len = meta.get("max_len", 64)

    def run(rows):
        return predict_logits(model, tok, [r["text"] for r in rows], device=device, max_len=max_len)

    out: dict = {"encoder": meta["encoder_name"], "best_epoch": meta.get("best_epoch"),
                 "trained_on": "en-US train only", "accuracy": {}}

    en_dev = load_split("en", "dev")
    lg = run(en_dev)
    save_logits("en_dev", en_dev, lg)
    out["accuracy"]["en_dev_full"] = round(accuracy(lg, en_dev), 4)

    for lang in LANGS:
        test = load_split(lang, "test")
        out["accuracy"][f"{lang}_test_full"] = round(accuracy(run(test), test), 4)
        for sub in ("dev_sub", "test_sub"):
            rows = load_subset(lang, sub)
            lg = run(rows)
            save_logits(f"{lang}_{sub}", rows, lg)
            out["accuracy"][f"{lang}_{sub}"] = round(accuracy(lg, rows), 4)
        print(lang, {k: v for k, v in out["accuracy"].items() if k.startswith(lang)})

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    texts = [r["text"] for lang in LANGS for r in load_subset(lang, "test_sub")]
    texts = [texts[i] for i in np.random.default_rng(0).permutation(len(texts))]
    out["cpu_latency_single_message"] = measure_cpu_latency(MODEL_DIR, texts)
    out["model_size_mb"] = round((MODEL_DIR / "model.pt").stat().st_size / 1e6, 1)
    out["num_parameters_m"] = round(
        sum(v.numel() for v in torch.load(MODEL_DIR / "model.pt", map_location="cpu",
                                          weights_only=True).values()) / 1e6, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "small_model.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
