"""Small intent model: a pretrained multilingual encoder + our own linear head.

    text -> tokenizer -> encoder (XLM-R / MuRIL) -> [CLS] vector -> dropout -> Linear(hidden, 60) -> logits
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class IntentClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.encoder_name = encoder_name
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = AutoConfig.from_pretrained(encoder_name).hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # first token (<s> / [CLS]) summarises the sentence
        return self.head(self.dropout(cls))  # [batch, num_labels] logits


def save_model(model: IntentClassifier, tokenizer, out_dir: Path, extra: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    meta = {"encoder_name": model.encoder_name, "num_labels": model.num_labels, **(extra or {})}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_model(model_dir: Path, device: str = "cpu"):
    """Load a saved checkpoint. Returns (model in eval mode, tokenizer, meta)."""
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model = IntentClassifier(meta["encoder_name"], meta["num_labels"])
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    return model, tokenizer, meta


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], device: str = "cpu", max_len: int = 64,
                   batch_size: int = 64) -> torch.Tensor:
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        chunks.append(model(enc["input_ids"], enc["attention_mask"]).float().cpu())
    return torch.cat(chunks) if chunks else torch.empty(0, model.num_labels)
