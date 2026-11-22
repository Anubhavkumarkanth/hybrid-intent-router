"""The small model as an ONNX file, run with onnxruntime (no PyTorch needed).

Used for the free 512 MB deployment: the INT8 model is about 4x smaller than the PyTorch checkpoint, and
onnxruntime + tokenizers use far less memory than PyTorch + transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxIntentModel:
    def __init__(self, model_dir: Path, threads: int = 1):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / self.meta["onnx_file"]), opts,
                                            providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.meta["max_len"])
        pad_id = self.tok.token_to_id("[PAD]")
        self.tok.enable_padding(pad_id=pad_id, pad_token="[PAD]")

    def logits(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i : i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            out.append(self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0])
        return np.concatenate(out) if out else np.empty((0, self.meta["num_labels"]), dtype=np.float32)
