"""Shrink the trained model for the free 512 MB deployment and measure what that costs (make quantize).

1. Vocabulary pruning: MuRIL's 197k-token vocabulary covers 17 scripts. Keep only tokens written in Latin,
   Devanagari or Kannada. WordPiece only matches vocabulary pieces that occur in the word itself, so English,
   Hindi and Kannada text gets exactly the same tokens (checked on every MASSIVE en/hi/kn sentence).
2. Export to ONNX (fp32), then
3. store the word-embedding table as int8 with one scale per row (Gather -> Cast -> Mul), and
4. quantize every Linear layer's weights to 8 bits in blocks of 32 (MatMulNBits), with int8 arithmetic.
   (Plain dynamic INT8 quantization also quantizes activations and agreed with the original on only 18-82% of
   sentences; that is why it is not used.)

Writes models/onnx/ (model.onnx, tokenizer.json, meta.json) and results/quantized.json. T and t are not
changed: they stay as chosen on dev data for the original model.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router import calibrate as C  # noqa: E402
from router import route as R  # noqa: E402
from router.data import load_labels, load_split, load_subset  # noqa: E402
from router.evaluate import load_logits  # noqa: E402
from router.llm import INVALID, Cache  # noqa: E402

SRC = ROOT / "models" / "small"
OUT = ROOT / "models" / "onnx"
RESULTS = ROOT / "results"
LANGS = ("en", "hi", "kn")


def in_scope(ch: str) -> bool:
    """Characters of English, Hindi and Kannada text (plus digits, punctuation and joiners)."""
    cp = ord(ch)
    return (cp < 0x0370 or 0x1E00 <= cp <= 0x1EFF or 0x2000 <= cp <= 0x20CF  # Latin, punctuation, currency
            or 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF or 0x1CD0 <= cp <= 0x1CFF  # Devanagari
            or 0x0C80 <= cp <= 0x0CFF)  # Kannada


def prune_tokenizer(tok_json: dict) -> tuple[dict, list[int]]:
    """Return (pruned tokenizer.json, kept old ids in order). Special tokens are always kept."""
    vocab = tok_json["model"]["vocab"]
    specials = {t["content"] for t in tok_json.get("added_tokens", [])}
    kept = sorted(i for tok, i in vocab.items()
                  if tok in specials or all(in_scope(c) for c in (tok[2:] if tok.startswith("##") else tok)))
    old_to_new = {old: new for new, old in enumerate(kept)}
    new = json.loads(json.dumps(tok_json))
    new["model"]["vocab"] = {tok: old_to_new[i] for tok, i in vocab.items() if i in old_to_new}
    for t in new.get("added_tokens", []):
        t["id"] = old_to_new[t["id"]]
    for spec in (new.get("post_processor") or {}).get("special_tokens", {}).values():
        spec["ids"] = [old_to_new[i] for i in spec["ids"]]
    new["truncation"], new["padding"] = None, None
    return new, kept


def quantize_embedding_rows(model, table_name: str):
    """Replace Gather(table, ids) with int8 rows + one float scale per row: Gather -> Cast -> Mul."""
    import onnx
    from onnx import helper, numpy_helper

    g = model.graph
    init = next(i for i in g.initializer if i.name == table_name)
    w = numpy_helper.to_array(init).astype(np.float32)
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    g.initializer.remove(init)
    g.initializer.extend([numpy_helper.from_array(q, table_name + "_int8"),
                          numpy_helper.from_array(scale.astype(np.float32), table_name + "_scale"),
                          numpy_helper.from_array(np.array([-1], dtype=np.int64), table_name + "_axes")])
    for pos, node in enumerate(list(g.node)):
        if node.op_type == "Gather" and node.input[0] == table_name:
            ids, out, p = node.input[1], node.output[0], table_name.replace(".", "_")
            new_nodes = [
                helper.make_node("Gather", [table_name + "_int8", ids], [p + "_q"], axis=0),
                helper.make_node("Gather", [table_name + "_scale", ids], [p + "_s"], axis=0),
                helper.make_node("Cast", [p + "_q"], [p + "_f"], to=onnx.TensorProto.FLOAT),
                helper.make_node("Unsqueeze", [p + "_s", table_name + "_axes"], [p + "_s1"]),
                helper.make_node("Mul", [p + "_f", p + "_s1"], [out]),
            ]
            g.node.remove(node)
            for k, n in enumerate(new_nodes):
                g.node.insert(pos + k, n)
    return model


def build() -> dict:
    import onnx
    import torch
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    from router.model import load_model

    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, meta = load_model(SRC, device="cpu")
    tok_json = json.loads((SRC / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    new_tok_json, kept = prune_tokenizer(tok_json)
    (OUT / "tokenizer.json").write_text(json.dumps(new_tok_json, ensure_ascii=False), encoding="utf-8")

    emb = model.encoder.get_input_embeddings()
    pruned = torch.nn.Embedding(len(kept), emb.embedding_dim, padding_idx=None)
    pruned.weight.data = emb.weight.data[torch.tensor(kept)].clone()
    model.encoder.set_input_embeddings(pruned)
    model.encoder.config.vocab_size = len(kept)

    from tokenizers import Tokenizer

    t2 = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    sample = t2.encode_batch(["wake me up at five am", "play some music please"])
    width = max(len(e.ids) for e in sample)
    ids = torch.tensor([e.ids + [0] * (width - len(e.ids)) for e in sample])
    mask = torch.tensor([e.attention_mask + [0] * (width - len(e.ids)) for e in sample])
    fp32 = OUT / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, (ids, mask), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                        "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}})
    m = onnx.load(str(fp32))
    table = next(i.name for i in m.graph.initializer if tuple(i.dims) == (len(kept), emb.embedding_dim))
    m = quantize_embedding_rows(m, table)
    # accuracy_level 4 = int8 arithmetic inside the 8-bit layers: ~8x faster on one CPU thread
    # (25 ms vs 206 ms per message) for a very small agreement cost (measured: 98.3% vs 98.9% on 180 sentences)
    q = MatMulNBitsQuantizer(m, bits=8, block_size=32, is_symmetric=True, accuracy_level=4,
                             op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
    q.process()
    q.model.save_model_to_file(str(OUT / "model.onnx"), use_external_data_format=False)
    fp32.unlink()
    (OUT / "meta.json").write_text(json.dumps({**meta, "onnx_file": "model.onnx", "vocab_size": len(kept),
                                               "original_vocab_size": len(tok_json["model"]["vocab"])}, indent=2))
    return {"original_vocab": len(tok_json["model"]["vocab"]), "kept_vocab": len(kept)}


def check_tokenizer() -> dict:
    """Every MASSIVE en/hi/kn sentence must get the same tokens (same strings) from the pruned tokenizer."""
    from tokenizers import Tokenizer

    old = Tokenizer.from_file(str(SRC / "tokenizer" / "tokenizer.json"))
    new = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    old.no_padding()  # the saved original pads batches to the longest sentence; compare real tokens only
    new.enable_truncation(max_length=64)  # same truncation as the original
    total = same = 0
    for lang in LANGS:
        for split in ("train", "dev", "test"):
            texts = [r["text"] for r in load_split(lang, split)]
            for a, b in zip(old.encode_batch(texts), new.encode_batch(texts), strict=True):
                total += 1
                same += a.tokens == b.tokens
    return {"sentences_checked": total, "identical_tokens": same}


PEAK = r"""
import sys; sys.path.insert(0, "src")
from router.onnx_model import OnnxIntentModel
m = OnnxIntentModel("models/onnx"); m.logits(["wake me up at five am"])
print(int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) / 1024)
"""


def evaluate() -> dict:
    from router.onnx_model import OnnxIntentModel

    m = OnnxIntentModel(OUT, threads=4)
    labels = load_labels()
    idx = {name: i for i, name in enumerate(labels)}
    routing = json.loads((RESULTS / "routing.json").read_text())
    T, t = routing["temperature"], routing["chosen_threshold"]["t"]
    model_id, pv = routing["llm"]["model"], routing["llm"]["prompt_version"]
    cache = Cache()
    out: dict = {"temperature": T, "threshold": t, "accuracy": {}, "agreement_with_original": {},
                 "router_test_sub": {}}
    for lang in LANGS:
        test = load_split(lang, "test")
        lg = m.logits([r["text"] for r in test])
        out["accuracy"][f"{lang}_test_full"] = round(float((lg.argmax(1) == [r["label"] for r in test]).mean()), 4)
        for sub in ("dev_sub", "test_sub"):
            rows = load_subset(lang, sub)
            lg = m.logits([r["text"] for r in rows])
            gold = np.array([r["label"] for r in rows])
            out["accuracy"][f"{lang}_{sub}"] = round(float((lg.argmax(1) == gold).mean()), 4)
            ref = load_logits(f"{lang}_{sub}")["logits"]
            out["agreement_with_original"][f"{lang}_{sub}"] = round(float((lg.argmax(1) == ref.argmax(1)).mean()), 4)
            if sub == "test_sub":
                probs = C.softmax(lg, T)
                answers = [cache.get(model_id, pv, "batch", lang, r["id"])["intent"] for r in rows]
                llm = np.array([idx.get(a, -1) if a != INVALID else -1 for a in answers])
                res = R.evaluate_threshold(probs.max(1), probs.argmax(1), llm, gold, t)
                out["router_test_sub"][lang] = {"accuracy": round(res["accuracy"], 4),
                                                "llm_share": round(res["llm_share"], 4)}
        print(lang, {k: v for k, v in out["accuracy"].items() if k.startswith(lang)}, flush=True)
    r = out["router_test_sub"]
    r["pooled"] = {k: round(sum(r[lg][k] for lg in LANGS) / 3, 4) for k in ("accuracy", "llm_share")}

    single = OnnxIntentModel(OUT, threads=1)
    texts = [x["text"] for lang in LANGS for x in load_subset(lang, "test_sub")][:220]
    times = []
    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        single.logits([text])
        if i >= 20:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    out["cpu_latency_single_message_1_thread"] = {"median_ms": round(statistics.median(times), 2),
                                                  "p95_ms": round(times[int(0.95 * len(times)) - 1], 2),
                                                  "n": len(times)}
    out["size_mb"] = {"pytorch_checkpoint": round((SRC / "model.pt").stat().st_size / 1e6, 1),
                      "onnx_small": round((OUT / "model.onnx").stat().st_size / 1e6, 1)}
    # VmHWM = this process's own peak memory; it resets on exec, unlike ru_maxrss
    peak = subprocess.run([sys.executable, "-c", PEAK], cwd=ROOT, capture_output=True, text=True, check=True)
    out["peak_memory_mb_load_and_predict"] = round(float(peak.stdout.split()[-1]), 1)
    return out


def main() -> int:
    res = {"method": "vocabulary pruned to Latin/Devanagari/Kannada tokens; word embeddings int8 per row; "
                     "Linear weights 8-bit (block 32), int8 arithmetic (accuracy_level 4)"}
    if "--eval-only" not in sys.argv:
        res["vocab"] = build()
    res["tokenizer_check"] = check_tokenizer()
    print("tokenizer check:", res["tokenizer_check"], flush=True)
    res.update(evaluate())
    (RESULTS / "quantized.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
