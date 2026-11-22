---
license: apache-2.0
language:
- en
- hi
- kn
base_model: google/muril-base-cased
datasets:
- AmazonScience/massive
pipeline_tag: text-classification
tags:
- intent-classification
- onnx
- zero-shot-cross-lingual
---

# Hybrid Intent Router: small ONNX model

The small model from [Hybrid Intent Router](https://github.com/Anubhavkumarkanth/hybrid-intent-router): MuRIL-base
with a linear head over the 60 MASSIVE intents, fine-tuned on English only and used zero-shot for Hindi and
Kannada. This copy is shrunk to run on a free 512 MB server with onnxruntime, without PyTorch.

How it was shrunk (`scripts/quantize.py` in the repo):

- The vocabulary keeps only Latin, Devanagari and Kannada tokens (90,022 of 197,258). English, Hindi and
  Kannada text gets exactly the same tokens; this was checked on all 49,563 MASSIVE en/hi/kn sentences.
- The word-embedding table is stored as int8 with one scale per row.
- Every Linear layer has 8-bit weights in blocks of 32 (ONNX `MatMulNBits`) with int8 arithmetic.

Result: 167.6 MB instead of 950.5 MB, about 313 MB of memory to load and predict, and about 26 ms per message
on one CPU thread. On the full MASSIVE test sets it scores 88.2% (English), 77.7% (Hindi) and 70.7% (Kannada),
against 88.1%, 77.9% and 70.7% for the original, and it gives the same answer as the original on 98.2% to
100% of the evaluation subsets.

Files: `model.onnx` (inputs `input_ids`, `attention_mask`; output `logits` over the labels in
`data/processed/labels.json` of the repo), `tokenizer.json` (the pruned tokenizer; truncate to 64 tokens),
`meta.json`. Apply softmax(logits / 1.342) for the calibrated confidence used by the router.

Trained on Amazon MASSIVE 1.1 (CC BY 4.0). Base model: google/muril-base-cased (Apache-2.0).
