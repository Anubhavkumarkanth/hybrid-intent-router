# Deployment: a live demo on a free server

## What I did
- Looked for free hosting. Hugging Face now needs a paid plan for Docker Spaces (the upload was refused with
  "402 Payment Required"), and the free plans of Render and similar hosts give 512 MB of RAM. The full model
  needs about 1.5 GB to load, so it had to get smaller.
- Wrote `scripts/quantize.py` (`make quantize`), which makes a 167.6 MB ONNX copy of the model and measures it.
- Added an ONNX path to the app (`ROUTER_ONNX_DIR`): onnxruntime + tokenizers, no PyTorch at all.
- Added a per-minute cap on Gemini calls (`ROUTER_LLM_MAX_PER_MIN`), so a public demo can't use up the API quota.
- Uploaded the small model to a free Hugging Face model repo, and added `render.yaml`: Render installs only
  `deploy/render/requirements.txt`, downloads the model, checks its SHA-256 and starts gunicorn.
- Tested a copy of the Render build from a clean clone before creating the real service, then tested the live
  site: health check, English/Hindi/Kannada predictions, an unsure message going to Gemini, the 400 errors and
  the demo page in a browser.

## How the model got small without losing accuracy
- MuRIL's vocabulary has 197,258 tokens in 17 scripts. I kept the 90,022 Latin, Devanagari and Kannada tokens.
  WordPiece can only use vocabulary pieces that appear in the word, so English, Hindi and Kannada text gets the
  same tokens; the script checks this on all 49,563 MASSIVE en/hi/kn sentences (all identical).
- The word-embedding table is stored as int8 with one scale per row (a Gather -> Cast -> Mul rewrite).
- Linear layers have 8-bit weights in blocks of 32 (`MatMulNBits`) with int8 arithmetic (`accuracy_level 4`).

## Result
The details are in the README table rendered from `results/quantized.json`. In short:
English / Hindi / Kannada full test 88.2% / 77.7% / 70.7% against 88.1% / 77.9% / 70.7% for the original, the
same answer as the original on 98.2-99.8% of test_sub, the router at 83.8% with 10.8% sent to the LLM, 26 ms per
message on one thread, and 313 MB to load and predict. In the simulated Render build the gunicorn worker peaked at
357 MB. Live, the small model answered in 33-228 ms on the server, and an unsure Kannada message went to Gemini
and came back correct.

## Errors and fixes
- The first quantization (standard dynamic INT8) gave 14-18% accuracy. A diagnosis showed the tokenizer
  and the fp32 export were exact, and the damage came from quantizing activations; per-channel weights only lifted
  agreement to 82%. Weight-only 8-bit layers kept 99.4%, but at 8 bits the quantizer left the 606 MB embedding
  table in float. 4-bit embeddings cost too much (92-94%), so I pruned the vocabulary and wrote my own 8-bit
  per-row embedding rewrite instead (unit test in `tests/test_quantize.py`).
- My first memory numbers said 4.5-6 GB. That was wrong: `ru_maxrss` survives `exec`, so a child started from a
  big parent reports the parent's peak. Measured with the child's own high-water mark (`VmHWM`), the real peak
  was 313 MB.
- The weight-only model took 206 ms per message on one thread. Switching its layers to int8 arithmetic made it
  25 ms, for a very small agreement cost (98.3% vs 98.9% on a 180-sentence sample).
- My tokenizer check first reported only 8 of 49,563 sentences identical. The saved original tokenizer pads
  batches to the longest sentence, so the comparison counted padding; with padding off, all 49,563 match.
- The Hugging Face token was saved with a note after it in the file, which gave "Invalid user token"; the upload
  script now reads only the first word.

## How to explain this in an interview
1. The free server had 512 MB, and my model needed about 1.5 GB, so I shrank it to 168 MB and measured that the answers barely changed.
2. I cut the vocabulary to the three scripts the app supports and proved on every sentence that tokenization didn't change.
3. Standard INT8 quantization broke the model, so I diagnosed it step by step and switched to 8-bit weight-only layers plus my own 8-bit embedding table.
4. The live app runs without PyTorch, downloads the model with a checksum check, and caps LLM calls so a public demo can't drain the API key.
