# Phase 4: Train the small model (plain PyTorch)

## What we did
- `src/router/model.py`: `IntentClassifier` = pretrained `google/muril-base-cased` encoder (`AutoModel`) +
  dropout + our own `nn.Linear(768, 60)` on the first token's vector. Output shape `[batch, 60]`.
- `src/router/train.py`: our own training loop (no Hugging Face `Trainer`): shuffled `DataLoader`, AdamW
  (encoder lr 2e-5, head lr 1e-3, weight decay 0.01, none on biases/LayerNorm), 10% linear warmup then
  linear decay, gradient clipping at 1.0, fp16 mixed precision on the GPU, seed 42. After every epoch it
  measures **English dev** accuracy, keeps the best checkpoint and stops early after 2 epochs without
  improvement. Trained on **English train only** (11,514 sentences).
- The big word-embedding matrix (197k tokens x 768) is frozen: it saves GPU memory on a 4 GB card and keeps
  the multilingual word vectors that make Hindi/Kannada transfer work.
- Before training: checked token lengths (`logs/04_token_lengths.log`): the longest sentence is 45 tokens,
  99% are under 25, so `max_len = 64` cuts nothing.
- 30-step smoke run (`results/smoke_run.json`): 0.187 s per step on the RTX 3050, estimated 5.6 minutes for
  5 epochs, far under the 4-hour limit, so we kept MuRIL (no need for the backup model).
- `src/router/evaluate.py`: accuracy on the full test sets and the subsets, per-item logits for routing,
  CPU latency for one message (4 threads, 200 messages after 20 warm-up), size on disk.

## Why
- Writing the head and the loop ourselves makes every PyTorch step visible and explainable.
- Early stopping and checkpoint choice use **English dev only**. Using Hindi or Kannada dev would make it
  no longer zero-shot.
- Latency is measured on the CPU with the same thread count as each gunicorn worker, because that is how
  the model is served.

## Key code (`src/router/train.py`)
```python
with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
    logits = model(input_ids, attn)
    loss = loss_fn(logits.float(), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
scaler.step(optimizer); scaler.update(); scheduler.step()
```

## Result
- Training (`results/training_log.json`, `logs/04_train.log`): 237.6M parameters, 86.1M trainable (embeddings
  frozen). English dev accuracy per epoch: 28.8, 54.3, 69.2, 79.2, 82.9, 85.2, 86.3, 86.8, 87.5, 87.8, 87.7,
  **88.0 (epoch 12, best)**, 87.9, 87.8 → early stopping after epoch 14. 14.0 minutes on the RTX 3050.
- Evaluation (`results/small_model.json`, `logs/04_evaluate.log`):

  | | English | Hindi | Kannada |
  |---|---|---|---|
  | Full test (2,974 each) | 88.1% | 77.9% | 70.7% |
  | test_sub (500) | 88.2% | 77.0% | 70.6% |
  | dev_sub (300) | 85.7% | 75.7% | 69.3% |

- Published zero-shot MuRIL on MASSIVE (IndicXTREME): 82.4% Hindi, 77.5% Kannada. We are about 4.5 and 7 points
  lower; that paper selected checkpoints differently and trained differently, and we used English dev only.
  We beat the MASSIVE paper's XLM-R base on both (74.8% / 63.5%).
- On English the small model (88.2% on test_sub) is **better** than the LLM (86.2%): it was trained on exactly
  this kind of data.
- CPU latency for one message (4 threads, 200 messages): median 37.5 ms, p95 48.4 ms. Model file 950.5 MB.
- `pytest tests/test_model.py -q`: 4 passed (shape [batch, 60], a tiny batch is overfit, save/load identical).
- A WSL restart killed one 15-epoch run during epoch 14; the run was simply repeated from the start (seeded, it
  reproduced the same losses and accuracies).

## Errors and fixes
1. Run 1 was under-trained. With the plan's 5-epoch cap, English dev accuracy was still rising at the
   last epoch (42.9% → 61.1% → 71.5% → 75.6% → 77.4%; `results/training_log_run1_5epochs.json`). Fix (a
   decision made on English dev only): raise the cap to 15 epochs and let early stopping on English dev decide.
   Run 1's log and curve are kept for honesty.
2. Make hid failures and progress. `python ... | tee log` returns tee's exit code, so a crashed command
   looked successful; fixed with `.SHELLFLAGS := -o pipefail -c`. Python also buffers its output when it is
   piped, so the log stayed empty until the end; fixed with `PYTHONUNBUFFERED=1` in the Makefile.
3. A lint warning (`F821`) came from `del model` inside a function that also has a closure using `model`;
   replaced with `model = None` to free GPU memory before the CPU latency test.

## How to explain this in an interview
1. I fine-tuned MuRIL, a multilingual encoder, with my own linear head and my own PyTorch training loop.
2. I trained only on English and picked the checkpoint with English dev accuracy, so Hindi and Kannada are true zero-shot tests.
3. I froze the word embeddings: it fits a 4 GB GPU and keeps the shared multilingual vocabulary intact.
4. A 30-step smoke run told me the full training would take minutes, so I did not need the smaller backup model.
5. My first run stopped too early because the dev accuracy was still climbing, so I raised the epoch cap and let early stopping decide.
