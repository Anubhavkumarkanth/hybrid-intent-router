# Final report: Hybrid Intent Router

## What was built

- A data pipeline for Amazon MASSIVE 1.1 (English, Hindi, Kannada) with fixed, seeded `dev_sub` (300) and
  `test_sub` (500) subsets that share sentence ids across languages.
- An LLM baseline (`gemini-3.5-flash-lite`, JSON output limited to the 60 labels) with a crash-safe,
  resumable answer cache that is committed, so the analysis runs without an API key.
- A small model: MuRIL-base + our own linear head, trained with our own PyTorch loop on English only.
- Temperature scaling, a confidence router, a dev-only threshold rule, and charts.
- A Flask API (`/health`, `/predict`) with input validation and an LLM-failure fallback, plus a demo page.
- gunicorn + NGINX serving as a normal user (no sudo), and a Locust load test.
- A quality gate (`make verify`: ruff, pytest, number check), README, interview guide, and a narrated video.

## Final results

<!-- RESULTS:routing:START -->
| Language | Small only | LLM only | Router | Oracle | Sent to LLM | Router cost /1000 | LLM-only cost /1000 | Router avg latency | LLM-only latency |
|---|---|---|---|---|---|---|---|---|---|
| English | 88.2% | 86.2% | **90.6%** | 93.4% | 3.8% | $0.0114 | $0.2991 | 89 ms | 1344 ms |
| Hindi | 77.0% | 84.2% | **82.0%** | 90.8% | 13.0% | $0.0389 | $0.2991 | 212 ms | 1344 ms |
| Kannada | 70.6% | 81.8% | **80.0%** | 90.2% | 16.4% | $0.0491 | $0.2991 | 258 ms | 1344 ms |
| All 3 (pooled) | 78.6% | 84.1% | **84.2%** | 91.5% | 11.1% | $0.0331 | $0.2991 | 186 ms | 1344 ms |

test_sub, 500 messages per language. Threshold t = 0.56 was chosen on dev_sub (pooled dev accuracy 83.2%, target 83.2%, 14.3% sent to LLM). Temperature T = 1.342.
<!-- RESULTS:routing:END -->

<!-- RESULTS:savings:START -->
| What (test_sub, all 3 languages) | Value |
|---|---|
| LLM calls avoided | 88.9% |
| LLM cost saved | 88.9% |
| Average latency saved per message | 1158 ms |
| Accuracy change vs LLM only | +0.1 percentage points |
<!-- RESULTS:savings:END -->

Small model alone:

<!-- RESULTS:small:START -->
| Language | Full test (2974) | test_sub (500) | dev_sub (300) |
|---|---|---|---|
| English | 88.1% | 88.2% | 85.7% |
| Hindi | 77.9% | 77.0% | 75.7% |
| Kannada | 70.7% | 70.6% | 69.3% |

Model: `google/muril-base-cased` + linear head, trained on English only (best epoch 12). 237.6M parameters, 950.5 MB on disk. CPU latency for one message (4 threads): median 37.53 ms, p95 48.36 ms.
<!-- RESULTS:small:END -->

LLM alone:

<!-- RESULTS:llm:START -->
| Language | dev_sub (300) | test_sub (500) |
|---|---|---|
| English | 84.3% | 86.2% |
| Hindi | 83.7% | 84.2% |
| Kannada | 84.7% | 81.8% |
| All 3 (pooled) | 84.2% | 84.1% |

Model `gemini-3.5-flash-lite`, prompt `v2_labels_examples`, 20 messages per request. Single-message calls (50 dev_sub items): median latency 1344 ms, p95 9114 ms, on average 890 input and 12.8 output tokens per call. Accuracy single vs batched on the same items: 82.0% vs 82.0%.
<!-- RESULTS:llm:END -->

Load test:

<!-- RESULTS:loadtest:START -->
| Metric | Value |
|---|---|
| Setup | 20 users, spawn 5/s, 2m, through NGINX, small model only |
| Workers | 3 gunicorn workers x 4 torch threads |
| Requests | 4908 |
| Failures | 0 |
| Requests per second | 41.2 |
| p50 latency | 180 ms |
| p95 latency | 260 ms |
| p99 latency | 300 ms |
<!-- RESULTS:loadtest:END -->

**In plain words:** pooled over English, Hindi and Kannada, the router is as accurate as sending everything to
the LLM (84.2% vs 84.1% on test_sub) while sending only 11.1% of messages to it. That avoids 88.9% of LLM calls
and cuts the average time per message from 1,344 ms to 186 ms. In English the router beats both models (90.6%).
In Hindi and Kannada it is about 2 points below the LLM alone (82.0% vs 84.2%, 80.0% vs 81.8%), because one pooled
threshold was chosen; English makes up the difference. The small model alone reached 77.9% (Hindi) and 70.7%
(Kannada) on the full test sets with no training data in those languages, a few points below the published MuRIL
numbers (82.4% / 77.5%) and above the MASSIVE paper's XLM-R base (74.8% / 63.5%).

## Time per phase

Wall-clock times on 22 Nov 2026 (from logs and git commit times). Two WSL restarts cost about 25 minutes.

| Phase | Time | Notes |
|---|---|---|
| 0 Environment + skeleton | 09:27-10:45 | the CUDA PyTorch download took ~30 min |
| 1 Research | 10:06-10:45 | notes with sources in `docs/research/` |
| 2 Data pipeline | 10:45-11:06 | |
| 3 LLM baseline | 11:06-11:37 | resumed after a WSL crash; rate limits on single calls |
| 4 Training + evaluation | 11:18-12:17 | run 1 (5 epochs), one run killed by a WSL restart, final run 14.0 min |
| 5 Calibration + routing | 12:17 | runs in seconds from cached answers and saved logits |
| 6 API + smoke test | 12:17-12:28 | |
| 7 Serving + load test | 12:17-12:28 | includes the worker-deadlock fix |
| 8-11 Quality gate, docs, video | from 12:28 | see `docs/journal/08` to `10` |

## Decisions and reasons

| Decision | Reason |
|---|---|
| MASSIVE 1.1, en-US / hi-IN / kn-IN, official tarball with SHA-256 | Parallel sentence ids across languages; exact reproducibility; CC BY 4.0 |
| dev_sub 300 / test_sub 500, same ids in all languages, stratified, seed 42 | Fits the LLM free tier; languages compared on the same sentences |
| MuRIL-base + own `nn.Linear(768, 60)` | Best published zero-shot Hindi/Kannada among base models; own head is easy to explain |
| Own PyTorch loop, English train only, early stopping on English dev | Keeps it truly zero-shot; every step visible |
| Frozen word embeddings, fp16, batch 32, lr 2e-5 (head 1e-3) | Fits a 4 GB GPU; keeps the multilingual vocabulary intact |
| 15-epoch cap (was 5) | Run 1 was still improving at epoch 5; decided on English dev only |
| gemini-3.5-flash-lite, JSON enum output, prompt v2 (labels + 1 example) | Cheap and fast; v2 won the dev pilot (86.7% vs 80.0%) |
| 20 messages per LLM request for evaluation, single calls for cost | Free-tier friendly; batching did not change accuracy (82.0% vs 82.0% on 50 items) |
| Temperature scaling on full English dev | Standard calibration; no Hindi/Kannada labels used |
| Threshold: fewest LLM calls within 1 pp of LLM-only on pooled dev_sub | Written down before results; always has an answer (t = 1.01) |
| gunicorn 3 sync workers x 4 threads, preload, NGINX as normal user | CPU-bound inference on 12 cores; no sudo needed |
| Piper TTS + Pillow + ffmpeg for the video | Offline, free, reproducible |

## Problems and fixes

1. **HTTP ReadTimeout crashed the LLM run** (`httpx2.ReadTimeout` is not `TimeoutError`): mapped all transport
   errors to a retried error; timeout 30 -> 60 s; regression test.
2. **WSL stopped mid-run and left 2,244 NUL bytes in the answer cache**: `fsync` after every write; loader drops
   crash padding but still fails on any other bad line; regression test. All 700 answers were kept.
3. **429 rate limits / 503 overload from the API**: exponential backoff with jitter (already in place).
4. **Run 1 under-trained** (77.4% en dev, still rising at epoch 5): cap raised to 15, early stopping decides.
5. **WSL restarts killed long jobs** whenever no Windows terminal was attached to WSL. Long jobs now run in a
   visible Ubuntu window that keeps WSL alive; the killed training run was repeated from the start.
6. **`make` hid failures and buffered logs**: `.SHELLFLAGS := -o pipefail -c` and `PYTHONUNBUFFERED=1`.
7. **transformers 5.x tokenizer change** (`vocab_file=` silently ignored, so every word became `[UNK]` in the
   test tokenizer): use `vocab=` and assert no `[UNK]`.
8. **Smoke test used a non-existent test id**: ids are unique across splits; the script now fails loudly.
9. **gunicorn workers deadlocked** (OpenMP pool inherited through `preload_app`; Locust showed 0 requests):
   master single-threaded, workers set 4 threads after fork; `/predict` check in `stack-up`; 30 s Locust timeout.

## What is left for you to do

Nothing was pushed or published. To put it on GitHub (your account, your decision):

```bash
cd ~/hybrid-intent-router
git log --oneline | head            # check the history first
# create an empty repo named hybrid-intent-router on github.com (no README), then:
git remote add origin https://github.com/<your-username>/hybrid-intent-router.git
git push -u origin main
```

- `.env` (your Gemini key), `.venv/`, `models/` (950 MB weights), the raw tarball and `video/*.mp4` are
  gitignored. The video is `video/tutorial.mp4`; upload it to YouTube or attach it to a GitHub release if you
  want it online.
- Optional public demo: the app needs the 950 MB model and about 1-2 GB RAM per worker; a small VM or a
  Hugging Face Space works. Run with `ROUTER_DISABLE_LLM=1` unless you want to pay for LLM calls.
- Keep an Ubuntu window open while long jobs run: WSL shuts down when no terminal is attached.
