# Phase 3: LLM baseline ("send everything to the LLM")

## What we did
- `src/router/llm.py`: a Gemini client (official `google-genai` SDK, `generate_content`) that returns one of
  the 60 intents. The answer is forced into JSON with a schema whose `intent` field is an enum of the 60
  labels, and the code still checks the label (an unknown label is retried once, then counted as `invalid`).
- Every answer is appended to `results/llm_cache.jsonl`, keyed by (model, prompt version, mode, language, id).
  A run that stops half way simply continues: cached keys are never sent to the API again.
- Prompt pilot on 60 `dev_sub` items (20 per language): v1 = label list only, v2 = label list + one short
  English train example per intent. The better one was frozen in `results/llm_prompt_version.json`.
- Full run: frozen prompt on `dev_sub` (300) and `test_sub` (500) in en, hi, kn = 2,400 classifications,
  20 messages per request (about 120 requests), to stay inside the free tier.
- Single-message test: 50 `dev_sub` items, one request each, to measure the real latency and tokens per call.

## Why
- JSON with an enum makes the answer machine-readable and almost always a valid label.
- Batching keeps us far below the free-tier daily request limit; single calls give the honest cost and latency
  a production bot would see.
- The cache makes results reproducible without an API key and protects the free quota.

## Key code
```python
# src/router/llm.py: transient network errors are retried with exponential backoff + jitter
delay = base_delay * (2**attempt) + random.uniform(0, 1)
...
except _network_errors() as e:          # httpx / httpx2 TransportError, TimeoutError, ConnectionError
    raise TransientError(f"{type(e).__name__}: {e}"[:300]) from e
```

## Result
- Pilot (`results/llm_pilot.json`): v1 labels only 80.0% pooled, v2 labels + examples 86.7% → v2 frozen.
- Baseline (`results/llm_baseline.json`, 20 messages per request, 120 requests, 0 invalid answers):

  | | English | Hindi | Kannada | Pooled |
  |---|---|---|---|---|
  | dev_sub (300) | 84.3% | 83.7% | 84.7% | 84.2% |
  | test_sub (500) | 86.2% | 84.2% | 81.8% | 84.1% |

- Single messages (`results/llm_single_call.json`, 50 dev_sub items, one request each): median latency
  1,344.5 ms, p95 9,113.5 ms (some calls were slow while the API returned 503 "overloaded" errors), on average
  890.3 input and 12.8 output tokens per call (0 thinking tokens). Accuracy single vs batched on the same 50
  items: 82.0% vs 82.0% (98% of answers identical), so batching did not change the quality.
- The cache holds exactly 2,510 answers: 2,400 batched + 60 pilot (v1) + 50 single calls.
- `pytest tests/test_llm.py -q`: 20 passed (fake backend, no network).

## Errors and fixes
1. **ReadTimeout crashed the run.** The SDK's HTTP layer (`httpx2`) raised `ReadTimeout`, which is not the
   builtin `TimeoutError`, so it skipped our retry logic and stopped the whole run
   (`logs/03_llm_run_crash1.log`). Fix: map all HTTP transport errors to `TransientError` (retried with
   backoff) and raise the request timeout from 30 s to 60 s. Regression test: `test_network_timeout_is_retried`.
2. **WSL stopped mid-run and corrupted the cache.** At 11:11 the WSL virtual machine stopped (after a network
   error) while the run was appending answers. The cache file ended with 2,244 zero bytes (`\x00`), and the
   next run crashed with `JSONDecodeError` on load. All 700 real answers were intact. Fix at the root:
   (a) the cache now `flush()`es and `fsync()`s after every write, so answers we already paid for reach the
   disk; (b) the loader drops NUL padding (the signature of a crash) but still raises on any other broken
   line, so real corruption is never hidden. Regression test: `test_cache_survives_crash_padding_but_not_corruption`.
3. Several `503 Service Unavailable` answers from the API: handled by the existing backoff (`retry 1/5 ...`
   lines in the logs), no code change needed.

## How to explain this in an interview
1. The LLM baseline answers "what if we just send everything to the LLM?", so I can compare the router against it.
2. I forced the LLM to answer in JSON with a fixed list of labels, and I still validate every answer.
3. Every answer is cached by model, prompt, language and sentence id, so runs are resumable and reproducible without a key.
4. I picked the prompt on dev data only, then measured single calls separately to get honest cost and latency.
5. When the machine crashed mid-run, I fixed the cache so it syncs every write to disk, and added a test for it.
