# Phase 8: Quality gate and independent review

## What we did
- `make verify` = `ruff check .` + the full `pytest -q` + `scripts/check_numbers.py` (every RESULTS section in the
  README, the final report and the interview guide must equal what `scripts/render_tables.py` would write now).
- An independent review checked the repo against `PLAN.md` and the
  ground rules: test-set leakage, numbers not backed by `results/`, secrets in git history, broken run
  instructions, missing tests.
- Fresh-clone test (`logs/08_fresh_clone.log`): `git clone` into `/tmp`, new venv, CPU PyTorch, requirements,
  `make data`, `make verify` - exactly the README quick start, without an API key or a GPU.
- Secret scan over the whole history: `git log -p --all | grep -iE "AIza...|AQ\...."` finds only documentation text
  (the research notes mention the key formats), never a key. `.env` was never tracked; it is now
  `chmod 600` (only the owner can read it).

## Review findings and what I did
| Finding | Fix |
|---|---|
| No test leakage, no secrets, ~40 hand-written numbers spot-checked: all correct; T and t recomputed by hand: identical | - |
| "Average latency" was built from the **median** LLM call (1,344.5 ms); the mean was 3,133.4 ms | Added `latency_ms_mean` to `llm_single_call.json` (recomputed from the cache, no API calls); tables and prose now say "typical (median-based)" and show the mean |
| Nothing checked that `results/` matches the code | New `tests/test_results.py` recomputes T, t and the pooled test result from the committed logits + cache |
| `read_settings` (threshold from `routing.json`) untested | New API test |
| Sync workers block during LLM calls; load test was small-model only | Added to the README limitations |
| Data tests did not skip cleanly in a fresh clone | Skip now checks `en_train.jsonl`; README says `make verify` runs after `make data` |
| "1,500" and "102 candidates" typed into the generated text | Now read from `routing.json` / `routing_curve.csv` |
| Time table not labelled UTC; curve peak "about 40%" | Labelled UTC; peak is 34-43% |
| `PLAN.md` listed a non-existent `tests/test_numbers.py`; `.env` readable by all users | Fixed; `chmod 600 .env` |

## Result
- `make verify` (`logs/08_verify.log`): ruff clean; all tests pass; `check_numbers: OK` (15 result sections).
- Fresh clone: `== FRESH CLONE TEST PASSED` (first attempt failed on a PyTorch download timeout; the README and
  the script now use `pip --timeout 120 --retries 10`).

## Errors and fixes
- The fresh-clone install timed out downloading PyTorch (network, not code). Fix: pip timeout + retries.

## How to explain this in an interview
1. One command, make verify, runs lint, every test and a check that the documents match the results files.
2. I had the repo reviewed independently for test leakage, unsupported numbers and secrets; the review found labelling issues, which I fixed.
3. The most useful catch: my "average latency" was really median-based; I now label it and report the mean too.
4. I cloned the repo into an empty folder and followed my own README to prove the instructions work.
