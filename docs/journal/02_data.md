# Phase 2: Data pipeline

## What we did
`make data` (code: `src/router/data.py`) downloads the MASSIVE 1.1 tarball, checks its SHA-256,
reads only en-US, hi-IN and kn-IN from inside the tarball, and writes
`data/processed/{lang}_{split}.jsonl` with `id, text, intent, label`. It also writes `labels.json`
(60 intents, sorted) and two fixed evaluation subsets: `dev_sub` (300 dev ids) and `test_sub` (500 test ids).

## Why
- SHA-256 check: we know we always use the exact same data.
- One label list from English train: label 7 means the same intent in every language.
- Subsets: the LLM free tier is limited, so the LLM sees 300 + 500 sentences per language, not 5,000.
  We use the **same ids** in all three languages, so English, Hindi and Kannada results are about the same
  sentences (MASSIVE ids are translations of each other).
- Stratified sampling: each intent gets a share proportional to its size, so rare intents still appear.

## Key code (`src/router/data.py`)
```python
quotas = {i: k * len(ids_by_intent[i]) / total for i in intents}   # proportional share per intent
alloc = {i: int(quotas[i]) for i in intents}                         # then largest remainder gets +1
...
chosen.extend(rng.sample(pool, alloc[i]))                            # seeded random pick (seed 42)
```

## Result
- 11,514 / 2,033 / 2,974 rows per language (train / dev / test), 60 intents, ids parallel. (`logs/02_data.log`)
- dev_sub: 300 items, 56 intents covered; test_sub: 500 items, 57 intents covered.
- Re-running `make data` gives byte-identical subset files (sha256 check in `logs/02_data.log`).
- `pytest tests/test_data.py -q`: 19 passed.
- Tarball SHA-256 `4cba5faa…0491577` recorded in `data/README.md`.

## Errors and fixes
- No code errors. The S3 download was very slow at one point, so the code skips
  the download when the file is present and the hash matches.

## How to explain this in an interview
1. I used the official MASSIVE tarball and checked its SHA-256, so the data is exactly reproducible.
2. MASSIVE sentences are translations with the same id, so I can test Hindi and Kannada on the same sentences as English.
3. I made small fixed subsets with a seed for the LLM, because the free API tier is limited.
4. The dev subset is only for decisions; the test subset is only for final numbers. Tests check they never overlap with each other or with train.
