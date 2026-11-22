# Phase 5: Calibration and routing

## What we did
- `src/router/calibrate.py`: temperature scaling. One number T divides all logits before the softmax. T is found
  by minimising the negative log-likelihood (NLL) on the **full English dev** logits (2,033 sentences): a coarse
  grid over log(T), then a golden-section search. ECE (expected calibration error, 15 equal-width bins) is
  reported before and after, on English dev and on each language's `dev_sub`.
- `src/router/route.py`: confidence >= t → the small model's answer; otherwise the cached LLM answer.
- `src/router/analysis.py` (`make route`): sweeps t = 0.00 ... 1.00 (step 0.01) plus 1.01 on `dev_sub` (pooled and
  per language), computing accuracy, the share sent to the LLM, LLM cost per 1,000 messages and average latency.
  It picks t with the fixed rule, applies it **once** to `test_sub`, and writes `results/routing.json`,
  `results/routing_curve.csv` and four charts.
- Cost per 1,000 messages = 1,000 x share sent to LLM x (input tokens x $0.30 + output tokens x $2.50) / 1M, using
  the real mean tokens from the single-call test. Average latency = small-model CPU median + share x LLM median.

## Why
- A threshold on raw softmax scores is fragile: fine-tuned networks are usually over-confident.
- The rule (fewest LLM calls within 1 percentage point of LLM-only accuracy on dev) was written in `PLAN.md`
  before any routing numbers existed, so it cannot be bent to make the test results look good.

## Key code (`src/router/route.py`)
```python
def choose_threshold(curve, llm_only_acc, tolerance=0.01):
    target = llm_only_acc - tolerance
    ok = [r for r in curve if r["accuracy"] >= target - 1e-12]
    return min(ok, key=lambda r: (r["llm_share"], -r["accuracy"], r["t"]))
```

## Result
- Temperature T = 1.342 (> 1: the model was over-confident). ECE on full English dev 0.076 → 0.016. On dev_sub:
  English 0.100 → 0.039, Hindi 0.140 → 0.068, Kannada 0.170 → 0.091. Calibration helps every language, but the
  languages the model never trained on stay less well calibrated.
- Threshold (decided on pooled dev_sub): LLM-only 84.2%, target 83.2% → **t = 0.56**, dev accuracy 83.2% with 14.3%
  of messages sent to the LLM. The small model alone (76.9% on dev) does not meet the target, so routing is needed.
- Final, on test_sub with t fixed (`results/routing.json`):

  | | Small only | LLM only | Router | Oracle | Sent to LLM |
  |---|---|---|---|---|---|
  | English | 88.2% | 86.2% | **90.6%** | 93.4% | 3.8% |
  | Hindi | 77.0% | 84.2% | 82.0% | 90.8% | 13.0% |
  | Kannada | 70.6% | 81.8% | 80.0% | 90.2% | 16.4% |
  | All 3 | 78.6% | 84.1% | **84.2%** | 91.5% | 11.1% |

- Pooled, the router is +0.13 percentage points above LLM-only while avoiding 88.9% of LLM calls: LLM cost per
  1,000 messages $0.2991 → $0.0331, average latency 1,344.5 ms → 186.3 ms.
- Honest reading: in English the router beats both models (the small model is better than the LLM there, and the
  LLM fixes some of its unsure cases). In Hindi and Kannada the router is about 2 points below the LLM alone; the
  pooled rule allows this because English makes up the difference. A per-language threshold would trade more
  LLM calls for Hindi/Kannada accuracy.
- The test curve peaks near 86% with about 35-45% of messages sent to the LLM (the two models make different
  mistakes). Our rule deliberately optimises for fewest LLM calls, not maximum accuracy.
- Charts: `accuracy_vs_llm_share.png`, `cost_per_1000.png`, `reliability.png`, `examples_table.png`.
- `pytest tests/test_route.py -q`: passes (t = 0 → nothing to LLM; t above max confidence → everything; hand-made
  accuracy example).

## Errors and fixes
- No errors in this phase. The analysis needs only cached LLM answers and saved logits, so it re-runs in seconds
  without an API key or a GPU.

## How to explain this in an interview
1. I calibrated the small model with temperature scaling, so its confidence can be trusted as a probability.
2. I wrote the threshold rule down before seeing results: fewest LLM calls while staying within one point of LLM-only accuracy on dev.
3. I chose the threshold on dev data and applied it once to test data, so the test numbers are honest.
4. Cost and latency use real token counts and real latencies from single API calls, not guesses.
