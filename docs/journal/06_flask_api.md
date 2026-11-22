# Phase 6: Flask API and demo page

## What we did
- `app/app.py` (Flask): `GET /health` → `{"status": "ok"}`; `POST /predict` with `{"text": "..."}` →
  `{"intent", "confidence", "routed_to", "latency_ms"}`; `GET /` → demo page.
- The model is loaded once at start-up (`TorchSmallModel`), the temperature T and threshold t are read from
  `results/routing.json`, and confidence = softmax(logits / T).max().
- Input validation: not JSON / no string `text` → 400; empty text → 400; more than 500 characters → 400,
  each with a clear error message.
- If the small model is unsure and the LLM call fails or times out (10 s timeout, 2 tries), the app returns the
  small model's answer with `routed_to = "small_fallback"`. It never crashes.
- `ROUTER_DISABLE_LLM=1` turns the LLM off (used for the load test, so no API quota is spent).
- `app/templates/index.html`: a text box, English / Hindi / Kannada example buttons, and the JSON result.
- `scripts/smoke_api.sh` (`make smoke-api`): starts the Flask dev server and sends real requests with `curl`.

## Why
- `create_app()` takes the model, the LLM function and the threshold as arguments, so tests can pass a fake
  small model and a fake LLM: no network, no model download, and the failure path is easy to test.
- A fallback answer is better for a voice bot than an error message.

## Key code (`app/app.py`)
```python
intent, conf = app.config["SMALL"].predict(text)
routed_to = "small"
if conf < app.config["THRESHOLD"]:
    llm = app.config["LLM_FN"]
    if llm is None:
        routed_to = "small_fallback"
    else:
        try:
            intent = llm(text)
            routed_to = "llm"
        except Exception as e:  # any LLM failure -> keep the small model's answer
            log.warning("LLM failed, using small model answer: %s: %s", type(e).__name__, e)
            routed_to = "small_fallback"
```

## Result
- `pytest tests/test_api.py -q`: passes (health, valid predict, 400 cases, LLM failure → `small_fallback`).
- `make smoke-api` (`logs/06_smoke.log`, LLM on, threshold 0.56), MASSIVE test ids 0 and 8 in all three languages:

  | Sentence | Confidence | Routed to | Answer | Latency |
  |---|---|---|---|---|
  | en "wake me up at five am this week" | 0.913 | small | alarm_set (correct) | 111.4 ms (first request) |
  | hi, same sentence | 0.862 | small | alarm_set (correct) | 37.0 ms |
  | kn, same sentence | 0.273 | llm | alarm_set (correct) | 1,576.1 ms |
  | en "pink is all we need" | 0.778 | small | iot_hue_lightchange (correct) | 149.4 ms |
  | hi, same sentence | 0.825 | small | iot_hue_lightchange (correct) | 34.6 ms |
  | kn, same sentence | 0.170 | llm | lists_createoradd (**wrong**) | 8,145.0 ms (API overloaded, retried) |

  Empty text and 501 characters both returned HTTP 400 with a clear message.
- This shows the whole idea in six requests: confident messages are answered in tens of milliseconds, unsure
  Kannada ones go to the LLM, which is slower and not always right either.

## Errors and fixes
- The first smoke script used test id `1`, which does not exist: MASSIVE ids are unique across train, dev and
  test, so the request body was empty and the API (correctly) returned 400. Fix: use test id `8`, and make the
  script exit with an error when an id is not found instead of sending an empty request.
- The fallback pill on the demo page was recoloured from amber to purple (style only).

## How to explain this in an interview
1. The API has two endpoints: health for monitoring, and predict, which returns the intent, the confidence, who answered and the latency.
2. I validate input at the boundary: empty or too-long text gets a clear 400 error, never a crash.
3. If the LLM fails or times out, the user still gets the small model's answer, marked small_fallback.
4. The app is built by a factory function, so tests inject a fake model and a fake LLM and run without network.
