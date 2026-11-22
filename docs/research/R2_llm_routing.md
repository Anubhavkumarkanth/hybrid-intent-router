# R2: LLM fallback (Gemini) and confidence-based routing

All facts below were checked on 2026-11-22. Google moved its Gemini docs: `ai.google.dev/gemini-api/docs/*`
now redirects (301) to `aistudio.google.com/docs/*`. Links below use the new addresses.

---

## A. Gemini API

### A1. Python SDK

- The current official Python SDK is **`google-genai`** (`pip install google-genai`, then `from google import genai`).
  The libraries page lists it as the recommended Python library.
  Source: https://aistudio.google.com/docs/libraries (checked 2026-11-22)
- The old **`google-generativeai`** package is legacy and marked "Not actively maintained". The page says the legacy
  libraries are "deprecated as of November 30th, 2025" and tells you to migrate to `google-genai`.
  Source: https://aistudio.google.com/docs/libraries (checked 2026-11-22)
- Latest `google-genai` version on PyPI: **3.6.1**.
  Source: https://pypi.org/pypi/google-genai/json (checked 2026-11-22)
- Note: the official docs pages now show most examples with a newer `client.interactions.create(...)` API.
  The classic `client.models.generate_content(...)` API is still in the SDK and still documented in the SDK reference
  (https://googleapis.github.io/python-genai/). We use `generate_content` below because it is stable and simple.

### A2. Flash / Flash-Lite models (free tier eligible)

From the models page (https://aistudio.google.com/docs/models, checked 2026-11-22) and the pricing page
(https://aistudio.google.com/docs/pricing, checked 2026-11-22). Every model below shows "Free of charge" in the free tier.

| Model id | Status | Note |
|---|---|---|
| `gemini-3.5-flash-lite` | Stable | Recommended for new projects |
| `gemini-3.1-flash-lite` | Stable | Older Flash-Lite |
| `gemini-3.8-flash` | Stable | Bigger, pricier Flash |
| `gemini-3.6-flash` | Stable | |
| `gemini-3-flash-preview` | Preview | Preview models get tighter limits |
| `gemini-2.5-flash` | Deprecated | Auto-routes to `gemini-3.5-flash-lite` on 2026-12-01 |
| `gemini-2.5-flash-lite` | Not deprecated | Only for users who already used 2.5 models |
| `gemini-3.1-flash-lite-preview` | Shut down | Do not use |

**Recommendation for intent classification: `gemini-3.5-flash-lite`.** It is stable, free-tier eligible, cheap,
and its default thinking level is already "minimal" (near-off), which is what a short classification task wants.
Do NOT pick `gemini-2.5-flash` (it is being redirected on 2026-12-01).

### A3. Free-tier rate limits

- Limits are measured as RPM (requests/minute), TPM (input tokens/minute) and RPD (requests/day). Going over any one gives an error.
- "Rate limits are applied per project, not per API key."
- "Requests per day (RPD) quotas reset at midnight Pacific time."
- Preview / experimental models have stricter limits.
  Source: https://aistudio.google.com/docs/rate-limits (checked 2026-11-22)
- **The exact free-tier RPM / TPM / RPD numbers for `gemini-3.5-flash-lite` are NOT printed on the public docs page.**
  The docs point to the per-project page in AI Studio: https://aistudio.google.com/rate-limit (needs login).
  Check it there for your project and put the numbers in config. Design the code to survive 429s anyway
  (retry with backoff, then fall back to the local model's answer). (checked 2026-11-22)

### A4. Paid-tier price and thinking

Paid prices per 1M tokens, from https://aistudio.google.com/docs/pricing (checked 2026-11-22):

| Model | Input (text) | Output (incl. thinking) |
|---|---|---|
| `gemini-3.5-flash-lite` (recommended) | **$0.30** | **$2.50** |
| `gemini-3.1-flash-lite` | $0.25 | $1.50 |
| `gemini-3-flash-preview` | $0.50 | $3.00 |
| `gemini-3.8-flash` / `gemini-3.6-flash` | $0.75 to 2026-12-31, then $1.50 from 2027-01-01 | $3.75, then $7.50 |
| `gemini-2.5-flash-lite` | $0.10 | $0.40 |

Thinking:
- Thinking tokens are billed. The pricing column is "output price (including thinking tokens)", and the thinking docs say
  "When thinking is turned on, response pricing is the sum of output tokens and thinking tokens." You pay for the full
  thought tokens even if only a summary is returned.
  Source: https://aistudio.google.com/docs/thinking (checked 2026-11-22)
- For 3.x models, thinking is controlled with **`thinking_level`**. For `gemini-3.5-flash-lite` the default is
  "On (minimal)" and the allowed levels are `minimal`, `low`, `medium`, `high`. `minimal` is described as
  "Near-off thinking, lowest latency". The docs do not show a way to turn thinking fully off for this model.
  Source: https://aistudio.google.com/docs/thinking (checked 2026-11-22)
- In the SDK, `types.ThinkingConfig` has both `thinking_level` (MINIMAL / LOW / MEDIUM / HIGH / XHIGH) and the older
  `thinking_budget` (int tokens; "0 is DISABLED. -1 is AUTOMATIC. ... model dependent"). `thinking_budget` is the
  Gemini 2.5-era control; for 3.x use `thinking_level="MINIMAL"`.
  Source: https://github.com/googleapis/python-genai/blob/main/google/genai/types.py (checked 2026-11-22)

### A5. Structured JSON output, timeouts, usage, errors (google-genai)

Facts (SDK reference https://googleapis.github.io/python-genai/ and source
https://github.com/googleapis/python-genai/blob/main/google/genai/types.py , checked 2026-11-22):
- JSON output: set `response_mime_type="application/json"` and pass either `response_schema=` (a Pydantic model or
  `types.Schema`) or `response_json_schema=` (a plain JSON Schema dict).
- Enum output: `response_mime_type="text/x.enum"` with a Python `Enum` class as `response_schema` returns one bare
  enum value as text. With `application/json` you get the same value wrapped in quotes.
- The docs advise not to repeat the schema inside the prompt; it can lower quality.
- The API ignores some schema constraints (e.g. `minLength`, `maxLength`), so always validate the result yourself.
  Source: https://aistudio.google.com/docs/structured-output (checked 2026-11-22)
- `temperature` is a normal `GenerateContentConfig` field. Use `0` for classification.
- Timeout: `types.HttpOptions(timeout=...)` is in **milliseconds**. It can be set on the `Client` or per call via
  `GenerateContentConfig(http_options=...)`.
- Usage: `response.usage_metadata` has `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`,
  `tool_use_prompt_token_count` and `total_token_count` (the sum of those). Any of them can be `None`, so use `or 0`.
- Errors: all API errors are `google.genai.errors.APIError` (has `.code`, `.status`, `.message`).
  HTTP 4xx raises `errors.ClientError`, 5xx raises `errors.ServerError` (both subclass `APIError`).
  A rate limit is therefore `ClientError` with `e.code == 429`.
  Source: https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py (checked 2026-11-22)

Short example:

```python
import os
from typing import Literal
from pydantic import BaseModel
from google import genai
from google.genai import types, errors

class Intent(BaseModel):
    intent: Literal["billing", "tech_support", "cancel", "other"]
    confidence: float

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(timeout=10_000))  # ms

config = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    response_schema=Intent,
    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
)

try:
    resp = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Classify this message: 'I was charged twice this month'",
        config=config,
    )
    result = Intent.model_validate_json(resp.text)   # or resp.parsed
    u = resp.usage_metadata
    in_tok  = u.prompt_token_count or 0
    out_tok = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)  # both billed as output
except errors.ClientError as e:
    if e.code == 429:
        ...  # rate limited: back off / fall back to local model
    else:
        raise
except errors.APIError as e:  # 5xx etc.
    ...
```

### A6. "AQ." API keys

- The official API key page does not mention key prefixes. It talks about "Standard API keys" and
  "Authorization (auth) keys". It shows `genai.Client(api_key="YOUR_API_KEY")` and says the client also reads
  `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment (`GOOGLE_API_KEY` wins if both are set).
  Source: https://aistudio.google.com/docs/api-key (checked 2026-11-22)
- Unofficial sources (community blog and forum) say new AI Studio keys are "Auth keys" starting with `AQ.` instead of the
  older `AIza...`. They say these work on the native Gemini endpoint (which `google-genai` uses) but can fail with
  OpenAI-compatible endpoints and some third-party tools. Some users report `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`
  on some projects. Not confirmed by official docs.
  Sources: https://dev.to/rapls/i-created-a-gemini-api-key-and-got-aq-instead-of-aiza-3dp3 ,
  https://discuss.ai.google.dev/t/aq-format-api-key-returns-401-unauthenticated-error-on-gemini-api/183091 (checked 2026-11-22)
- Practical advice: pass the key as-is with `api_key=` (or set `GEMINI_API_KEY`). Do not check for an `AIza` prefix in code.
  If you get a 401, make a new key in AI Studio and test with one small call.

---

## B. Confidence and routing (background for the README)

### B1. Max softmax probability as confidence (Hendrycks & Gimpel, ICLR 2017)
"A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks."
The paper shows that the highest softmax probability of a classifier is a simple, useful signal: correct predictions
tend to have a higher max probability than wrong or out-of-distribution inputs. The raw value is not a true probability
(networks are often overconfident), but it ranks inputs well enough to decide "trust this answer or not". This is the
basic idea behind our router: if the local model's top probability is below a threshold, send the input to the LLM.
https://arxiv.org/abs/1610.02136 (checked 2026-11-22)

### B2. Temperature scaling and ECE (Guo et al., ICML 2017)
"On Calibration of Modern Neural Networks." The paper finds that modern deep networks are badly calibrated, usually
overconfident. It measures this with Expected Calibration Error (ECE): split predictions into confidence bins and take the
weighted average gap between confidence and accuracy in each bin. The best simple fix was **temperature scaling**:
divide the logits by one learned number T (fit on a validation set) before softmax. It does not change which class is
predicted, so accuracy stays the same, but confidences become much more honest. For us, this makes a confidence
threshold mean something closer to "probability of being right".
https://arxiv.org/abs/1706.04599 (checked 2026-11-22)

### B3. RouteLLM (Ong et al., 2024)
"RouteLLM: Learning to Route LLMs with Preference Data." It trains a router that chooses, per query, between a strong
expensive LLM and a weak cheap one. The routers (e.g. matrix factorization, BERT classifier, similarity-weighted ranking)
are trained on human preference data (Chatbot Arena) plus augmentation, and a threshold sets the cost/quality trade-off.
The authors report large cost savings (over 2x in some settings) while keeping response quality close to the strong
model, and the routers transfer to new model pairs. It supports the idea of a cheap first stage with a tunable threshold.
https://arxiv.org/abs/2406.18665 (checked 2026-11-22)

### B4. Hybrid LLM (Ding et al., ICLR 2024)
"Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing." It routes each query to a small model or a large model
based on predicted query difficulty and a quality level the user picks. A router (a small encoder) predicts whether the
small model's answer will be about as good as the large model's; the threshold can be tuned at test time. The paper
reports up to 40% fewer calls to the large model with no drop in response quality. This is the closest match to our
"local model first, LLM only when unsure" design.
https://arxiv.org/abs/2404.14618 (checked 2026-11-22)
