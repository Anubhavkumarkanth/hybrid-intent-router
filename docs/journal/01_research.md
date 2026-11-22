# Phase 1: Research

## What we did
The research is in three notes, each fact with its source and the date it was checked:
`docs/research/R1_data_model_recipe.md`, `R2_llm_routing.md`, `R3_serving_video.md`.
The short summary is `RESEARCH.md`; the final decisions are in `PLAN.md`.

## Why
Choosing the model, LLM, price and server setup from real sources (not memory) keeps the project honest.

## Key decisions
- **Small model: MuRIL-base** (`google/muril-base-cased`). Published zero-shot MASSIVE intent accuracy
  (IndicXTREME, Table 19): MuRIL hi 82.4 / kn 77.5 vs XLM-R hi 74.9 / kn 61.7. Backup: Multilingual-MiniLM.
- **Reference points**: MASSIVE paper (Table 8) XLM-R base zero-shot: hi-IN 74.8, kn-IN 63.5.
- **Recipe**: lr 2e-5, batch 32, max_len 64, warmup 10%, up to 5 epochs, early stop on English dev,
  fp16, word embeddings frozen (a 4 GB GPU cannot hold full AdamW state for a 236M-parameter model).
- **LLM**: `gemini-3.5-flash-lite` via the `google-genai` SDK, JSON schema output, temperature 0.
  Paid price $0.30 / 1M input tokens, $2.50 / 1M output tokens (checked 2026-11-22).
- **Serving**: gunicorn 3 sync workers x 4 torch threads; NGINX as a normal user on port 8080.

## Errors and fixes
- Google moved the Gemini docs from ai.google.dev to aistudio.google.com; links were updated.
- The exact free-tier rate limits are only visible after login, so the code is built to survive 429 errors
  (backoff + resumable cache) instead of relying on a number.

## How to explain this in an interview
1. I compared two multilingual encoders using published zero-shot results; MuRIL was clearly better on Kannada.
2. A 236M-parameter model with AdamW needs more than 4 GB of GPU memory, so I froze the word embeddings, which the MASSIVE authors also did.
3. I chose the cheapest stable Gemini model and recorded its paid price with the date, because prices change.
4. Every fact has a source link and the date it was checked.
