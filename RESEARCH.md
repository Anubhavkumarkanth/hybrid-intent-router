# RESEARCH

All facts checked on **2026-11-22**. Full notes, with a source link
next to every fact, are in:

- `docs/research/R1_data_model_recipe.md` (questions 1–3)
- `docs/research/R2_llm_routing.md` (questions 4–5)
- `docs/research/R3_serving_video.md` (questions 6–8)

This file is the short summary. Every line below has its source.

---

## 1. Dataset: Amazon MASSIVE 1.1

- Official tarball: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
  (40,251,390 bytes). Data license CC BY 4.0, code Apache-2.0.
  Sources: https://github.com/alexa/massive , https://github.com/alexa/massive/blob/main/LICENSE.txt (checked 2026-11-22)
- Files `1.1/data/{locale}.jsonl`, fields `id, locale, partition, intent, utt`. Each locale has
  train 11,514 / dev 2,033 / test 2,974 and 60 intents. `id`s are parallel across locales.
  We re-checked this ourselves in Phase 2 (`tests/test_data.py`).
- Version 1.1 only adds Catalan, so hi-IN and kn-IN are the same as in the paper.
- **Reference numbers (zero-shot from English, intent accuracy, %):**

  | Source | Model | hi-IN | kn-IN |
  |---|---|---|---|
  | MASSIVE paper, App. D, Table 8 (https://arxiv.org/abs/2204.08582) | XLM-R base | 74.8 ± 1.6 | 63.5 ± 1.7 |
  | MASSIVE paper, App. D, Table 8 | mT5 text-to-text | 62.4 | 56.7 |
  | IndicXTREME, Table 19 (https://arxiv.org/abs/2212.05409) | MuRIL | 82.4 | 77.5 |
  | IndicXTREME, Table 19 | XLM-R | 74.9 | 61.7 |
  | IndicXTREME, Table 19 | IndicBERT v2 | 82.7 | 78.2 |

  Note: the MASSIVE paper picked checkpoints with dev data from all locales and trained intent + slots
  together. We pick with **English dev only**, so our numbers may be a bit lower. (checked 2026-11-22)

## 2. Small model choice

| | google/muril-base-cased | FacebookAI/xlm-roberta-base |
|---|---|---|
| Params | ~236M | ~278M |
| Vocab | 197,285 | 250,002 |
| Languages | 17 Indian langs + English (hi, kn included) | 100 langs (hi, kn included) |
| License | Apache-2.0 | MIT |
| Zero-shot MASSIVE (IndicXTREME) hi / kn | 82.4 / 77.5 | 74.9 / 61.7 |

Sources: https://huggingface.co/google/muril-base-cased , https://arxiv.org/abs/2103.10730 ,
https://huggingface.co/FacebookAI/xlm-roberta-base , https://arxiv.org/abs/1911.02116 (checked 2026-11-22)

- CPU speed: both are 12-layer, hidden 768 encoders, so per-token compute is about the same. The big
  vocabularies only affect the embedding lookup (cheap), not the compute.
- **Primary: MuRIL-base** (built for Indian languages, trained with translation and transliteration
  pairs, clearly better on Kannada in published zero-shot results).
- **Backup: microsoft/Multilingual-MiniLM-L12-H384** (~117M params, MIT, about a quarter of the compute).
  https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384 . distilbert-multilingual was rejected:
  mBERT scores very low zero-shot on hi/kn in IndicXTREME.

## 3. Training recipe

- max_len 64 with dynamic padding (MASSIVE sentences are short), batch 32, AdamW, lr 2e-5,
  weight decay 0.01, 10% linear warmup then linear decay, 3–5 epochs, gradient clipping 1.0,
  fp16 mixed precision on GPU, best checkpoint by English dev accuracy.
- Memory: full AdamW fine-tuning of a base model needs about 4 GB+, too much for a 4 GB laptop GPU.
  **Freeze the word-embedding matrix** (the MASSIVE authors also froze word embeddings for their
  zero-shot XLM-R run). This also keeps the multilingual word vectors unchanged.
- Risk: English dev accuracy is not always a good guide to target-language accuracy
  (Keung et al., EMNLP 2020, https://aclanthology.org/2020.emnlp-main.40 ; Gaschi et al. 2023,
  https://arxiv.org/abs/2306.02790). So keep epochs low and lr small. We still choose only on English dev
  (rule 2), and report hi/kn only as results.
- Time estimate (not measured): about 1–2 minutes per epoch on the RTX 3050; 8–20 minutes per epoch on CPU.
  We measure it with a 30-step smoke run in Phase 4.

## 4. LLM for the baseline and fallback (Gemini)

- SDK: **`google-genai`** (`from google import genai`), version 3.6.1 on PyPI. The old
  `google-generativeai` is deprecated (2025-11-30). https://aistudio.google.com/docs/libraries (checked 2026-11-22)
- Model: **`gemini-3.5-flash-lite`** (stable, free-tier eligible, default thinking level "minimal").
  https://aistudio.google.com/docs/models (checked 2026-11-22)
- Paid price for this model: **$0.30 per 1M input tokens, $2.50 per 1M output tokens** (output includes
  thinking tokens). https://aistudio.google.com/docs/pricing (checked 2026-11-22)
- JSON output: `GenerateContentConfig(response_mime_type="application/json", response_json_schema=...)`.
  Always validate the answer yourself. Timeout `HttpOptions(timeout=...)` is in milliseconds.
  Tokens from `usage_metadata.prompt_token_count / candidates_token_count / thoughts_token_count`.
  https://aistudio.google.com/docs/structured-output , https://googleapis.github.io/python-genai/ (checked 2026-11-22)
- Errors: 4xx -> `errors.ClientError`, 5xx -> `errors.ServerError`; rate limit = `ClientError` with code 429.
  https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py (checked 2026-11-22)
- Rate limits are per project, not per key. Daily limits (RPD) reset at midnight Pacific time.
  Exact free-tier numbers are only shown in AI Studio after login.
  https://aistudio.google.com/docs/rate-limits (checked 2026-11-22)
- "AQ." keys work with the native endpoint (which `google-genai` uses). Not in official docs; community reports only.

## 5. Confidence and routing

- **Max softmax probability** as a confidence signal: Hendrycks & Gimpel, ICLR 2017,
  https://arxiv.org/abs/1610.02136
- **Temperature scaling and ECE**: Guo et al., ICML 2017, https://arxiv.org/abs/1706.04599 .
  Divide logits by one number T fit on a validation set; accuracy unchanged, confidences more honest.
- **RouteLLM**: Ong et al., 2024, https://arxiv.org/abs/2406.18665 . Learned routers between a strong and a weak LLM.
- **Hybrid LLM**: Ding et al., ICLR 2024, https://arxiv.org/abs/2404.14618 . Route easy queries to a small model,
  with a quality/cost knob.
- Our router is the simplest version of this idea: a threshold on the calibrated confidence. (all checked 2026-11-22)

## 6. Serving

- gunicorn's `(2 x cores) + 1` is only a starting point. For ML, RAM limits workers: each worker
  holds a model copy (~1 GB fp32 for MuRIL). https://gunicorn.org/design/ (checked 2026-11-22)
- Use sync workers for CPU-bound work behind a proxy. Set `torch.set_num_threads(cores // workers)` in each
  worker to avoid oversubscription. https://docs.pytorch.org/docs/2.15/notes/multiprocessing.html (checked 2026-11-22)
- `preload_app` shares memory (copy-on-write) but forking after torch thread pools start can deadlock
  (https://github.com/benoitc/gunicorn/issues/2478). We load weights in the master but never run inference
  there, and set threads in `post_fork`.
- Plan: **3 sync workers x 4 threads** on 12 cores / 7.4 GB RAM; timeout 120 s.
- NGINX as a normal user: `nginx -p <prefix> -c <conf> -e <errorlog>` (`-e` since 1.19.5), `pid` and all five
  `*_temp_path`s inside the project, no `user` directive. Stop with the same `-p/-c` plus `-s stop`.
  Tested on this machine. https://nginx.org/en/docs/switches.html (checked 2026-11-22)

## 7. Load testing

- `locust -f deploy/locustfile.py --headless -u 20 -r 5 -t 2m --host http://127.0.0.1:8080 --csv results/loadtest --only-summary`
- `_stats.csv`: row `Name == "Aggregated"`, columns `50%`, `95%` (ms), `Requests/s`, `Failure Count`.
  https://docs.locust.io/en/stable/configuration.html , https://github.com/locustio/locust/blob/master/locust/stats.py (checked 2026-11-22)

## 8. Video tooling

- Piper TTS 1.8.0: `PiperVoice.load`, `voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=...))`.
  https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
- Pillow: wrap text with `ImageDraw.textlength`, measure with `multiline_textbbox`, shrink font until it fits
  (`textsize` was removed in Pillow 10). https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
- Pygments `ImageFormatter(font_name="DejaVu Sans Mono", line_numbers=True, ...)`; finds fonts via `fc-list`.
  https://pygments.org/docs/formatters/
- ffmpeg: `-loop 1 -i img -i audio -c:v libx264 -tune stillimage -pix_fmt yuv420p -r 30 -c:a aac -shortest`,
  join with `-f concat -safe 0 -i list.txt -c copy`; `-af volumedetect` prints `mean_volume`.
  https://ffmpeg.org/ffmpeg-formats.html#concat-1 , https://ffmpeg.org/ffmpeg-filters.html#volumedetect
- SRT: index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text, blank line. https://en.wikipedia.org/wiki/SubRip
  (all checked 2026-11-22)
