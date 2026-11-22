# PLAN

Final decisions (from RESEARCH.md, decided 2026-11-22). Anything chosen later from dev data is
recorded in `results/` and in the phase journals (`docs/journal/`).

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Data | MASSIVE 1.1 tarball, en-US / hi-IN / kn-IN only | Official, CC BY 4.0, parallel ids |
| Eval subsets | `dev_sub` = 300 dev ids, `test_sub` = 500 test ids, seed 42, stratified by intent, same ids in all 3 languages | Fits LLM free-tier quota; parallel ids make languages comparable |
| Small model | **google/muril-base-cased** + our own `nn.Linear(768, 60)` head on the first token | Best published zero-shot hi/kn among base models; Apache-2.0 |
| Backup model | microsoft/Multilingual-MiniLM-L12-H384 | Use if training estimate > ~4 h |
| Training | English train only; max_len 64, batch 32, AdamW lr 2e-5 (head 1e-3), wd 0.01, warmup 10% + linear decay, ~~5~~ **15 epochs max** (raised after run 1: English dev accuracy was still rising at epoch 5; decided on English dev only, see docs/journal/04), early stop patience 2 on English dev accuracy, grad clip 1.0, fp16 on GPU, seed 42, word embeddings frozen | Standard recipe; frozen embeddings fit 4 GB GPU and keep multilingual vectors |
| LLM | **gemini-3.5-flash-lite** via `google-genai`, `generate_content`, temperature 0, JSON schema with enum of 60 labels, thinking level MINIMAL | Stable, free-tier, cheap |
| LLM prompt | Pilot v1 (labels only) vs v2 (labels + 1 English train example each) on 60 dev_sub items; better pooled accuracy wins, tie -> v1 | Dev-only decision |
| LLM batching | 20 items per request for the bulk run; single calls for 50 dev_sub items to measure latency/tokens | Saves quota; single calls give realistic cost |
| Cost price | $0.30 / 1M input, $2.50 / 1M output (paid tier, 2026-11-22) | Official pricing page |
| Confidence | Max softmax probability after temperature scaling | Simple, well known |
| Temperature | One T fit on **full English dev** logits by minimising NLL | Guo et al. 2017 |
| Threshold rule | t in {0.00..1.00 step 0.01, 1.01}; pick the lowest LLM share with pooled dev_sub accuracy >= LLM-only dev_sub accuracy − 1.0 pp | Decided on dev only; always has an answer |
| Serving | Flask app; gunicorn 3 sync workers, 4 torch threads each, preload_app, timeout 120 s, bind 127.0.0.1:8000; NGINX non-root on 127.0.0.1:8080 | 12 cores / 7.4 GB RAM |
| Load test | Locust headless, 20 users, spawn 5/s, 2 min, `ROUTER_DISABLE_LLM=1`, through NGINX | No API quota spent |
| Video | Pillow slides + Pygments code images + Piper TTS (en_US-lessac-medium) + ffmpeg concat | All local, free |

## File layout

```
src/router/  data.py model.py train.py evaluate.py llm.py llm_baseline.py calibrate.py route.py costs.py analysis.py
app/         app.py templates/index.html
deploy/      gunicorn.conf.py nginx/nginx.conf locustfile.py
scripts/     render_tables.py check_numbers.py loadtest_summary.py make_diagram.py
tests/       test_data.py test_llm.py test_model.py test_route.py test_api.py test_results.py (doc numbers: scripts/check_numbers.py in make verify)
results/     *.json, *.csv, logits/, charts/, llm_cache.jsonl
video/       script.md build_video.py check_video.py
```

## Makefile targets

`setup data llm-pilot llm llm-single smoke train evaluate route tables test lint verify
serve nginx-start nginx-stop stack-up stack-down loadtest video video-check`

## Acceptance checks per phase

| Phase | Check |
|---|---|
| 0 | `logs/00_env.log` exists; `.gitignore` has `.env`; `git status` clean |
| 1 | RESEARCH.md has sources + dates per question; PLAN.md names every decision |
| 2 | `pytest tests/test_data.py -q` passes |
| 3 | `tests/test_llm.py` passes; `results/llm_baseline.json`, `results/llm_single_call.json` exist |
| 4 | `tests/test_model.py` passes; `results/small_model.json`, `results/charts/training_curve.png` exist |
| 5 | `tests/test_route.py` passes; `results/routing.json`, `results/routing_curve.csv`, charts exist |
| 6 | `pytest tests/test_api.py -q` passes; `logs/06_smoke.log` has en/hi/kn curl results |
| 7 | `curl :8080/health` via NGINX; load test 0 failures; `make stack-down` stops everything |
| 8 | `make verify` exits 0; review findings + fixes in docs/journal/08; fresh-clone test works |
| 9 | README, INTERVIEW_GUIDE, FINAL_REPORT exist; `make verify` passes |
| 10 | `make video-check` exits 0; frame review notes in docs/journal/10 |
| 11 | final `make verify` + `make video-check` output shown |
