# Phase 7: Production-style serving on Linux and load test

## What we did
- `deploy/gunicorn.conf.py`: bind `127.0.0.1:8000`, 3 sync workers, `preload_app = True`, timeout 120 s,
  `max_requests` 2000 (+ jitter) to recycle workers, and a `post_fork` hook that sets
  `torch.set_num_threads(12 // 3 = 4)` in every worker.
- `deploy/nginx/nginx.conf`: NGINX as a normal user on `127.0.0.1:8080`, proxying to gunicorn with keep-alive
  connections, `client_max_body_size 16k`, proxy timeouts just above gunicorn's, access log inside the project.
  `pid` and every `*_temp_path` live under `deploy/nginx/`, and there is no `user` directive, so no sudo is needed.
- Makefile: `serve`, `serve-bg` (waits until `/health` answers), `nginx-start` (runs `nginx -t` first),
  `nginx-stop`, `stack-up`, `stack-down`, `loadtest`.
- `deploy/locustfile.py`: each simulated user posts a random `test_sub` sentence (en, hi or kn) to `/predict`,
  waiting 0.1-0.5 s between requests. Headless run through NGINX with `ROUTER_DISABLE_LLM=1`.
- `scripts/loadtest_summary.py` turns Locust's CSV into `results/loadtest.json`.

## Why
- PyTorch inference is CPU-bound and holds the GIL, so threads inside one process would not help; separate
  processes (sync workers) do. 3 workers x 4 threads = 12 threads = 12 cores.
- `preload_app` loads the model once in the master; the workers share those memory pages after `fork`.
- NGINX in front: buffers slow clients, rejects oversized bodies, and is the standard front door on Linux.

## Key commands
```bash
ROUTER_DISABLE_LLM=1 make stack-up      # gunicorn (daemon) + nginx; checks /health through :8080
curl -s http://127.0.0.1:8080/health    # {"status":"ok"}
make loadtest                            # locust -u 20 -r 5 -t 2m --headless --csv results/loadtest
make stack-down                          # nginx -s stop + kill gunicorn master
```

## Result
- `make stack-up`: `/health` and a real `/predict` through NGINX on :8080 both answer.
- Load test (`results/loadtest.json`, `logs/07_loadtest.log`): 20 users, spawn 5/s, 2 minutes, small model only,
  through NGINX → **4,908 requests, 0 failures, 41.2 requests/s, p50 180 ms, p95 260 ms, p99 300 ms** (max 370 ms).
- `make stack-down` stops NGINX and gunicorn (both pid files removed).
- The bottleneck is CPU-bound model inference. One message alone takes 37.5 ms (median, 4 threads). Under
  load, 3 workers serving 41.2 requests/s means each worker spends about 3 / 41.2 = 73 ms per request: the
  12 hardware threads are shared by 12 PyTorch threads *and* the Locust load generator on the same laptop, so
  each inference runs slower, and requests wait in gunicorn's queue (p50 180 ms). NGINX and Flask add very little.
  More throughput needs more CPU (or a faster model, e.g. INT8 quantization), not more workers.

## Errors and fixes
- Every request hung; Locust reported 0 requests. NGINX logged the calls as `499` (client gave up) and
  gunicorn's access log had only `/health`. A direct `curl` to gunicorn timed out after 20 s. Root cause:
  `preload_app` loads the model in the master process, which started PyTorch's OpenMP thread pool; the forked
  workers inherit a broken copy of that pool and deadlock on their first parallel operation. Fix
  (`deploy/gunicorn.conf.py`): `torch.set_num_threads(1)` in the master before the model loads; each worker
  still sets 4 threads in `post_fork`. Predictions then took 34-93 ms.
- Two guards so this can never pass silently again: `make stack-up` now sends a real `/predict` (not only
  `/health`), and the Locust request has a 30 s timeout, so a hang shows up as failures instead of "0 requests".
- `pkill -f "gunicorn ..."` killed my own shell once (its command line contained the pattern). The Makefile
  stops gunicorn by the pid in its pid file instead.

## How to explain this in an interview
1. Flask's own server is for development, so I served the app with gunicorn worker processes behind NGINX.
2. Inference is CPU-bound, so I used 3 processes with 4 PyTorch threads each, one thread per core, and preloaded the model before forking.
3. NGINX runs as a normal user with a project-local config, so the whole stack starts without sudo.
4. The load test went through NGINX with the LLM switched off, so it measured my server, not Google's API.
5. The bottleneck is CPU inference: 41 requests per second with 0 failures on one laptop; to go faster I would quantize the model or add CPU, not add workers.
6. My first load test hung because forked workers inherited PyTorch's thread pool from the master; I fixed it at the root and added checks so a hang can never look like success.
