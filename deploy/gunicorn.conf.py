"""gunicorn settings for the router (CPU-bound PyTorch inference).

12 cores / 7.4 GB RAM: 3 sync workers x 4 torch threads = 12 threads, one per core.
Each worker needs the model in memory (~1 GB fp32). preload_app loads the app (and model
weights) once in the master; workers share those pages copy-on-write. The master never runs
inference, and each worker sets its own torch thread count right after fork.
"""

import os

import torch

# preload_app loads the model in this master process before forking. If the master starts PyTorch's
# OpenMP thread pool, forked workers inherit a broken pool and hang on their first request (we hit
# this: every /predict timed out). So the master stays single-threaded; workers set threads in post_fork.
torch.set_num_threads(1)

bind = os.environ.get("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.environ.get("GUNICORN_WORKERS", "3"))
torch_threads = int(os.environ.get("TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // workers))))
worker_class = "sync"
preload_app = True
timeout = 120  # a worker silent for longer than this is killed and restarted
graceful_timeout = 30
keepalive = 2
max_requests = 2000  # recycle workers now and then (guards against slow memory growth)
max_requests_jitter = 200
accesslog = os.environ.get("GUNICORN_ACCESSLOG", "-")
errorlog = "-"
loglevel = "info"
wsgi_app = "app.app:make_app()"


def post_fork(server, worker):
    import torch

    torch.set_num_threads(torch_threads)
    server.log.info(f"worker {worker.pid}: torch threads = {torch.get_num_threads()}")
