# Hugging Face Space (Docker SDK). Built from the folder that scripts/build_space.sh assembles.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

# 2 vCPUs on the free tier: 2 workers x 1 PyTorch thread. At most 10 LLM calls per minute per worker.
ENV PYTHONPATH=/home/user/app/src:/home/user/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    GUNICORN_WORKERS=2 \
    TORCH_THREADS=1 \
    ROUTER_LLM_MAX_PER_MIN=10

EXPOSE 7860
CMD ["bash", "start.sh"]
