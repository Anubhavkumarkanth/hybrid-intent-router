"""Locust load test: POST random test_sub sentences (en, hi, kn) to /predict.

Run (through NGINX, small model only):
  ROUTER_DISABLE_LLM=1 make stack-up
  make loadtest
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def _load_texts() -> list[str]:
    ids = set(json.loads((PROC / "test_sub_ids.json").read_text()))
    texts = []
    for lang in ("en", "hi", "kn"):
        with open(PROC / f"{lang}_test.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in ids:
                    texts.append(r["text"])
    return texts


TEXTS = _load_texts()


class RouterUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        # timeout: a hung server must show up as failures, not as "0 requests" (it happened once)
        with self.client.post("/predict", json={"text": random.choice(TEXTS)}, catch_response=True,
                              timeout=30) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif "intent" not in r.json():
                r.failure("no intent in response")
