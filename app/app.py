"""Flask API + demo page for the hybrid intent router.

GET  /health   -> {"status": "ok"}
POST /predict  {"text": "..."} -> {"intent", "confidence", "routed_to", "latency_ms"}
     routed_to is "small" (confident), "llm" (not confident, LLM answered) or
     "small_fallback" (not confident, but the LLM failed / timed out / is disabled).
GET  /         demo page

Env:
  ROUTER_DISABLE_LLM=1       small model only (used for load tests, saves API quota)
  ROUTER_MODEL_DIR           default models/small
  ROUTER_LLM_MAX_PER_MIN=N   at most N LLM calls per minute per worker process (0 = no limit); protects the
                             API quota on a public demo. Over the limit, unsure messages get small_fallback.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Protocol

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_CHARS = 500
log = logging.getLogger("router.app")


class SmallModel(Protocol):
    def predict(self, text: str) -> tuple[str, float]: ...


class TorchSmallModel:
    """Loads the fine-tuned encoder once; applies the temperature from calibration."""

    def __init__(self, model_dir: Path, temperature: float, labels: list[str]):
        import torch

        from router.model import load_model

        self.torch = torch
        self.model, self.tok, meta = load_model(model_dir, device="cpu")
        self.max_len = meta.get("max_len", 64)
        self.T = temperature
        self.labels = labels

    def predict(self, text: str) -> tuple[str, float]:
        torch = self.torch
        with torch.inference_mode():
            enc = self.tok([text], truncation=True, max_length=self.max_len, return_tensors="pt")
            logits = self.model(enc["input_ids"], enc["attention_mask"])[0]
            probs = torch.softmax(logits / self.T, -1)
            conf, idx = probs.max(-1)
        return self.labels[int(idx)], float(conf)


def read_settings(root: Path = ROOT) -> dict:
    routing = json.loads((root / "results" / "routing.json").read_text())
    return {
        "threshold": routing["chosen_threshold"]["t"],
        "temperature": routing["temperature"],
        "prompt_version": routing["llm"]["prompt_version"],
    }


def build_llm_fn(labels: list[str], prompt_version: str) -> Callable[[str], str] | None:
    from router import llm as L
    from router.llm_baseline import prompt_examples

    L.load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        log.warning("GEMINI_API_KEY not set: LLM disabled")
        return None
    backend = L.GeminiBackend(timeout_s=10.0)
    ex = prompt_examples(labels) if prompt_version == "v2_labels_examples" else None
    instructions = L.build_instructions(labels, prompt_version, ex)
    return lambda text: L.predict_one(backend, instructions, labels, text)


class RateLimit:
    """Allow at most `per_min` events in any 60-second window (0 = unlimited). Thread-safe."""

    def __init__(self, per_min: int, clock: Callable[[], float] = time.monotonic):
        self.per_min, self.clock = per_min, clock
        self.times: deque[float] = deque()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        if self.per_min <= 0:
            return True
        now = self.clock()
        with self.lock:
            while self.times and now - self.times[0] >= 60:
                self.times.popleft()
            if len(self.times) >= self.per_min:
                return False
            self.times.append(now)
            return True


def create_app(small: SmallModel | None = None, llm_fn: Callable[[str], str] | None = None,
               threshold: float | None = None, use_llm: bool | None = None,
               llm_limit: RateLimit | None = None) -> Flask:
    app = Flask(__name__)
    if use_llm is None:
        use_llm = os.environ.get("ROUTER_DISABLE_LLM", "0") != "1"
    if llm_limit is None:
        llm_limit = RateLimit(int(os.environ.get("ROUTER_LLM_MAX_PER_MIN", "0")))

    if small is None or threshold is None:
        from router.data import load_labels

        labels = load_labels()
        settings = read_settings()
        threshold = settings["threshold"] if threshold is None else threshold
        if small is None:
            model_dir = Path(os.environ.get("ROUTER_MODEL_DIR", ROOT / "models" / "small"))
            small = TorchSmallModel(model_dir, settings["temperature"], labels)
        if use_llm and llm_fn is None:
            llm_fn = build_llm_fn(labels, settings["prompt_version"])
    if not use_llm:
        llm_fn = None

    app.config.update(SMALL=small, LLM_FN=llm_fn, THRESHOLD=threshold)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/")
    def index():
        return render_template("index.html", threshold=app.config["THRESHOLD"],
                               llm_enabled=app.config["LLM_FN"] is not None)

    @app.post("/predict")
    def predict():
        t0 = time.perf_counter()
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            return jsonify(error='send JSON like {"text": "..."}'), 400
        text = body["text"].strip()
        if not text:
            return jsonify(error="text is empty"), 400
        if len(text) > MAX_CHARS:
            return jsonify(error=f"text is longer than {MAX_CHARS} characters"), 400

        intent, conf = app.config["SMALL"].predict(text)
        routed_to = "small"
        if conf < app.config["THRESHOLD"]:
            llm = app.config["LLM_FN"]
            if llm is None:
                routed_to = "small_fallback"
            elif not llm_limit.allow():
                log.warning("LLM call limit reached, using small model answer")
                routed_to = "small_fallback"
            else:
                try:
                    intent = llm(text)
                    routed_to = "llm"
                except Exception as e:  # any LLM failure -> keep the small model's answer
                    log.warning("LLM failed, using small model answer: %s: %s", type(e).__name__, e)
                    routed_to = "small_fallback"
        return jsonify(intent=intent, confidence=round(conf, 4), routed_to=routed_to,
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    return app


def make_app() -> Flask:
    """Entry point for gunicorn: `gunicorn 'app.app:make_app()'`."""
    logging.basicConfig(level=logging.INFO)
    return create_app()


if __name__ == "__main__":
    make_app().run(host="127.0.0.1", port=5000)
