"""Flask API + demo page for the hybrid intent router.

GET  /health   -> {"status": "ok"}
POST /predict  {"text": "..."} -> {"intent", "confidence", "routed_to", "latency_ms"}
     routed_to is "small" (confident), "llm" (not confident, LLM answered) or
     "small_fallback" (not confident, but the LLM failed / timed out / is disabled).
GET  /         demo page

Env:
  ROUTER_DISABLE_LLM=1       small model only (used for load tests, saves API quota)
  ROUTER_MODEL_DIR           default models/small (PyTorch checkpoint)
  ROUTER_ONNX_DIR            use the small ONNX model in this folder instead (e.g. models/onnx), no PyTorch
  ROUTER_ONNX_THREADS        onnxruntime threads (default 1)
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

from flask import Flask, jsonify, render_template, request, send_from_directory

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

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        torch = self.torch
        with torch.inference_mode():
            enc = self.tok([text], truncation=True, max_length=self.max_len, return_tensors="pt")
            logits = self.model(enc["input_ids"], enc["attention_mask"])[0]
            probs = torch.softmax(logits / self.T, -1)
            conf, idx = probs.topk(k)
        return [(self.labels[int(i)], float(c)) for c, i in zip(conf, idx, strict=True)]

    def predict(self, text: str) -> tuple[str, float]:
        return self.predict_top(text, 1)[0]


class OnnxSmallModel:
    """The shrunken ONNX model (scripts/quantize.py) on onnxruntime: no PyTorch, fits a 512 MB server."""

    def __init__(self, model_dir: Path, temperature: float, labels: list[str], threads: int = 1):
        from router.onnx_model import OnnxIntentModel

        self.model = OnnxIntentModel(model_dir, threads=threads)
        self.T = temperature
        self.labels = labels

    def predict_top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        import numpy as np

        z = self.model.logits([text])[0] / self.T
        p = np.exp(z - z.max())
        p /= p.sum()
        return [(self.labels[int(i)], float(p[i])) for i in np.argsort(-p)[:k]]

    def predict(self, text: str) -> tuple[str, float]:
        return self.predict_top(text, 1)[0]


CHARTS = {"architecture.png", "serving.png", "accuracy_vs_llm_share.png", "cost_per_1000.png", "reliability.png",
          "training_curve.png", "examples_table.png"}


def _load(root: Path, rel: str):
    p = root / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def page_data(root: Path = ROOT, deployed_onnx: bool = False) -> dict:
    """Everything the demo page shows besides live predictions, read from results/ (never typed in by hand).
    A missing file just leaves its part of the page out."""
    routing = _load(root, "results/routing.json")
    small = _load(root, "results/small_model.json")
    single = _load(root, "results/llm_single_call.json")
    quant = _load(root, "results/quantized.json")
    load = _load(root, "results/loadtest.json")
    labels = _load(root, "data/processed/labels.json") or []
    examples = _load(root, "results/prompt_examples.json") or {}
    domains: dict[str, list[dict]] = {}
    for lab in labels:
        dom, _, action = lab.partition("_")
        domains.setdefault(dom, []).append({"intent": lab, "action": action or dom, "example": examples.get(lab, "")})
    data: dict = {"intents": domains, "samples": _load(root, "results/demo_samples.json") or {},
                  "deployed_onnx": deployed_onnx}
    if routing:
        p = routing["test_sub"]["pooled"]
        data["headline"] = {
            "router_acc": p["router_acc"], "llm_acc": p["llm_only_acc"], "small_acc": p["small_only_acc"],
            "llm_share": p["llm_share"], "calls_avoided": routing["savings_test_sub_pooled"]["llm_calls_avoided_share"],
            "cost_llm": p["llm_only_cost_per_1000_usd"], "cost_router": p["router_cost_per_1000_usd"],
            "lat_llm": p["llm_only_avg_latency_ms"], "lat_router": p["router_avg_latency_ms"],
            "n_per_lang": routing["test_sub"]["en"]["n"], "threshold": routing["chosen_threshold"]["t"],
            "temperature": routing["temperature"],
        }
        data["per_lang"] = {lg: {k: routing["test_sub"][lg][k] for k in ("small_only_acc", "llm_only_acc",
                                                                         "router_acc", "llm_share")}
                            for lg in ("en", "hi", "kn", "pooled")}
    if small:
        data["small_model"] = {"en": small["accuracy"]["en_test_full"], "hi": small["accuracy"]["hi_test_full"],
                               "kn": small["accuracy"]["kn_test_full"],
                               "cpu_ms": small["cpu_latency_single_message"]["median_ms"]}
    if single:
        data["llm_ms"] = single["latency_ms_median"]
    if quant:
        agree = quant["agreement_with_original"]
        data["deployed"] = {"size_mb": quant["size_mb"]["onnx_small"],
                            "orig_mb": quant["size_mb"]["pytorch_checkpoint"],
                            "agree_min": min(agree.values()), "agree_max": max(agree.values()),
                            "cpu_ms": quant["cpu_latency_single_message_1_thread"]["median_ms"],
                            "memory_mb": quant["peak_memory_mb_load_and_predict"]}
    if load:
        data["load"] = {"rps": load["requests_per_sec"], "p95": load["p95_ms"], "failures": load["failures"],
                        "requests": load["requests"]}
    hin = _load(root, "results/hinglish.json")
    if hin:
        data["hinglish"] = {k: hin[k] for k in ("n", "small_only_acc", "llm_only_acc", "router_acc", "llm_share")}
    curve_file = root / "results" / "routing_curve.csv"
    if curve_file.exists():
        import csv

        curve: dict[str, list] = {}
        with open(curve_file, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test_sub":
                    curve.setdefault(row["group"], []).append(
                        [round(float(row["llm_share"]) * 100, 1), round(float(row["accuracy"]) * 100, 1),
                         float(row["t"])])
        data["curve"] = curve
    return data


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
               llm_limit: RateLimit | None = None, metrics=None) -> Flask:
    from router.actions import describe, run_action

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
            onnx_dir = os.environ.get("ROUTER_ONNX_DIR")
            if onnx_dir:
                small = OnnxSmallModel(ROOT / onnx_dir, settings["temperature"], labels,
                                       threads=int(os.environ.get("ROUTER_ONNX_THREADS", "1")))
            else:
                model_dir = Path(os.environ.get("ROUTER_MODEL_DIR", ROOT / "models" / "small"))
                small = TorchSmallModel(model_dir, settings["temperature"], labels)
        if use_llm and llm_fn is None:
            llm_fn = build_llm_fn(labels, settings["prompt_version"])
    if not use_llm:
        llm_fn = None

    app.config.update(SMALL=small, LLM_FN=llm_fn, THRESHOLD=threshold)
    data = page_data(deployed_onnx=isinstance(small, OnnxSmallModel))  # read once; results don't change at runtime

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/")
    def index():
        return render_template("index.html", threshold=app.config["THRESHOLD"],
                               llm_enabled=app.config["LLM_FN"] is not None, data=data)

    @app.get("/charts/<name>")
    def chart(name: str):
        if name not in CHARTS:  # only the project's own charts, never arbitrary paths
            return jsonify(error="not found"), 404
        return send_from_directory(ROOT / "results" / "charts", name, max_age=86400)

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

        small_model = app.config["SMALL"]
        top = small_model.predict_top(text, 3) if hasattr(small_model, "predict_top") else [small_model.predict(text)]
        intent, conf = top[0]
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
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Do the task for intents that have an action (weather, time, currency, ...); otherwise confirm it.
        action = run_action(intent, text)
        reply = action["reply"] if action else f"Got it. You want me to {describe(intent)}."
        language = body.get("lang") if body.get("lang") in LANGS else script_language(text)
        voice = body.get("voice") is True
        if metrics is not None:
            metrics.record(language=language, routed_to=routed_to, intent=intent, confidence=round(conf, 4),
                           latency_ms=latency_ms, action=action["kind"] if action else None, voice=voice)
        return jsonify(intent=intent, confidence=round(conf, 4), routed_to=routed_to, latency_ms=latency_ms,
                       threshold=app.config["THRESHOLD"], language=language,
                       small_model_top=[{"intent": i, "confidence": round(c, 4)} for i, c in top],
                       action=action, reply=reply, total_ms=round((time.perf_counter() - t0) * 1000, 1))

    summary_cache: dict = {}

    @app.get("/api/metrics")
    def api_metrics():
        if metrics is None:
            return jsonify(error="metrics are off"), 404
        hours =min(max(request.args.get("hours", 24, type=float), 1), 168)
        hit = summary_cache.get(hours)
        if hit and time.monotonic() - hit[0] < 5:  # many open dashboards poll this; one query per 5 s is enough
            return jsonify(hit[1])
        s = metrics.summary(hours=hours, cost_per_call=cost_per_call())
        summary_cache[hours] = (time.monotonic(), s)
        return jsonify(s)

    return app


LANGS = {"en", "hi", "kn", "hinglish"}


def script_language(text: str) -> str:
    """Language guess from the writing system (Latin text is counted as English)."""
    dev = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    kan = sum(0x0C80 <= ord(c) <= 0x0CFF for c in text)
    lat = sum(c.isascii() and c.isalpha() for c in text)
    best = max((dev, "hi"), (kan, "kn"), (lat, "en"))
    return best[1] if best[0] else "en"


def cost_per_call(root: Path = ROOT) -> float:
    from router import costs

    single = _load(root, "results/llm_single_call.json")
    return costs.cost_per_call(single["mean_input_tokens"], single["mean_output_tokens"]) if single else 0.0


def make_app() -> Flask:
    """Entry point for gunicorn: `gunicorn 'app.app:make_app()'`."""
    logging.basicConfig(level=logging.INFO)
    store = None
    url = os.environ.get("DATABASE_URL")
    if url:
        from router.metrics import MetricsStore

        try:
            store = MetricsStore.from_url(url)
            log.info("metrics: PostgreSQL")
        except Exception as e:  # a database problem must not stop the demo from serving
            log.warning("metrics off, database unavailable: %s: %s", type(e).__name__, e)
    else:
        log.info("metrics off: no DATABASE_URL")
    return create_app(metrics=store)


if __name__ == "__main__":
    make_app().run(host="127.0.0.1", port=5000)
