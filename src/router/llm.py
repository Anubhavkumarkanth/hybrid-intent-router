"""LLM intent classifier (Gemini) with JSON output, validation, retries and a disk cache.

Main ideas:
- The LLM must answer with one of the 60 MASSIVE intent labels. We ask for JSON and
  check that every answer is a real label. A bad answer is retried once, then saved
  as "invalid" (counts as wrong).
- Batch mode: 20 numbered messages per request (saves free-tier quota).
  Single mode: one message per request (realistic production latency and tokens).
- Every answer is cached in results/llm_cache.jsonl, keyed by
  (model, prompt_version, mode, lang, id). A key is never sent twice, so runs resume.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "results" / "llm_cache.jsonl"

MODEL_ID = "gemini-3.5-flash-lite"
PROMPT_VERSIONS = ("v1_labels", "v2_labels_examples")
BATCH_SIZE = 20
INVALID = "invalid"


# ---------------------------------------------------------------- prompts

def build_instructions(labels: list[str], version: str, examples: dict[str, str] | None = None) -> str:
    head = (
        "You classify short messages sent to a voice assistant into exactly one intent.\n"
        "Messages may be in English, Hindi or Kannada. Choose the single best intent label "
        "from the list. Answer only with labels from the list, spelled exactly.\n\n"
    )
    if version == "v1_labels":
        body = "Intent labels:\n" + "\n".join(labels)
    elif version == "v2_labels_examples":
        if not examples:
            raise ValueError("v2 needs one example per label")
        body = "Intent labels, each with one English example:\n" + "\n".join(
            f"{lab}: \"{examples[lab]}\"" for lab in labels
        )
    else:
        raise ValueError(f"unknown prompt version {version}")
    return head + body


def build_batch_prompt(instructions: str, texts: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        f"{instructions}\n\n"
        f"Classify each of the {len(texts)} numbered messages below. Return JSON with a list "
        f"\"results\" holding one object per message: {{\"n\": <message number>, \"intent\": <label>}}.\n\n"
        f"Messages:\n{lines}"
    )


def build_single_prompt(instructions: str, text: str) -> str:
    return f"{instructions}\n\nReturn JSON {{\"intent\": <label>}} for this message:\n{text}"


def batch_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "intent": {"type": "string", "enum": list(labels)},
                    },
                    "required": ["n", "intent"],
                },
            }
        },
        "required": ["results"],
    }


def single_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(labels)}},
        "required": ["intent"],
    }


# ---------------------------------------------------------------- parsing

class BadAnswer(ValueError):
    """The model's answer could not be used (bad JSON, wrong count, unknown label)."""


def _load_json(text: str) -> object:
    text = (text or "").strip()
    # Some models wrap JSON in ``` fences even in JSON mode; strip them.
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise BadAnswer(f"not JSON: {e}") from e


def parse_single(text: str, labels: set[str]) -> str:
    obj = _load_json(text)
    if not isinstance(obj, dict) or "intent" not in obj:
        raise BadAnswer("missing 'intent'")
    intent = str(obj["intent"]).strip()
    if intent not in labels:
        raise BadAnswer(f"unknown label {intent!r}")
    return intent


def parse_batch(text: str, n_items: int, labels: set[str]) -> list[str]:
    """Return one label per item (in order). Unknown labels become INVALID.

    Raises BadAnswer if the structure is broken (bad JSON, wrong count or numbering),
    because then we cannot tell which answer belongs to which message.
    """
    obj = _load_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        raise BadAnswer("missing 'results' list")
    results = obj["results"]
    if len(results) != n_items:
        raise BadAnswer(f"expected {n_items} results, got {len(results)}")
    out: dict[int, str] = {}
    for r in results:
        if not isinstance(r, dict) or "n" not in r or "intent" not in r:
            raise BadAnswer(f"bad item {r!r}")
        try:
            n = int(r["n"])
        except (TypeError, ValueError) as e:
            raise BadAnswer(f"bad number {r['n']!r}") from e
        if not 1 <= n <= n_items or n in out:
            raise BadAnswer(f"bad or duplicate number {n}")
        intent = str(r["intent"]).strip()
        out[n] = intent if intent in labels else INVALID
    return [out[i] for i in range(1, n_items + 1)]


# ---------------------------------------------------------------- backend

@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0  # visible output + thinking tokens (both billed as output)
    thinking_tokens: int = 0
    latency_ms: float = 0.0


class Backend(Protocol):
    def generate(self, prompt: str, schema: dict) -> LLMResponse: ...


class RateLimited(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """Daily quota is used up. Stop and resume after the reset (midnight Pacific)."""


class TransientError(RuntimeError):
    pass


class GeminiBackend:
    """Thin wrapper around google-genai `generate_content` with JSON schema output."""

    def __init__(self, model: str = MODEL_ID, timeout_s: float = 60.0, api_key: str | None = None):
        from google import genai
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        self._types = types
        self.model = model
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )

    def generate(self, prompt: str, schema: dict) -> LLMResponse:
        from google.genai import errors

        types = self._types
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )
        t0 = time.perf_counter()
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.ClientError as e:
            if e.code == 429:
                msg = str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise QuotaExhausted(msg[:300]) from e
                raise RateLimited(msg[:300]) from e
            raise
        except errors.ServerError as e:
            raise TransientError(str(e)[:300]) from e
        except _network_errors() as e:
            # Read timeouts / dropped connections from the SDK's HTTP layer (httpx or httpx2).
            # They are not builtin TimeoutError, so map them to TransientError to get retried.
            raise TransientError(f"{type(e).__name__}: {e}"[:300]) from e
        latency = (time.perf_counter() - t0) * 1000
        u = resp.usage_metadata
        cand = (u.candidates_token_count or 0) if u else 0
        thoughts = (u.thoughts_token_count or 0) if u else 0
        return LLMResponse(
            text=resp.text or "",
            input_tokens=(u.prompt_token_count or 0) if u else 0,
            output_tokens=cand + thoughts,
            thinking_tokens=thoughts,
            latency_ms=latency,
        )


def _network_errors() -> tuple[type[BaseException], ...]:
    """HTTP transport errors (timeouts, connection errors) from httpx / httpx2, whichever is installed."""
    errs: list[type[BaseException]] = [TimeoutError, ConnectionError]
    for mod in ("httpx", "httpx2"):
        try:
            errs.append(__import__(mod).TransportError)
        except ImportError:
            continue
    return tuple(errs)


def call_with_backoff(
    backend: Backend,
    prompt: str,
    schema: dict,
    max_tries: int = 6,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMResponse:
    """Retry 429 / 5xx / timeouts with exponential backoff and jitter."""
    for attempt in range(max_tries):
        try:
            return backend.generate(prompt, schema)
        except QuotaExhausted:
            raise
        except (RateLimited, TransientError, TimeoutError) as e:
            if attempt == max_tries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 1)
            print(f"  retry {attempt + 1}/{max_tries - 1} after {delay:.1f}s: {type(e).__name__}")
            sleep(delay)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- cache

@dataclass
class Cache:
    path: Path = CACHE_PATH
    _data: dict[tuple, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path.exists():
            # A machine crash mid-append can leave NUL padding at the end of the file; drop it.
            # Any other bad line still raises, so real corruption is never hidden.
            text = self.path.read_text(encoding="utf-8").replace("\x00", "")
            for line in text.splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[self.key(rec)] = rec

    @staticmethod
    def key(rec: dict) -> tuple:
        return (rec["model"], rec["prompt_version"], rec["mode"], rec["lang"], rec["id"])

    def get(self, model: str, pv: str, mode: str, lang: str, id_: str) -> dict | None:
        return self._data.get((model, pv, mode, lang, id_))

    def add_many(self, recs: Iterable[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in recs:
                self._data[self.key(rec)] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # paid-for answers must survive a crash (WSL died once mid-run)

    def records(self) -> list[dict]:
        return list(self._data.values())


# ---------------------------------------------------------------- classify

@dataclass
class Classifier:
    backend: Backend
    labels: list[str]
    prompt_version: str
    model: str = MODEL_ID
    examples: dict[str, str] | None = None
    cache: Cache = field(default_factory=Cache)
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.label_set = set(self.labels)
        self.instructions = build_instructions(self.labels, self.prompt_version, self.examples)

    def classify_batch(self, lang: str, items: list[dict], batch_size: int = BATCH_SIZE) -> list[str]:
        """items: dicts with id and text. Returns intents in the same order (from cache when possible)."""
        todo = [r for r in items if not self.cache.get(self.model, self.prompt_version, "batch", lang, r["id"])]
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            self._run_batch(lang, chunk)
        return [self.cache.get(self.model, self.prompt_version, "batch", lang, r["id"])["intent"] for r in items]

    def _run_batch(self, lang: str, chunk: list[dict]) -> None:
        prompt = build_batch_prompt(self.instructions, [r["text"] for r in chunk])
        schema = batch_schema(self.labels)
        last_err = None
        for attempt in range(2):  # first try + one retry on a bad answer
            resp = call_with_backoff(self.backend, prompt, schema, sleep=self.sleep)
            try:
                intents = parse_batch(resp.text, len(chunk), self.label_set)
                break
            except BadAnswer as e:
                last_err = e
                print(f"  bad batch answer (try {attempt + 1}): {e}")
        else:
            # Structure still broken: fall back to one call per item for this chunk.
            print(f"  batch failed twice ({last_err}); classifying {len(chunk)} items one by one")
            for r in chunk:
                self._single_into_cache(lang, r, mode="batch")
            return
        self.cache.add_many(
            {
                "model": self.model,
                "prompt_version": self.prompt_version,
                "mode": "batch",
                "lang": lang,
                "id": r["id"],
                "intent": intent,
                "batch_size": len(chunk),
                "batch_input_tokens": resp.input_tokens,
                "batch_output_tokens": resp.output_tokens,
                "batch_latency_ms": round(resp.latency_ms, 1),
            }
            for r, intent in zip(chunk, intents, strict=True)
        )

    def classify_single(self, lang: str, item: dict) -> dict:
        rec = self.cache.get(self.model, self.prompt_version, "single", lang, item["id"])
        if rec is None:
            rec = self._single_into_cache(lang, item, mode="single")
        return rec

    def _single_into_cache(self, lang: str, item: dict, mode: str) -> dict:
        prompt = build_single_prompt(self.instructions, item["text"])
        schema = single_schema(self.labels)
        intent = INVALID
        tries = []
        for _ in range(2):
            resp = call_with_backoff(self.backend, prompt, schema, sleep=self.sleep)
            tries.append(resp)
            try:
                intent = parse_single(resp.text, self.label_set)
                break
            except BadAnswer as e:
                print(f"  bad single answer for {lang}/{item['id']}: {e}")
        first = tries[0]
        rec = {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "mode": mode,
            "lang": lang,
            "id": item["id"],
            "intent": intent,
            "input_tokens": first.input_tokens,
            "output_tokens": first.output_tokens,
            "thinking_tokens": first.thinking_tokens,
            "latency_ms": round(first.latency_ms, 1),
            "attempts": len(tries),
        }
        self.cache.add_many([rec])
        return rec


def predict_one(backend: Backend, instructions: str, labels: list[str], text: str) -> str:
    """Uncached single classification, used by the web app. Raises on failure."""
    resp = call_with_backoff(backend, build_single_prompt(instructions, text), single_schema(labels), max_tries=2)
    return parse_single(resp.text, set(labels))


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ (without printing anything)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
