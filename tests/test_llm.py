"""LLM client tests with a fake backend (no network, no API key)."""

import json

import pytest

from router import llm as L

LABELS = ["alarm_set", "play_music", "weather_query"]


class FakeBackend:
    """Returns scripted responses in order; records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, schema):
        self.prompts.append(prompt)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return L.LLMResponse(text=r, input_tokens=100, output_tokens=5, latency_ms=12.0)


def batch_json(intents):
    return json.dumps({"results": [{"n": i + 1, "intent": x} for i, x in enumerate(intents)]})


def make_clf(tmp_path, responses, version="v1_labels"):
    return L.Classifier(
        backend=FakeBackend(responses),
        labels=LABELS,
        prompt_version=version,
        model="fake-model",
        cache=L.Cache(path=tmp_path / "cache.jsonl"),
        sleep=lambda s: None,
    )


# ---- parsing

def test_parse_single_ok_and_fenced():
    assert L.parse_single('{"intent": "play_music"}', set(LABELS)) == "play_music"
    assert L.parse_single('```json\n{"intent": "alarm_set"}\n```', set(LABELS)) == "alarm_set"


@pytest.mark.parametrize("text", ["not json", '{"x": 1}', '{"intent": "fly_plane"}'])
def test_parse_single_rejects_bad(text):
    with pytest.raises(L.BadAnswer):
        L.parse_single(text, set(LABELS))


def test_parse_batch_orders_by_number():
    text = json.dumps({"results": [{"n": 2, "intent": "play_music"}, {"n": 1, "intent": "alarm_set"}]})
    assert L.parse_batch(text, 2, set(LABELS)) == ["alarm_set", "play_music"]


def test_parse_batch_unknown_label_is_invalid():
    assert L.parse_batch(batch_json(["alarm_set", "nope"]), 2, set(LABELS)) == ["alarm_set", L.INVALID]


@pytest.mark.parametrize(
    "text,n",
    [
        (batch_json(["alarm_set"]), 2),  # wrong count
        (json.dumps({"results": [{"n": 1, "intent": "alarm_set"}, {"n": 1, "intent": "alarm_set"}]}), 2),
        (json.dumps({"results": [{"n": 5, "intent": "alarm_set"}]}), 1),  # out of range
        ("[]", 0),
    ],
)
def test_parse_batch_rejects_broken_structure(text, n):
    with pytest.raises(L.BadAnswer):
        L.parse_batch(text, n, set(LABELS))


# ---- prompts

def test_prompt_versions():
    v1 = L.build_instructions(LABELS, "v1_labels")
    assert all(lab in v1 for lab in LABELS)
    ex = {lab: f"example for {lab}" for lab in LABELS}
    v2 = L.build_instructions(LABELS, "v2_labels_examples", ex)
    assert 'alarm_set: "example for alarm_set"' in v2
    with pytest.raises(ValueError):
        L.build_instructions(LABELS, "v2_labels_examples")
    p = L.build_batch_prompt(v1, ["a", "b"])
    assert "1. a\n2. b" in p


# ---- batching + cache

def test_batching_splits_and_caches(tmp_path):
    items = [{"id": str(i), "text": f"t{i}"} for i in range(5)]
    resp = [batch_json(["alarm_set", "play_music"]), batch_json(["weather_query", "alarm_set"]),
            batch_json(["play_music"])]
    clf = make_clf(tmp_path, resp)
    out = clf.classify_batch("en", items, batch_size=2)
    assert out == ["alarm_set", "play_music", "weather_query", "alarm_set", "play_music"]
    assert len(clf.backend.prompts) == 3
    # Second run: everything from cache, no calls (backend has no responses left).
    clf2 = make_clf(tmp_path, [])
    assert clf2.classify_batch("en", items, batch_size=2) == out
    # Cache file persisted one line per item.
    lines = (tmp_path / "cache.jsonl").read_text().splitlines()
    assert len(lines) == 5


def test_resume_only_calls_missing(tmp_path):
    items = [{"id": str(i), "text": f"t{i}"} for i in range(3)]
    make_clf(tmp_path, [batch_json(["alarm_set", "alarm_set"])]).classify_batch("en", items[:2], batch_size=2)
    clf = make_clf(tmp_path, [batch_json(["play_music"])])
    assert clf.classify_batch("en", items, batch_size=2) == ["alarm_set", "alarm_set", "play_music"]
    assert len(clf.backend.prompts) == 1


def test_bad_batch_retried_then_single_fallback(tmp_path):
    items = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    resp = ["garbage", batch_json(["alarm_set"]),  # two broken batch answers
            '{"intent": "play_music"}', '{"intent": "bogus"}', '{"intent": "bogus"}']
    clf = make_clf(tmp_path, resp)
    assert clf.classify_batch("en", items) == ["play_music", L.INVALID]


def test_cache_key_separates_language_and_version(tmp_path):
    items = [{"id": "1", "text": "a"}]
    make_clf(tmp_path, [batch_json(["alarm_set"])]).classify_batch("en", items)
    clf = make_clf(tmp_path, [batch_json(["play_music"])])
    assert clf.classify_batch("hi", items) == ["play_music"]
    clf3 = make_clf(tmp_path, [batch_json(["weather_query"])], version="v1_labels")
    assert clf3.classify_batch("en", items) == ["alarm_set"]  # cached


def test_single_records_tokens_and_latency(tmp_path):
    clf = make_clf(tmp_path, ['{"intent": "weather_query"}'])
    rec = clf.classify_single("kn", {"id": "9", "text": "x"})
    assert rec["intent"] == "weather_query"
    assert rec["input_tokens"] == 100 and rec["output_tokens"] == 5 and rec["latency_ms"] == 12.0


# ---- backoff

def test_backoff_retries_rate_limit_then_succeeds():
    be = FakeBackend([L.RateLimited("429"), L.TransientError("503"), '{"intent": "alarm_set"}'])
    delays = []
    r = L.call_with_backoff(be, "p", {}, sleep=delays.append)
    assert r.text == '{"intent": "alarm_set"}'
    assert len(delays) == 2 and delays[1] > delays[0]


def test_backoff_gives_up_and_quota_is_not_retried():
    be = FakeBackend([L.RateLimited("429")] * 3)
    with pytest.raises(L.RateLimited):
        L.call_with_backoff(be, "p", {}, max_tries=3, sleep=lambda s: None)
    be2 = FakeBackend([L.QuotaExhausted("daily")])
    with pytest.raises(L.QuotaExhausted):
        L.call_with_backoff(be2, "p", {}, sleep=lambda s: None)
