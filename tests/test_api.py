"""API tests with a fake small model and a fake LLM (no weights, no network)."""

import json
from pathlib import Path

import pytest

from app.app import MAX_CHARS, create_app


class FakeSmall:
    def __init__(self, conf):
        self.conf = conf

    def predict(self, text):
        return "alarm_set", self.conf


def client(conf=0.9, llm_fn=None, use_llm=True, threshold=0.5):
    app = create_app(small=FakeSmall(conf), llm_fn=llm_fn, threshold=threshold, use_llm=use_llm)
    app.config["TESTING"] = True
    return app.test_client()


def test_llm_call_limit_falls_back_to_small_model():
    from app.app import RateLimit

    now = [0.0]
    limit = RateLimit(2, clock=lambda: now[0])
    app = create_app(small=FakeSmall(0.1), llm_fn=lambda t: "play_music", threshold=0.5, use_llm=True,
                     llm_limit=limit)
    c = app.test_client()
    routed = [c.post("/predict", json={"text": "x"}).get_json()["routed_to"] for _ in range(3)]
    assert routed == ["llm", "llm", "small_fallback"]  # third unsure message in the same minute
    now[0] = 60.0  # a minute later the window is free again
    assert c.post("/predict", json={"text": "x"}).get_json()["routed_to"] == "llm"


class FakeTop(FakeSmall):
    def predict_top(self, text, k=3):
        return [("alarm_set", self.conf), ("alarm_query", 0.2), ("play_music", 0.05)][:k]


def test_predict_returns_small_model_top3_and_threshold():
    app = create_app(small=FakeTop(0.9), llm_fn=None, threshold=0.5, use_llm=False)
    j = app.test_client().post("/predict", json={"text": "wake me up"}).get_json()
    assert j["threshold"] == 0.5 and j["intent"] == "alarm_set"
    assert [t["intent"] for t in j["small_model_top"]] == ["alarm_set", "alarm_query", "play_music"]


def test_llm_answer_keeps_small_model_guesses():
    app = create_app(small=FakeTop(0.3), llm_fn=lambda t: "play_music", threshold=0.5, use_llm=True)
    j = app.test_client().post("/predict", json={"text": "x"}).get_json()
    assert j["routed_to"] == "llm" and j["intent"] == "play_music"
    assert j["small_model_top"][0] == {"intent": "alarm_set", "confidence": 0.3}


def test_charts_only_serves_known_files():
    c = client()
    assert c.get("/charts/architecture.png").status_code == 200
    assert c.get("/charts/routing.json").status_code == 404
    assert c.get("/charts/..%2Frouting.json").status_code == 404


def test_index_shows_headline_numbers_from_results():
    html = client().get("/").get_data(as_text=True)
    routing = json.loads((Path(__file__).resolve().parents[1] / "results" / "routing.json").read_text())
    assert f"{routing['test_sub']['pooled']['router_acc'] * 100:.1f}%" in html
    assert "Live dashboard" in html and "page-data" in html and "/api/" not in html.split("<script")[0]


def test_predict_does_the_task_and_records_metrics_without_text(monkeypatch):
    from router import actions as A
    from router.metrics import MetricsStore
    from tests.test_metrics import FakePostgres

    monkeypatch.setattr(A, "run_action", lambda intent, text: {"kind": "maths", "title": "2 + 2", "reply": "2 + 2 = 4",
                                                                "detail": {"value": 4}, "source": "calculator"})
    db = FakePostgres()
    app = create_app(small=FakeTop(0.9), llm_fn=None, threshold=0.5, use_llm=False,
                     metrics=MetricsStore(db, background=False))
    c = app.test_client()
    j = c.post("/predict", json={"text": "what is two plus two my secret", "lang": "en", "voice": True}).get_json()
    assert j["reply"] == "2 + 2 = 4" and j["action"]["kind"] == "maths" and j["language"] == "en"
    s = c.get("/api/metrics").get_json()
    assert s["total"] == 1 and s["voice"] == 1 and s["actions"] == {"maths": 1}
    assert not any("secret" in str(v) for row in db.rows for v in row)  # the message text is never stored


def test_reply_for_intent_without_action_and_language_guess():
    from app.app import script_language

    j = client(conf=0.9).post("/predict", json={"text": "x"}).get_json()
    assert j["action"] is None and j["reply"] == "Got it. You want me to set an alarm."
    assert script_language("मुझे जगा दो") == "hi" and script_language("ಹಾಡು ಹಾಕು") == "kn"
    assert script_language("wake me up") == "en"


def test_metrics_endpoint_is_off_without_a_store():
    assert client().get("/api/metrics").status_code == 404


def test_rate_limit_zero_means_unlimited():
    from app.app import RateLimit

    assert all(RateLimit(0).allow() for _ in range(1000))


def test_health():
    r = client().get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_read_settings_takes_threshold_and_temperature_from_routing_json(tmp_path):
    from app.app import read_settings

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "routing.json").write_text(json.dumps(
        {"chosen_threshold": {"t": 0.42}, "temperature": 1.5, "llm": {"prompt_version": "v1_labels"}}))
    assert read_settings(tmp_path) == {"threshold": 0.42, "temperature": 1.5, "prompt_version": "v1_labels"}


def test_index_page():
    r = client().get("/")
    assert r.status_code == 200 and b"id=\"text\"" in r.data and b"id=\"mic\"" in r.data


def test_confident_uses_small_model():
    calls = []
    r = client(conf=0.9, llm_fn=lambda t: calls.append(t) or "play_music").post("/predict", json={"text": "hi"})
    j = r.get_json()
    assert r.status_code == 200
    assert j["intent"] == "alarm_set" and j["routed_to"] == "small" and j["confidence"] == 0.9
    assert isinstance(j["latency_ms"], float)
    assert calls == []


def test_unsure_goes_to_llm():
    r = client(conf=0.2, llm_fn=lambda t: "play_music").post("/predict", json={"text": "gaana bajao"})
    j = r.get_json()
    assert j["intent"] == "play_music" and j["routed_to"] == "llm"


def test_llm_failure_falls_back():
    def boom(text):
        raise TimeoutError("LLM timed out")

    r = client(conf=0.2, llm_fn=boom).post("/predict", json={"text": "x"})
    j = r.get_json()
    assert r.status_code == 200
    assert j["intent"] == "alarm_set" and j["routed_to"] == "small_fallback"


def test_llm_disabled_falls_back():
    r = client(conf=0.2, llm_fn=lambda t: "play_music", use_llm=False).post("/predict", json={"text": "x"})
    assert r.get_json()["routed_to"] == "small_fallback"


@pytest.mark.parametrize(
    "payload,msg",
    [
        ({"text": ""}, "empty"),
        ({"text": "   "}, "empty"),
        ({"text": "a" * (MAX_CHARS + 1)}, "longer"),
        ({"words": "hi"}, "JSON"),
        ({"text": 5}, "JSON"),
    ],
)
def test_bad_input_400(payload, msg):
    r = client().post("/predict", json=payload)
    assert r.status_code == 400 and msg in r.get_json()["error"]


def test_non_json_body_400():
    r = client().post("/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_exactly_max_chars_ok():
    r = client().post("/predict", json={"text": "a" * MAX_CHARS})
    assert r.status_code == 200
