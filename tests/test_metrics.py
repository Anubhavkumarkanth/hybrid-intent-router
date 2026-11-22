"""Metrics: the pure summary maths, and the store against a fake Postgres connection (no database needed)."""

import time

import pytest

from router import metrics as M


class FakePostgres:
    """Just enough of a DB-API connection to run MetricsStore's SQL: keeps rows in a list."""

    def __init__(self):
        self.rows, self.sql = [], []

    def __call__(self):  # used as the `connect` function
        return self

    def cursor(self):
        return self

    def close(self):
        pass

    def execute(self, sql, params=()):
        self.sql.append(sql)
        if sql == M.SELECT_TOTALS:
            self.result = [(len(self.rows), min((r[0] for r in self.rows), default=None))]
        elif sql == M.SELECT_WINDOW:
            self.result = sorted((r for r in self.rows if r[0] >= params[0]), reverse=True)
        else:
            self.result = []
        if sql.startswith("INSERT"):
            n, width = len(params) // len(M.COLUMNS), len(M.COLUMNS)
            assert sql == M.insert_sql(n) and len(params) == n * width
            self.rows.extend(tuple(params[i:i + width]) for i in range(0, len(params), width))

    def fetchall(self):
        return self.result


def test_summary_counts_routes_languages_cost_and_timeline():
    now = time.time()
    rec = [dict(zip(M.COLUMNS, r, strict=True)) for r in [
        (now, "en", "small", "alarm_set", 0.9, 20.0, None, False),
        (now - 60, "hi", "small", "weather_query", 0.8, 30.0, "weather", True),
        (now - 120, "kn", "llm", "play_music", 0.3, 900.0, None, False),
        (now - 180, "hinglish", "small_fallback", "qa_maths", 0.4, 40.0, "maths", False)]]
    s = M.summarize(rec, hours=24, cost_per_call=0.001, now=now)
    assert s["total"] == 4 and s["routes"] == {"small": 2, "llm": 1, "small_fallback": 1}
    assert s["without_llm_share"] == 0.75
    assert s["languages"] == {"en": 1, "hi": 1, "kn": 1, "hinglish": 1}
    assert s["actions"] == {"weather": 1, "maths": 1} and s["voice"] == 1
    assert s["cost_usd"] == {"router": 0.001, "all_llm": 0.004, "saved": 0.003, "per_llm_call": 0.001}
    assert s["latency_ms"]["p50"] == 35.0
    # regression: a request at exactly `now` used to fall outside the last bucket
    assert len(s["timeline"]) == 24 and sum(b["small"] + b["llm"] + b["small_fallback"] for b in s["timeline"]) == 4


def test_summary_of_no_traffic():
    s = M.summarize([], hours=6, cost_per_call=0.001, now=time.time())
    assert s["total"] == 0 and s["without_llm_share"] is None and s["latency_ms"]["p95"] is None
    assert len(s["timeline"]) == 36  # 10-minute buckets for short windows


def test_store_creates_table_writes_in_background_and_reads_window():
    db = FakePostgres()
    store = M.MetricsStore(db, background=True)
    assert db.sql[:2] == [M.SCHEMA, M.INDEX]
    now = time.time()
    for _ in range(5):
        store.record(language="en", routed_to="small", intent="alarm_set", confidence=0.9, latency_ms=12.0,
                     action=None, voice=False)
    store.record(ts=now - 3 * 86400, language="en", routed_to="small", intent="old", confidence=0.9,
                 latency_ms=10.0, action=None, voice=False)
    store.flush()
    s = store.summary(hours=24)
    assert s["total"] == 5 and s["total_all_time"] == 6 and s["backend"] == "postgres"
    assert "old" not in dict(s["top_intents"])


def test_only_postgres_urls_are_accepted():
    with pytest.raises(ValueError):
        M.MetricsStore.from_url("mysql://user@host/db")
