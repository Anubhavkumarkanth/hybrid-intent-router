"""Live request metrics for the dashboard, stored in PostgreSQL (a free Neon database for the live demo).

Stores no message text and no IP address: only the time, language, route, intent, confidence, latency, which
task ran (if any) and whether the message came by voice. That is enough for the graphs and keeps visitors'
messages private.

Writes go through a background thread, so a slow database never slows down a prediction; if the queue is full
or the database is down, the event is dropped and logged, never raised. Without a DATABASE_URL the app runs
with metrics switched off.
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from collections import Counter
from typing import Callable

log = logging.getLogger("router.metrics")
COLUMNS = ("ts", "language", "routed_to", "intent", "confidence", "latency_ms", "action", "voice")
SCHEMA = ("CREATE TABLE IF NOT EXISTS requests (ts DOUBLE PRECISION NOT NULL, language TEXT, routed_to TEXT, "
          "intent TEXT, confidence DOUBLE PRECISION, latency_ms DOUBLE PRECISION, action TEXT, voice BOOLEAN)")
INDEX = "CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts)"
ROW = f"({', '.join(['%s'] * len(COLUMNS))})"
SELECT_WINDOW = f"SELECT {', '.join(COLUMNS)} FROM requests WHERE ts >= %s ORDER BY ts DESC LIMIT 20000"
SELECT_TOTALS = "SELECT COUNT(*), MIN(ts) FROM requests"


def insert_sql(n_rows: int) -> str:
    return f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES {', '.join([ROW] * n_rows)}"


def postgres_connect(url: str) -> Callable[[], object]:
    def connect():
        import psycopg

        return psycopg.connect(url, autocommit=True, connect_timeout=10)
    return connect


class MetricsStore:
    """`connect` returns a DB-API connection (psycopg in production, a fake in tests)."""

    def __init__(self, connect: Callable[[], object], background: bool = True):
        self.connect = connect
        self._read_lock = threading.Lock()
        self._reader = None
        conn = connect()
        try:
            cur = conn.cursor()
            cur.execute(SCHEMA)
            cur.execute(INDEX)
        finally:
            conn.close()
        self.q: queue.Queue = queue.Queue(maxsize=2000)
        self.background = background
        self.written = self.failed = 0
        self.last_error = None
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._writer, daemon=True, name="metrics-writer")
            self._thread.start()

    def status(self) -> dict:
        import sys
        import traceback
        t = self._thread
        frame = sys._current_frames().get(t.ident) if t else None
        where = [ln.strip().splitlines()[0] for ln in traceback.format_stack(frame)[-4:]] if frame else None
        return {"queued": self.q.qsize(), "written": self.written, "failed": self.failed,
                "last_error": self.last_error, "writer_alive": bool(t and t.is_alive()), "writer_at": where}

    @classmethod
    def from_url(cls, url: str) -> MetricsStore:
        if not url.startswith(("postgres://", "postgresql://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return cls(postgres_connect(url))

    # ---- writing
    def record(self, **event) -> None:
        item = (event.get("ts") or time.time(), *(event.get(k) for k in COLUMNS[1:]))
        if not self.background:
            self._insert([item])
            return
        try:
            self.q.put_nowait(item)
        except queue.Full:
            log.warning("metrics queue full, dropping one event")

    def _insert(self, items: list[tuple]) -> None:
        conn = self.connect()
        try:
            # One plain multi-row INSERT: psycopg's executemany() uses pipeline mode, and through Neon's
            # connection pooler that saved nothing and raised nothing on the live site.
            conn.cursor().execute(insert_sql(len(items)), [v for item in items for v in item])
        finally:
            conn.close()

    def _writer(self) -> None:
        while True:
            items = [self.q.get()]
            while len(items) < 200:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._insert(items)
                self.written += len(items)
            except Exception as e:  # metrics must never take the app down
                self.failed += len(items)
                self.last_error = type(e).__name__
                log.warning("metrics write failed, dropped %d events: %s: %s", len(items), type(e).__name__, e)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued events are written (used by tests)."""
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.1)

    # ---- reading
    def _query(self, sql: str, params=()) -> list[tuple]:
        with self._read_lock:
            for attempt in (1, 2):
                try:
                    if self._reader is None:
                        self._reader = self.connect()
                    cur = self._reader.cursor()
                    cur.execute(sql, params)
                    return cur.fetchall()
                except Exception:
                    self._reader = None  # reconnect once (Neon closes idle connections)
                    if attempt == 2:
                        raise
        return []

    def summary(self, hours: float = 24, cost_per_call: float = 0.0, now: float | None = None) -> dict:
        now = now or time.time()
        total_all, first_ts = self._query(SELECT_TOTALS)[0]
        rows = self._query(SELECT_WINDOW, (now - hours * 3600,))
        records = [dict(zip(COLUMNS, r, strict=True)) for r in rows]
        return {"backend": "postgres", "total_all_time": int(total_all or 0), "first_ts": first_ts,
                **summarize(records, hours, cost_per_call, now)}


def summarize(records: list[dict], hours: float, cost_per_call: float, now: float) -> dict:
    """Dashboard numbers from raw request records (pure function: easy to test)."""
    n = len(records)
    routes = Counter(r["routed_to"] for r in records)
    lat = sorted(r["latency_ms"] for r in records if r["latency_ms"] is not None)
    bucket_s = 3600 if hours > 6 else 600
    n_buckets = int(hours * 3600 // bucket_s)
    start = now - n_buckets * bucket_s
    timeline = [{"t": start + i * bucket_s, "small": 0, "llm": 0, "small_fallback": 0} for i in range(n_buckets)]
    for r in records:
        i = min(int((r["ts"] - start) // bucket_s), n_buckets - 1)  # a request at exactly `now` is in the last one
        if 0 <= i and r["routed_to"] in timeline[i]:
            timeline[i][r["routed_to"]] += 1
    llm_calls = routes.get("llm", 0)
    return {
        "hours": hours, "bucket_seconds": bucket_s, "total": n, "routes": dict(routes),
        "without_llm_share": round((n - llm_calls) / n, 4) if n else None,
        "languages": dict(Counter(r["language"] or "unknown" for r in records)),
        "top_intents": Counter(r["intent"] for r in records).most_common(8),
        "actions": dict(Counter(r["action"] for r in records if r["action"])),
        "voice": sum(1 for r in records if r["voice"]),
        "latency_ms": {"p50": round(statistics.median(lat), 1) if lat else None,
                       "p95": round(lat[max(0, int(0.95 * len(lat)) - 1)], 1) if lat else None},
        "cost_usd": {"router": round(llm_calls * cost_per_call, 6), "all_llm": round(n * cost_per_call, 6),
                     "saved": round((n - llm_calls) * cost_per_call, 6), "per_llm_call": cost_per_call},
        "timeline": timeline,
    }
