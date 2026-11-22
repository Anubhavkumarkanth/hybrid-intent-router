"""Read Locust's results/loadtest_stats.csv (Aggregated row) and write results/loadtest.json."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    stats = ROOT / "results" / "loadtest_stats.csv"
    with open(stats, newline="") as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    meta = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = {
        **meta,
        "requests": int(agg["Request Count"]),
        "failures": int(agg["Failure Count"]),
        "requests_per_sec": round(float(agg["Requests/s"]), 2),
        "p50_ms": float(agg["50%"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
        "avg_ms": round(float(agg["Average Response Time"]), 1),
        "max_ms": round(float(agg["Max Response Time"]), 1),
    }
    (ROOT / "results" / "loadtest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
