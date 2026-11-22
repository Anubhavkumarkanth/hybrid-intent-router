"""Render result tables from results/*.json into README.md and FINAL_REPORT.md.

Each table lives between markers:
    <!-- RESULTS:<name>:START -->
    ...generated...
    <!-- RESULTS:<name>:END -->
Only files in results/ are read. A missing results file renders as "TBD".
`python scripts/render_tables.py`          rewrite the marked sections
`python scripts/check_numbers.py`          fail if any marked section is out of date
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = [ROOT / "README.md", ROOT / "FINAL_REPORT.md", ROOT / "docs" / "INTERVIEW_GUIDE.md"]
LANG = {"en": "English", "hi": "Hindi", "kn": "Kannada", "pooled": "All 3 (pooled)"}


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def usd(x: float) -> str:
    return f"${x:.4f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def t_small() -> str:
    s = load("small_model.json")
    if not s:
        return "TBD"
    a = s["accuracy"]
    rows = [[LANG[lg], pct(a[f"{lg}_test_full"]), pct(a[f"{lg}_test_sub"]), pct(a[f"{lg}_dev_sub"])]
            for lg in ("en", "hi", "kn")]
    lat = s["cpu_latency_single_message"]
    return (table(["Language", "Full test (2974)", "test_sub (500)", "dev_sub (300)"], rows)
            + f"\n\nModel: `{s['encoder']}` + linear head, trained on English only (best epoch "
              f"{s['best_epoch']}). {s['num_parameters_m']}M parameters, {s['model_size_mb']} MB on disk. "
              f"CPU latency for one message ({lat['threads']} threads): median {lat['median_ms']} ms, "
              f"p95 {lat['p95_ms']} ms.")


def t_llm() -> str:
    b, s = load("llm_baseline.json"), load("llm_single_call.json")
    if not b:
        return "TBD"
    a = b["accuracy"]
    rows = [[LANG[lg], pct(a[f"{lg}_dev_sub"]), pct(a[f"{lg}_test_sub"])] for lg in ("en", "hi", "kn")]
    rows.append([LANG["pooled"], pct(a["pooled_dev_sub"]), pct(a["pooled_test_sub"])])
    text = table(["Language", "dev_sub (300)", "test_sub (500)"], rows)
    text += f"\n\nModel `{b['model']}`, prompt `{b['prompt_version']}`, {b['batch_size']} messages per request."
    if s:
        text += (f" Single-message calls ({s['n']} dev_sub items): median latency {s['latency_ms_median']:.0f} ms, "
                 f"p95 {s['latency_ms_p95']:.0f} ms, on average {s['mean_input_tokens']:.0f} input and "
                 f"{s['mean_output_tokens']:.1f} output tokens per call. Accuracy single vs batched on the same "
                 f"items: {pct(s['single_call_accuracy'])} vs {pct(s['batched_accuracy_same_items'])}.")
    return text


def t_routing() -> str:
    r = load("routing.json")
    if not r:
        return "TBD"
    rows = []
    for g in ("en", "hi", "kn", "pooled"):
        s = r["test_sub"][g]
        rows.append([LANG[g], pct(s["small_only_acc"]), pct(s["llm_only_acc"]), f"**{pct(s['router_acc'])}**",
                     pct(s["oracle_acc"]), pct(s["llm_share"]), usd(s["router_cost_per_1000_usd"]),
                     usd(s["llm_only_cost_per_1000_usd"]), f"{s['router_avg_latency_ms']:.0f} ms",
                     f"{s['llm_only_avg_latency_ms']:.0f} ms"])
    c = r["chosen_threshold"]
    head = ["Language", "Small only", "LLM only", "Router", "Oracle", "Sent to LLM", "Router cost /1000",
            "LLM-only cost /1000", "Router avg latency", "LLM-only latency"]
    return (table(head, rows)
            + f"\n\ntest_sub, 500 messages per language. Threshold t = {c['t']:.2f} was chosen on dev_sub "
              f"(pooled dev accuracy {pct(c['accuracy'])}, target {pct(r['target_dev_sub_acc'])}, "
              f"{pct(c['llm_share'])} sent to LLM). Temperature T = {r['temperature']:.3f}.")


def t_savings() -> str:
    r = load("routing.json")
    if not r:
        return "TBD"
    s = r["savings_test_sub_pooled"]
    return table(["What (test_sub, all 3 languages)", "Value"], [
        ["LLM calls avoided", pct(s["llm_calls_avoided_share"])],
        ["LLM cost saved", pct(s["cost_saving_share"])],
        ["Average latency saved per message", f"{s['latency_saving_ms']:.0f} ms"],
        ["Accuracy change vs LLM only", f"{s['accuracy_change_pp']:+.1f} percentage points"],
    ])


def t_calibration() -> str:
    r = load("routing.json")
    if not r:
        return "TBD"
    c = r["calibration"]
    rows = [["English dev (full)", f"{c['ece_en_dev_full']['before']:.3f}", f"{c['ece_en_dev_full']['after']:.3f}"]]
    rows += [[f"{LANG[lg]} dev_sub", f"{c['ece_dev_sub'][lg]['before']:.3f}", f"{c['ece_dev_sub'][lg]['after']:.3f}"]
             for lg in ("en", "hi", "kn")]
    return (table(["Data", "ECE before (T = 1)", f"ECE after (T = {r['temperature']:.3f})"], rows)
            + "\n\nECE = expected calibration error (lower is better), 15 bins.")


def t_loadtest() -> str:
    lt = load("loadtest.json")
    if not lt:
        return "TBD"
    return table(["Metric", "Value"], [
        ["Setup", f"{lt.get('users', 'TBD')} users, spawn {lt.get('spawn_rate', 'TBD')}/s, "
                  f"{lt.get('duration', 'TBD')}, through NGINX, small model only"],
        ["Workers", f"{lt.get('workers', 'TBD')} gunicorn workers x {lt.get('torch_threads', 'TBD')} torch threads"],
        ["Requests", str(lt["requests"])],
        ["Failures", str(lt["failures"])],
        ["Requests per second", f"{lt['requests_per_sec']:.1f}"],
        ["p50 latency", f"{lt['p50_ms']:.0f} ms"],
        ["p95 latency", f"{lt['p95_ms']:.0f} ms"],
        ["p99 latency", f"{lt['p99_ms']:.0f} ms"],
    ])


def t_reference() -> str:
    ref, s = load("references.json"), load("small_model.json")
    if not ref or not s:
        return "TBD"
    a = s["accuracy"]
    rows = [[f"{e['model']} ({e['source']})", f"{e['hi']:.1f}%", f"{e['kn']:.1f}%"] for e in ref["entries"]]
    rows.append([f"**This project: {s['encoder']}** (English dev for model selection)",
                 f"**{pct(a['hi_test_full'])}**", f"**{pct(a['kn_test_full'])}**"])
    return (table(["Zero-shot from English, full test set", "Hindi", "Kannada"], rows)
            + "\n\nPublished numbers are copied from the papers (see `results/references.json`); they used "
              "different checkpoint selection, so compare with care.")


SECTIONS = {
    "small": t_small, "llm": t_llm, "routing": t_routing, "savings": t_savings,
    "calibration": t_calibration, "loadtest": t_loadtest, "reference": t_reference,
}
MARK = re.compile(r"(<!-- RESULTS:(\w+):START -->\n)(.*?)(\n<!-- RESULTS:\2:END -->)", re.S)


def render_text(text: str) -> str:
    def sub(m):
        name = m.group(2)
        if name not in SECTIONS:
            raise KeyError(f"unknown results section {name!r}")
        return m.group(1) + SECTIONS[name]() + m.group(4)

    return MARK.sub(sub, text)


def main() -> int:
    for doc in DOCS:
        if doc.exists():
            old = doc.read_text()
            new = render_text(old)
            if new != old:
                doc.write_text(new)
                print(f"updated {doc.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
