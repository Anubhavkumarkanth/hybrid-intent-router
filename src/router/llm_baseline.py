"""Phase 3 runner: LLM prompt pilot, full baseline run, and single-message measurement.

    python -m router.llm_baseline check    # one tiny call to confirm the key + model work
    python -m router.llm_baseline pilot    # 2 prompt versions x 60 dev_sub items -> results/llm_pilot.json
    python -m router.llm_baseline run      # frozen prompt on dev_sub + test_sub, en/hi/kn -> llm_baseline.json
    python -m router.llm_baseline single   # 50 dev_sub items, one call each -> llm_single_call.json

All answers are cached in results/llm_cache.jsonl, so every command can be re-run safely.
Only dev data is used for decisions (pilot). Test data only produces final numbers.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

from router import llm as L
from router.data import load_labels, load_split, load_subset

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FROZEN = RESULTS / "llm_prompt_version.json"
LANGS = ("en", "hi", "kn")
SEED = 42


def train_examples(labels: list[str]) -> dict[str, str]:
    """One short English example per intent, from the TRAIN split (seeded choice of a short one)."""
    rng = random.Random(SEED)
    by_intent: dict[str, list[str]] = {}
    for r in load_split("en", "train"):
        by_intent.setdefault(r["intent"], []).append(r["text"])
    out = {}
    for lab in labels:
        texts = sorted(t for t in by_intent[lab] if 3 <= len(t.split()) <= 8) or sorted(by_intent[lab])
        out[lab] = rng.choice(texts)
    return out


EXAMPLES_FILE = RESULTS / "prompt_examples.json"


def prompt_examples(labels: list[str]) -> dict[str, str]:
    """The v2 prompt's examples. Read from results/prompt_examples.json when present, so a server
    (e.g. the public demo) does not need the training data; otherwise built from the train split."""
    if EXAMPLES_FILE.exists():
        return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    return train_examples(labels)


def cmd_examples() -> None:
    EXAMPLES_FILE.write_text(json.dumps(train_examples(load_labels()), ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"wrote {EXAMPLES_FILE.relative_to(ROOT)}")


def make_classifier(version: str, backend=None) -> L.Classifier:
    labels = load_labels()
    if backend is None:
        L.load_env()
        backend = L.GeminiBackend()
    ex = train_examples(labels) if version == "v2_labels_examples" else None
    return L.Classifier(backend=backend, labels=labels, prompt_version=version, examples=ex)


def acc(preds: list[str], rows: list[dict]) -> float:
    return sum(p == r["intent"] for p, r in zip(preds, rows, strict=True)) / len(rows)


def frozen_version() -> str:
    return json.loads(FROZEN.read_text())["prompt_version"]


def cmd_check() -> None:
    L.load_env()
    be = L.GeminiBackend()
    resp = be.generate('Reply with JSON {"ok": true}.',
                       {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    print(f"model={be.model} ok text={resp.text!r} in={resp.input_tokens} out={resp.output_tokens} "
          f"latency_ms={resp.latency_ms:.0f}")


def cmd_pilot() -> None:
    """60 dev_sub items (first 20 per language, same ids) x 2 prompt versions."""
    ids = None
    result = {"model": L.MODEL_ID, "n_per_lang": 20, "data": "dev_sub (first 20 ids)", "versions": {}}
    for version in L.PROMPT_VERSIONS:
        clf = make_classifier(version)
        per_lang = {}
        for lang in LANGS:
            rows = load_subset(lang, "dev_sub")[:20]
            ids = [r["id"] for r in rows]
            preds = clf.classify_batch(lang, rows)
            per_lang[lang] = round(acc(preds, rows), 4)
        pooled = round(sum(per_lang.values()) / len(per_lang), 4)
        result["versions"][version] = {"per_lang": per_lang, "pooled": pooled}
        print(version, per_lang, "pooled", pooled)
    result["ids"] = ids
    # Freeze the better one. Tie -> simpler prompt (v1, fewer tokens).
    best = max(L.PROMPT_VERSIONS, key=lambda v: (result["versions"][v]["pooled"], v == "v1_labels"))
    result["chosen"] = best
    (RESULTS / "llm_pilot.json").write_text(json.dumps(result, indent=2))
    FROZEN.write_text(json.dumps({"prompt_version": best, "model": L.MODEL_ID,
                                  "chosen_by": "pilot on 60 dev_sub items"}, indent=2))
    print("frozen prompt version:", best)


def cmd_run() -> None:
    version = frozen_version()
    clf = make_classifier(version)
    out = {"model": L.MODEL_ID, "prompt_version": version, "mode": "batch", "batch_size": L.BATCH_SIZE,
           "accuracy": {}, "invalid": {}, "n": {}}
    for sub in ("dev_sub", "test_sub"):
        for lang in LANGS:
            rows = load_subset(lang, sub)
            print(f"{sub} {lang}: {len(rows)} items")
            preds = clf.classify_batch(lang, rows)
            out["accuracy"][f"{lang}_{sub}"] = round(acc(preds, rows), 4)
            out["invalid"][f"{lang}_{sub}"] = sum(p == L.INVALID for p in preds)
            out["n"][f"{lang}_{sub}"] = len(rows)
            print(f"  acc {out['accuracy'][f'{lang}_{sub}']}")
        out["accuracy"][f"pooled_{sub}"] = round(
            sum(out["accuracy"][f"{lg}_{sub}"] for lg in LANGS) / len(LANGS), 4)
    batch_recs = [r for r in clf.cache.records() if r["prompt_version"] == version and r["mode"] == "batch"
                  and "batch_input_tokens" in r]
    out["batch_requests_cached"] = len({(r["lang"], r["batch_input_tokens"], r["batch_latency_ms"])
                                        for r in batch_recs})
    (RESULTS / "llm_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_single() -> None:
    """50 dev_sub items spread over languages (17 en, 17 hi, 16 kn), one request each."""
    version = frozen_version()
    clf = make_classifier(version)
    rng = random.Random(SEED)
    picked = rng.sample([r["id"] for r in load_subset("en", "dev_sub")], 50)
    plan = [("en", picked[:17]), ("hi", picked[17:34]), ("kn", picked[34:])]
    recs, agree, single_correct, batch_correct = [], 0, 0, 0
    for lang, id_list in plan:
        rows = {r["id"]: r for r in load_subset(lang, "dev_sub")}
        batch_preds = dict(zip(id_list, clf.classify_batch(lang, [rows[i] for i in id_list]), strict=True))
        for i in id_list:
            rec = clf.classify_single(lang, rows[i])
            recs.append(rec)
            gold = rows[i]["intent"]
            single_correct += rec["intent"] == gold
            batch_correct += batch_preds[i] == gold
            agree += rec["intent"] == batch_preds[i]
    lat = sorted(r["latency_ms"] for r in recs)
    n = len(recs)
    out = {
        "model": L.MODEL_ID,
        "prompt_version": version,
        "n": n,
        "per_lang_n": {lang: len(ids_) for lang, ids_ in plan},
        "latency_ms_median": round(statistics.median(lat), 1),
        "latency_ms_p95": round(lat[int(0.95 * n) - 1], 1),
        "latency_ms_mean": round(statistics.fmean(lat), 1),  # slow outliers (API overload) pull the mean up
        "mean_input_tokens": round(statistics.fmean(r["input_tokens"] for r in recs), 1),
        "mean_output_tokens": round(statistics.fmean(r["output_tokens"] for r in recs), 1),
        "mean_thinking_tokens": round(statistics.fmean(r["thinking_tokens"] for r in recs), 1),
        "single_call_accuracy": round(single_correct / n, 4),
        "batched_accuracy_same_items": round(batch_correct / n, 4),
        "accuracy_diff_single_minus_batch_pp": round(100 * (single_correct - batch_correct) / n, 2),
        "agreement_single_vs_batch": round(agree / n, 4),
    }
    (RESULTS / "llm_single_call.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmds = {"check": cmd_check, "pilot": cmd_pilot, "run": cmd_run, "single": cmd_single, "examples": cmd_examples}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: python -m router.llm_baseline {{{'|'.join(cmds)}}}")
    cmds[sys.argv[1]]()
