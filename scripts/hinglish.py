"""Hinglish (romanized, code-mixed Hindi + English) test: how does the router handle the way many Indian users
actually type? (make hinglish)

1. Build the test set: the 500 English test_sub sentences rewritten by Gemini as casual Hinglish ("kal subah
   6 baje alarm laga do"). The intent labels are MASSIVE's own, because each sentence keeps the meaning of its
   labelled English original. Saved to results/hinglish_test_sub.jsonl (resumable; written in batches of 20).
2. Classify it with the LLM (same frozen prompt, cached like every other LLM answer) and with the small model.
3. Route with the same T and t chosen on dev data (nothing is re-tuned on this set) -> results/hinglish.json.

Caveat written into the results: the same LLM family wrote the Hinglish and classifies it, which may flatter
the LLM-only number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router import calibrate as C  # noqa: E402
from router import llm as L  # noqa: E402
from router import route as R  # noqa: E402
from router.data import load_labels, load_subset  # noqa: E402
from router.llm_baseline import make_classifier  # noqa: E402

OUT = ROOT / "results" / "hinglish_test_sub.jsonl"
RESULT = ROOT / "results" / "hinglish.json"
PROMPT = (
    "Rewrite each numbered English message as a casual Hinglish message that an Indian user might type or say to "
    "a voice assistant: mostly Hindi words written in the Latin alphabet, mixed with English words where people "
    "naturally use them (app, alarm, music, playlist, email, numbers, names). Keep the exact same request and all "
    "details (times, names, places, numbers). Do not translate names. Do not add anything.\n\n"
    "Examples:\nwake me up at five am this week -> is hafte subah paanch baje mujhe utha dena\n"
    "what is the weather like today -> aaj mausam kaisa hai\n\nMessages:\n"
)
SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["n", "text"],
            },
        }
    },
    "required": ["items"],
}


def build(backend) -> list[dict]:
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    rows = load_subset("en", "test_sub")
    todo = [r for r in rows if r["id"] not in done]
    for start in range(0, len(todo), 20):
        chunk = todo[start : start + 20]
        prompt = PROMPT + "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(chunk))
        for attempt in range(3):
            resp = L.call_with_backoff(backend, prompt, SCHEMA)
            try:
                items = sorted(json.loads(resp.text)["items"], key=lambda x: x["n"])
                assert [x["n"] for x in items] == list(range(1, len(chunk) + 1))
                assert all(x["text"].strip() for x in items)
                break
            except (ValueError, KeyError, AssertionError) as e:
                print(f"  bad batch ({type(e).__name__}), retry {attempt + 1}")
        else:
            raise RuntimeError("Hinglish batch failed 3 times")
        with open(OUT, "a", encoding="utf-8") as f:
            for r, x in zip(chunk, items, strict=True):
                rec = {
                    "id": r["id"],
                    "text": x["text"].strip(),
                    "intent": r["intent"],
                    "label": r["label"],
                    "english": r["text"],
                }
                done[r["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  hinglish: {len(done)}/{len(rows)}", flush=True)
    return [done[r["id"]] for r in rows]


def small_logits(rows: list[dict]) -> np.ndarray:
    import torch

    from router.model import load_model, predict_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, meta = load_model(ROOT / "models" / "small", device=device)
    return predict_logits(model, tok, [r["text"] for r in rows], device=device, max_len=meta.get("max_len", 64)).numpy()


def main() -> int:
    L.load_env()
    backend = L.GeminiBackend()
    rows = build(backend)
    labels = load_labels()
    idx = {name: i for i, name in enumerate(labels)}
    gold = np.array([r["label"] for r in rows])
    routing = json.loads((ROOT / "results" / "routing.json").read_text())
    T, t = routing["temperature"], routing["chosen_threshold"]["t"]
    pv = routing["llm"]["prompt_version"]

    clf = make_classifier(pv, backend=backend)
    llm_answers = clf.classify_batch("hinglish", rows)
    llm = np.array([idx.get(a, -1) if a != L.INVALID else -1 for a in llm_answers])
    probs = C.softmax(small_logits(rows), T)
    small, conf = probs.argmax(1), probs.max(1)
    router = R.evaluate_threshold(conf, small, llm, gold, t)
    en = routing["test_sub"]["en"]
    out = {
        "n": len(rows),
        "threshold": t,
        "temperature": T,
        "small_only_acc": round(float((small == gold).mean()), 4),
        "llm_only_acc": round(float((llm == gold).mean()), 4),
        "router_acc": round(router["accuracy"], 4),
        "llm_share": round(router["llm_share"], 4),
        "oracle_acc": round(R.oracle_accuracy(small, llm, gold), 4),
        "llm_invalid": int((llm == -1).sum()),
        "english_same_sentences": {k: en[k] for k in ("small_only_acc", "llm_only_acc", "router_acc", "llm_share")},
        "how_made": "The 500 English test_sub sentences rewritten as casual Hinglish by the same Gemini model; intent "
        "labels are MASSIVE's. T and t are the ones chosen on dev data, not re-tuned.",
        "caveat": "The same LLM family wrote and classifies the Hinglish, which may flatter the LLM-only number.",
        "examples": [{"english": r["english"], "hinglish": r["text"], "intent": r["intent"]} for r in rows[:6]],
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
