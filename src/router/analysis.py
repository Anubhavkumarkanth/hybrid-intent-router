"""Phase 5: calibration + routing analysis + charts.

Decisions use DEV data only:
  - temperature T: fit on full English dev logits
  - threshold t:  chosen on pooled dev_sub (en + hi + kn)
Then the frozen T and t are applied once to test_sub for the final numbers.

Inputs : results/logits/*.npz (Phase 4), results/llm_cache.jsonl (Phase 3),
         results/llm_single_call.json, results/small_model.json
Outputs: results/routing.json, results/routing_curve.csv, results/charts/*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from router import calibrate as C
from router import costs
from router import route as R
from router.data import load_labels, load_subset
from router.evaluate import load_logits
from router.llm import INVALID, Cache

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
LANGS = ("en", "hi", "kn")
LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "pooled": "All 3 (pooled)"}
COLORS = {"en": "#2a6fdb", "hi": "#7b5ea7", "kn": "#1a9e5c", "pooled": "#333333"}
TOLERANCE = 0.01  # router may be at most 1.0 percentage point below LLM-only on dev_sub


def llm_preds(lang: str, sub: str, ids: np.ndarray, label_idx: dict, cache: Cache, model: str, pv: str):
    out = []
    for i in ids:
        rec = cache.get(model, pv, "batch", lang, str(i))
        if rec is None:
            raise RuntimeError(f"no cached LLM answer for {lang}/{sub}/{i}; run `make llm` first")
        out.append(label_idx.get(rec["intent"], -1) if rec["intent"] != INVALID else -1)
    return np.array(out)


def load_part(lang, sub, T, label_idx, cache, model, pv):
    d = load_logits(f"{lang}_{sub}")
    probs = C.softmax(d["logits"], T)
    return {
        "ids": d["ids"],
        "gold": d["labels"],
        "logits": d["logits"],
        "conf": probs.max(1),
        "small": probs.argmax(1),
        "llm": llm_preds(lang, sub, d["ids"], label_idx, cache, model, pv),
    }


def pool(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in ("gold", "conf", "small", "llm")}


def add_cost_latency(row: dict, stats: dict, small_ms: float) -> dict:
    row = dict(row)
    row["llm_cost_per_1000_usd"] = costs.cost_per_1000(row["llm_share"], stats["input_tokens"],
                                                       stats["output_tokens"])
    row["avg_latency_ms"] = costs.avg_latency_ms(row["llm_share"], small_ms, stats["llm_latency_ms"])
    return row


def summarize(part: dict, t: float, stats: dict, small_ms: float) -> dict:
    gold, conf, small, llm = part["gold"], part["conf"], part["small"], part["llm"]
    router = R.evaluate_threshold(conf, small, llm, gold, t)
    llm_only_cost = costs.cost_per_1000(1.0, stats["input_tokens"], stats["output_tokens"])
    return {
        "n": int(len(gold)),
        "small_only_acc": float((small == gold).mean()),
        "llm_only_acc": float((llm == gold).mean()),
        "router_acc": router["accuracy"],
        "oracle_acc": R.oracle_accuracy(small, llm, gold),
        "llm_share": router["llm_share"],
        "llm_invalid": int((llm == -1).sum()),
        "router_cost_per_1000_usd": costs.cost_per_1000(router["llm_share"], stats["input_tokens"],
                                                        stats["output_tokens"]),
        "llm_only_cost_per_1000_usd": llm_only_cost,
        "router_avg_latency_ms": costs.avg_latency_ms(router["llm_share"], small_ms, stats["llm_latency_ms"]),
        "llm_only_avg_latency_ms": stats["llm_latency_ms"],
        "small_only_avg_latency_ms": small_ms,
    }


def rnd(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: rnd(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rnd(v, nd) for v in obj]
    return obj


def main() -> None:
    labels = load_labels()
    label_idx = {name: i for i, name in enumerate(labels)}
    baseline = json.loads((RESULTS / "llm_baseline.json").read_text())
    model, pv = baseline["model"], baseline["prompt_version"]
    stats = costs.load_single_call_stats()
    small_ms = json.loads((RESULTS / "small_model.json").read_text())["cpu_latency_single_message"]["median_ms"]
    cache = Cache()

    # ---- 1. temperature on full English dev
    en_dev = load_logits("en_dev")
    T = C.fit_temperature(en_dev["logits"], en_dev["labels"])
    print(f"temperature T = {T:.4f}")

    # ---- 2. ECE before/after
    calib = {"temperature": T, "fit_on": "full English dev (2033 items)",
             "ece_en_dev_full": {"before": C.ece(C.softmax(en_dev["logits"]), en_dev["labels"]),
                                 "after": C.ece(C.softmax(en_dev["logits"], T), en_dev["labels"])},
             "ece_dev_sub": {}, "nll_en_dev_full": {"before": C.nll(en_dev["logits"], en_dev["labels"], 1.0),
                                                     "after": C.nll(en_dev["logits"], en_dev["labels"], T)}}
    dev = {lang: load_part(lang, "dev_sub", T, label_idx, cache, model, pv) for lang in LANGS}
    for lang in LANGS:
        lg, gold = dev[lang]["logits"], dev[lang]["gold"]
        calib["ece_dev_sub"][lang] = {"before": C.ece(C.softmax(lg), gold), "after": C.ece(C.softmax(lg, T), gold)}
    print("ECE dev_sub:", rnd(calib["ece_dev_sub"]))

    # ---- 3. sweep on dev_sub
    dev["pooled"] = pool([dev[lang] for lang in LANGS])
    curves = {}
    for name, part in dev.items():
        curves[name] = [add_cost_latency(r, stats, small_ms)
                        for r in R.sweep(part["conf"], part["small"], part["llm"], part["gold"])]

    # ---- 4. choose t on pooled dev_sub
    llm_only_dev = float((dev["pooled"]["llm"] == dev["pooled"]["gold"]).mean())
    chosen = R.choose_threshold(curves["pooled"], llm_only_dev, TOLERANCE)
    t = chosen["t"]
    small_alone_ok = t == 0.0
    print(f"LLM-only pooled dev_sub acc {llm_only_dev:.4f}; target {llm_only_dev - TOLERANCE:.4f}")
    print(f"chosen t = {t} (dev pooled acc {chosen['accuracy']:.4f}, LLM share {chosen['llm_share']:.4f})")

    # ---- 5. apply to test_sub (once)
    test = {lang: load_part(lang, "test_sub", T, label_idx, cache, model, pv) for lang in LANGS}
    test["pooled"] = pool([test[lang] for lang in LANGS])
    test_summary = {name: summarize(part, t, stats, small_ms) for name, part in test.items()}
    dev_summary = {name: summarize(part, t, stats, small_ms) for name, part in dev.items()}
    for name, s in test_summary.items():
        print(f"test_sub {name}: small {s['small_only_acc']:.4f} llm {s['llm_only_acc']:.4f} "
              f"router {s['router_acc']:.4f} oracle {s['oracle_acc']:.4f} share {s['llm_share']:.4f}")
    test_curves = {name: R.sweep(p["conf"], p["small"], p["llm"], p["gold"]) for name, p in test.items()}

    pooled = test_summary["pooled"]
    out = {
        "temperature": T,
        "calibration": calib,
        "llm": {"model": model, "prompt_version": pv, "single_call_stats": stats,
                "price_per_1m_input_usd": costs.PRICE_INPUT_PER_M,
                "price_per_1m_output_usd": costs.PRICE_OUTPUT_PER_M, "price_source": costs.PRICE_SOURCE},
        "small_model_cpu_latency_ms": small_ms,
        "threshold_rule": f"lowest LLM share with pooled dev_sub accuracy >= LLM-only dev_sub accuracy - "
                          f"{TOLERANCE * 100:.1f} pp",
        "llm_only_dev_sub_pooled_acc": llm_only_dev,
        "target_dev_sub_acc": llm_only_dev - TOLERANCE,
        "chosen_threshold": {k: chosen[k] for k in ("t", "accuracy", "llm_share", "llm_cost_per_1000_usd",
                                                    "avg_latency_ms")},
        "small_model_alone_meets_target_on_dev": small_alone_ok,
        "dev_sub": dev_summary,
        "test_sub": test_summary,
        "savings_test_sub_pooled": {
            "llm_calls_avoided_share": 1 - pooled["llm_share"],
            "cost_saving_share": 1 - pooled["router_cost_per_1000_usd"] / pooled["llm_only_cost_per_1000_usd"],
            "latency_saving_ms": pooled["llm_only_avg_latency_ms"] - pooled["router_avg_latency_ms"],
            "accuracy_change_pp": 100 * (pooled["router_acc"] - pooled["llm_only_acc"]),
        },
    }
    (RESULTS / "routing.json").write_text(json.dumps(rnd(out), indent=2))

    with open(RESULTS / "routing_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "group", "t", "accuracy", "llm_share", "llm_cost_per_1000_usd", "avg_latency_ms"])
        for name, rows in curves.items():
            for r in rows:
                w.writerow(["dev_sub", name, r["t"], round(r["accuracy"], 4), round(r["llm_share"], 4),
                            round(r["llm_cost_per_1000_usd"], 6), round(r["avg_latency_ms"], 2)])
        for name, rows in test_curves.items():
            for r in rows:
                r = add_cost_latency(r, stats, small_ms)
                w.writerow(["test_sub", name, r["t"], round(r["accuracy"], 4), round(r["llm_share"], 4),
                            round(r["llm_cost_per_1000_usd"], 6), round(r["avg_latency_ms"], 2)])

    CHARTS.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(curves, test_curves, t)
    plot_cost(test_summary)
    plot_reliability(dev, en_dev, T)
    example_table(test, labels, t)
    print("wrote results/routing.json, results/routing_curve.csv and charts")


# ------------------------------------------------------------------ charts

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_tradeoff(dev_curves, test_curves, t):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
    for ax, curves, title in ((axes[0], dev_curves, "dev_sub (used to choose t)"),
                              (axes[1], test_curves, "test_sub (final, t fixed)")):
        for name in ("en", "hi", "kn", "pooled"):
            rows = curves[name]
            ax.plot([100 * r["llm_share"] for r in rows], [100 * r["accuracy"] for r in rows],
                    color=COLORS[name], lw=2.4 if name == "pooled" else 1.6,
                    ls="--" if name == "pooled" else "-", label=LANG_NAMES[name])
            pt = next(r for r in rows if r["t"] == t)
            ax.plot(100 * pt["llm_share"], 100 * pt["accuracy"], "o", color=COLORS[name], ms=8,
                    mec="white", mew=1.5, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("% of messages sent to the LLM")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"Accuracy vs share sent to the LLM (dots = chosen threshold t = {t:.2f})")
    fig.tight_layout()
    fig.savefig(CHARTS / "accuracy_vs_llm_share.png")
    plt.close(fig)


def plot_cost(test_summary):
    plt = _plt()
    names = ["en", "hi", "kn", "pooled"]
    x = np.arange(len(names))
    llm = [test_summary[n]["llm_only_cost_per_1000_usd"] for n in names]
    rtr = [test_summary[n]["router_cost_per_1000_usd"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    b1 = ax.bar(x - 0.2, llm, 0.4, label="LLM only", color="#9aa5b8")
    b2 = ax.bar(x + 0.2, rtr, 0.4, label="Router", color="#2a6fdb")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"${b.get_height():.4f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, [LANG_NAMES[n] for n in names])
    ax.set_ylabel("LLM cost per 1000 messages (USD, paid tier)")
    ax.set_title("LLM cost per 1000 messages on test_sub: LLM only vs router")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "cost_per_1000.png")
    plt.close(fig)


def plot_reliability(dev, en_dev, T, n_bins=10):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, sharey=True)
    for ax, temp, title in ((axes[0], 1.0, "Before (T = 1)"), (axes[1], T, f"After temperature scaling (T = {T:.2f})")):
        ax.plot([0, 100], [0, 100], color="#999", ls=":", label="perfect calibration")
        for lang in LANGS:
            probs = C.softmax(dev[lang]["logits"], temp)
            conf, correct = probs.max(1), probs.argmax(1) == dev[lang]["gold"]
            edges = np.linspace(0, 1, n_bins + 1)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:
                    xs.append(100 * conf[m].mean())
                    ys.append(100 * correct[m].mean())
            ax.plot(xs, ys, "o-", color=COLORS[lang], label=LANG_NAMES[lang])
        ax.set_title(title)
        ax.set_xlabel("mean confidence in bin (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy in bin (%)")
    axes[1].legend(loc="upper left")
    fig.suptitle("Reliability on dev_sub (bins with >= 5 items)")
    fig.tight_layout()
    fig.savefig(CHARTS / "reliability.png")
    plt.close(fig)


def example_table(test, labels, t, n_each=4, seed=7):
    """8 test_sub examples: 4 the small model kept, 4 sent to the LLM, mixed languages. Drawn with Pillow."""
    from PIL import Image, ImageDraw

    from router.fonts import font_for, font_for_text

    rng = np.random.default_rng(seed)
    texts = {lang: {r["id"]: r["text"] for r in load_subset(lang, "test_sub")} for lang in LANGS}
    rows = []
    for kept in (True, False):
        cands = []
        for lang in LANGS:
            p = test[lang]
            for k in range(len(p["ids"])):
                if (p["conf"][k] >= t) == kept:
                    cands.append((lang, k))
        cands_by_lang = {lang: [c for c in cands if c[0] == lang] for lang in LANGS}
        picks = []
        order = ["en", "hi", "kn", "kn"] if kept else ["hi", "kn", "en", "hi"]
        for lang in order:
            pool_ = [c for c in cands_by_lang[lang] if c not in picks]
            if pool_:
                picks.append(pool_[rng.integers(len(pool_))])
        for lang, k in picks[:n_each]:
            p = test[lang]
            final = p["small"][k] if kept else p["llm"][k]
            rows.append({
                "lang": LANG_NAMES[lang], "text": texts[lang][str(p["ids"][k])],
                "gold": labels[p["gold"][k]], "small": labels[p["small"][k]],
                "conf": f"{100 * p['conf'][k]:.1f}%", "route": "small model" if kept else "LLM",
                "final": labels[final] if final >= 0 else "invalid", "ok": bool(final == p["gold"][k]),
            })
    (RESULTS / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    cols = [("Language", 120), ("Message", 560), ("True intent", 250), ("Small model (conf.)", 330),
            ("Answered by", 150), ("Final answer", 250), ("", 50)]
    W, row_h, pad = sum(w for _, w in cols) + 40, 64, 20
    H = 70 + row_h * (len(rows) + 1) + 30
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 18), f"Example test_sub messages (threshold t = {t:.2f})", font=font_for("latin", 26, True),
           fill="#1d2330")
    y = 70
    x = pad
    for name, w in cols:
        d.text((x, y + 20), name, font=font_for("latin", 19, True), fill="#5b6475")
        x += w
    y += row_h
    for i, r in enumerate(rows):
        if i % 2 == 0:
            d.rectangle([pad - 8, y, W - pad + 8, y + row_h], fill="#f3f5f8")
        cells = [r["lang"], r["text"], r["gold"], f"{r['small']} ({r['conf']})", r["route"], r["final"],
                 "OK" if r["ok"] else "X"]
        x = pad
        for (name, w), val in zip(cols, cells, strict=True):
            size = 19
            f = font_for_text(val, size)
            while d.textlength(val, font=f) > w - 14 and size > 11:
                size -= 1
                f = font_for_text(val, size)
            color = "#1d2330"
            if name == "Answered by":
                color = "#17602e" if val == "small model" else "#1d4fa8"
            if name == "":
                color = "#17602e" if val == "OK" else "#b3261e"
            d.text((x, y + row_h // 2), val, font=f, fill=color, anchor="lm")
            x += w
        y += row_h
    img.save(CHARTS / "examples_table.png")


if __name__ == "__main__":
    main()
