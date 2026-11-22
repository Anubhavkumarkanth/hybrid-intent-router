"""Confidence-based routing between the small model and the LLM.

Rule: if calibrated confidence >= t, use the small model's answer; otherwise use the LLM's.
"""

from __future__ import annotations

import numpy as np

THRESHOLDS = [round(i / 100, 2) for i in range(101)] + [1.01]  # 1.01 = everything goes to the LLM


def route(conf: np.ndarray, small_pred: np.ndarray, llm_pred: np.ndarray, t: float):
    """Return (final predictions, boolean mask of items sent to the LLM)."""
    to_llm = conf < t
    final = np.where(to_llm, llm_pred, small_pred)
    return final, to_llm


def evaluate_threshold(conf, small_pred, llm_pred, gold, t) -> dict:
    final, to_llm = route(conf, small_pred, llm_pred, t)
    return {"t": t, "accuracy": float((final == gold).mean()), "llm_share": float(to_llm.mean())}


def sweep(conf, small_pred, llm_pred, gold, thresholds=THRESHOLDS) -> list[dict]:
    return [evaluate_threshold(conf, small_pred, llm_pred, gold, t) for t in thresholds]


def choose_threshold(curve: list[dict], llm_only_acc: float, tolerance: float = 0.01) -> dict:
    """Lowest LLM share whose accuracy >= LLM-only accuracy - tolerance.

    Ties on LLM share are broken by higher accuracy, then lower t. t = 1.01 (all to LLM)
    always reaches LLM-only accuracy, so a choice always exists.
    """
    target = llm_only_acc - tolerance
    ok = [r for r in curve if r["accuracy"] >= target - 1e-12]
    if not ok:
        raise ValueError("no threshold meets the target (curve must include t=1.01)")
    return min(ok, key=lambda r: (r["llm_share"], -r["accuracy"], r["t"]))


def oracle_accuracy(small_pred, llm_pred, gold) -> float:
    """Upper bound: correct if EITHER model is correct."""
    return float(((small_pred == gold) | (llm_pred == gold)).mean())
