"""Cost and latency math for the router.

Prices are the PAID-tier prices for the LLM model (see RESEARCH.md, checked 2026-11-22).
Token counts and LLM latency come from the single-message measurement in
results/llm_single_call.json (real API usage metadata), not from guesses.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# USD per 1M tokens, paid tier, gemini-3.5-flash-lite (official pricing page, checked 2026-11-22).
PRICE_INPUT_PER_M = 0.30
PRICE_OUTPUT_PER_M = 2.50  # output price includes thinking tokens
PRICE_SOURCE = "https://aistudio.google.com/docs/pricing (checked 2026-11-22)"


def cost_per_call(input_tokens: float, output_tokens: float) -> float:
    return input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M


def cost_per_1000(llm_share: float, input_tokens: float, output_tokens: float) -> float:
    """LLM cost in USD for 1000 messages when `llm_share` of them go to the LLM."""
    return 1000 * llm_share * cost_per_call(input_tokens, output_tokens)


def avg_latency_ms(llm_share: float, small_ms: float, llm_ms: float, small_always_runs: bool = True) -> float:
    """Average latency per message. The router always runs the small model first."""
    return (small_ms if small_always_runs else 0.0) + llm_share * llm_ms


def load_single_call_stats(path: Path = ROOT / "results" / "llm_single_call.json") -> dict:
    d = json.loads(Path(path).read_text())
    return {
        "input_tokens": d["mean_input_tokens"],
        "output_tokens": d["mean_output_tokens"],
        "llm_latency_ms": d["latency_ms_median"],
    }
