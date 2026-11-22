"""Routing, calibration and cost math on tiny hand-made examples."""

import numpy as np
import pytest

from router import calibrate as C
from router import costs
from router import route as R

# 4 messages. gold labels, small-model predictions with confidences, LLM predictions.
GOLD = np.array([0, 1, 2, 3])
SMALL = np.array([0, 1, 0, 0])  # right, right, wrong, wrong
CONF = np.array([0.95, 0.80, 0.40, 0.60])
LLM = np.array([0, 2, 2, -1])  # right, wrong, right, invalid(-1)


def test_t0_sends_nothing_to_llm():
    final, to_llm = R.route(CONF, SMALL, LLM, 0.0)
    assert not to_llm.any()
    assert (final == SMALL).all()


def test_t_above_max_sends_everything():
    final, to_llm = R.route(CONF, SMALL, LLM, 1.01)
    assert to_llm.all()
    assert (final == LLM).all()


def test_accuracy_math_by_hand():
    # t = 0.7: items 2 (0.40) and 3 (0.60) go to LLM -> final [0,1,2,-1] -> 3/4 correct
    r = R.evaluate_threshold(CONF, SMALL, LLM, GOLD, 0.7)
    assert r["llm_share"] == 0.5
    assert r["accuracy"] == 0.75
    assert R.evaluate_threshold(CONF, SMALL, LLM, GOLD, 0.0)["accuracy"] == 0.5
    assert R.evaluate_threshold(CONF, SMALL, LLM, GOLD, 1.01)["accuracy"] == 0.5
    assert R.oracle_accuracy(SMALL, LLM, GOLD) == 0.75


def test_choose_threshold_picks_lowest_share_meeting_target():
    curve = R.sweep(CONF, SMALL, LLM, GOLD)
    assert curve[-1]["t"] == 1.01 and len(curve) == 102
    # LLM-only acc = 0.5 -> target 0.49 -> t = 0 already qualifies (share 0)
    assert R.choose_threshold(curve, 0.5)["t"] == 0.0
    # Pretend LLM-only were 0.76 -> target 0.75 -> lowest share reaching 0.75 is share 0.25 (only item 2 routed)
    best = R.choose_threshold(curve, 0.76)
    assert best["accuracy"] == 0.75 and best["llm_share"] == 0.25


def test_choose_threshold_always_has_answer():
    curve = [{"t": 0.5, "accuracy": 0.1, "llm_share": 0.2}, {"t": 1.01, "accuracy": 0.9, "llm_share": 1.0}]
    assert R.choose_threshold(curve, 0.9)["t"] == 1.01


def test_temperature_scaling_recovers_known_T_and_keeps_argmax():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 2
    labels = np.array([rng.choice(5, p=p) for p in C.softmax(true_logits)])
    overconfident = true_logits * 3  # same ranking, too sharp -> best T is about 3
    T = C.fit_temperature(overconfident, labels)
    assert 2.6 < T < 3.4
    assert (C.softmax(overconfident, T).argmax(1) == overconfident.argmax(1)).all()
    assert C.ece(C.softmax(overconfident, T), labels) < C.ece(C.softmax(overconfident), labels)


def test_ece_perfect_and_bad():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert C.ece(probs, np.array([0, 1])) == pytest.approx(0.0)
    assert C.ece(probs, np.array([1, 0])) == pytest.approx(1.0)


def test_cost_math():
    # 1000 input tokens and 10 output tokens per call
    per_call = costs.cost_per_call(1000, 10)
    assert per_call == pytest.approx(1000 / 1e6 * costs.PRICE_INPUT_PER_M + 10 / 1e6 * costs.PRICE_OUTPUT_PER_M)
    assert costs.cost_per_1000(0.25, 1000, 10) == pytest.approx(250 * per_call)
    assert costs.avg_latency_ms(0.25, 20, 800) == pytest.approx(220)
