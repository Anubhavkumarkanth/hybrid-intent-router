"""Temperature scaling (Guo et al., 2017) and Expected Calibration Error (ECE).

Temperature scaling divides all logits by one number T before softmax.
- T > 1 makes the model less confident, T < 1 more confident.
- The predicted class does not change, so accuracy does not change.
We pick T that minimises the negative log-likelihood (NLL) on ENGLISH dev logits.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 10.0) -> float:
    """Find T minimising NLL. NLL is smooth in log(T), so a coarse grid + golden-section search is enough."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    losses = [nll(logits, labels, t) for t in grid]
    i = int(np.argmin(losses))
    a = np.log(grid[max(i - 1, 0)])
    b = np.log(grid[min(i + 1, len(grid) - 1)])
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - phi * (b - a), a + phi * (b - a)
        if nll(logits, labels, np.exp(c)) < nll(logits, labels, np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    ECE = sum over bins of (bin size / N) * |accuracy in bin - mean confidence in bin|.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)
