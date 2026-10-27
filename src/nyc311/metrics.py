"""Forecast accuracy, interval and surge-alert metrics."""

from __future__ import annotations

import numpy as np


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def wape(y, yhat) -> float:
    y, yhat = _a(y), _a(yhat)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def mase(y, yhat, y_naive) -> float:
    """MAE(yhat) / MAE(naive); NaN if the naive MAE is 0."""
    y = _a(y)
    denom = np.abs(y - _a(y_naive)).mean()
    if denom == 0:
        return float("nan")
    return float(np.abs(y - _a(yhat)).mean() / denom)


def coverage(y, lo, hi) -> float:
    y = _a(y)
    return float(((y >= _a(lo)) & (y <= _a(hi))).mean())


def mean_width(lo, hi) -> float:
    return float((_a(hi) - _a(lo)).mean())


def surge_prf(actual_flag, alert_flag) -> dict[str, float]:
    a = np.asarray(actual_flag, dtype=bool)
    b = np.asarray(alert_flag, dtype=bool)
    tp = float((a & b).sum())
    precision = tp / b.sum() if b.sum() else 0.0
    recall = tp / a.sum() if a.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
