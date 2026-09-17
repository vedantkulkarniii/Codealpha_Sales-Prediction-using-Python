"""Evaluation helpers for regression models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return the standard regression metrics used by the project."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
