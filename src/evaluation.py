"""Evaluation helpers for regression models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return stable, interpretable regression metrics."""
    actual_values = np.asarray(actual)
    predicted_values = np.asarray(predicted)
    non_zero_actual = actual_values != 0
    mape = (
        np.mean(
            np.abs(
                (actual_values[non_zero_actual] - predicted_values[non_zero_actual])
                / actual_values[non_zero_actual]
            )
        )
        if np.any(non_zero_actual)
        else 0.0
    )
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape": float(mape),
        "r2": float(r2_score(actual_values, predicted_values)),
    }
