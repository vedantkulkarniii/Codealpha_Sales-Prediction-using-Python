"""Prediction utilities for the sales prediction project."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from .data_cleaning import normalize_column_names
from .feature_engineering import build_features


def load_model_artifact(model_path: str) -> dict:
    """Load and validate a persisted model artifact."""
    artifact_path = Path(model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    artifact = joblib.load(artifact_path)
    required_keys = {"pipeline", "feature_columns", "target_column"}
    missing_keys = sorted(required_keys - set(artifact))
    if missing_keys:
        raise ValueError(f"Invalid model artifact; missing keys: {missing_keys}")
    return artifact


def predict_sales(
    input_data: pd.DataFrame,
    model_path: str = "models/sales_model.joblib",
) -> pd.DataFrame:
    """Predict sales for rows using a persisted training artifact."""
    artifact = load_model_artifact(model_path)
    features = build_features(normalize_column_names(input_data))
    expected_columns = artifact["feature_columns"]
    missing_columns = sorted(set(expected_columns) - set(features.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    predictions = artifact["pipeline"].predict(features[expected_columns])
    return pd.DataFrame({"predicted_sales": predictions}, index=input_data.index)
