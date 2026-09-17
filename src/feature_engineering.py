"""Feature engineering utilities for the sales prediction pipeline."""

from __future__ import annotations

import pandas as pd


def add_campaign_length(df: pd.DataFrame, start_col: str, end_col: str, output_col: str = "campaign_length_days") -> pd.DataFrame:
    """Create a simple duration feature based on start/end dates when available."""
    cleaned = df.copy()
    if start_col in cleaned.columns and end_col in cleaned.columns:
        cleaned[output_col] = (pd.to_datetime(cleaned[end_col]) - pd.to_datetime(cleaned[start_col])).dt.days
    return cleaned


def add_total_ad_spend(
    df: pd.DataFrame,
    spend_columns: list[str],
    output_col: str = "total_ad_spend",
) -> pd.DataFrame:
    """Combine available advertising channel spend into one feature."""
    engineered = df.copy()
    available_columns = [column for column in spend_columns if column in engineered.columns]
    if available_columns:
        engineered[output_col] = engineered[available_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return engineered


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply optional date and advertising features supported by the dataset."""
    engineered = add_campaign_length(df, "campaign_start", "campaign_end")
    return add_total_ad_spend(engineered, ["tv_spend", "social_spend", "search_spend"])
