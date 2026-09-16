"""Feature engineering utilities for the sales prediction pipeline."""

from __future__ import annotations

import pandas as pd


def add_campaign_length(df: pd.DataFrame, start_col: str, end_col: str, output_col: str = "campaign_length_days") -> pd.DataFrame:
    """Create a simple duration feature based on start/end dates when available."""
    cleaned = df.copy()
    if start_col in cleaned.columns and end_col in cleaned.columns:
        cleaned[output_col] = (pd.to_datetime(cleaned[end_col]) - pd.to_datetime(cleaned[start_col])).dt.days
    return cleaned
