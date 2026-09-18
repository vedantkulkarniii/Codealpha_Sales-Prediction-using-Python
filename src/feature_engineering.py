"""Feature engineering utilities for the sales prediction pipeline."""

from __future__ import annotations

import pandas as pd


def add_campaign_length(df: pd.DataFrame, start_col: str, end_col: str, output_col: str = "campaign_length_days") -> pd.DataFrame:
    """Create a simple duration feature based on start/end dates when available."""
    cleaned = df.copy()
    if start_col in cleaned.columns and end_col in cleaned.columns:
        start_dates = pd.to_datetime(cleaned[start_col], errors="coerce")
        end_dates = pd.to_datetime(cleaned[end_col], errors="coerce")
        cleaned[output_col] = (end_dates - start_dates).dt.days
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
        spend = engineered[available_columns].apply(pd.to_numeric, errors="coerce")
        engineered[output_col] = spend.sum(axis=1, min_count=1)
    return engineered


def add_spend_concentration(
    df: pd.DataFrame,
    spend_column: str = "total_ad_spend",
    output_column: str = "largest_channel_share",
) -> pd.DataFrame:
    """Add the largest channel's share of the total advertising spend."""
    engineered = df.copy()
    channel_columns = ["tv_spend", "social_spend", "search_spend"]
    available_columns = [column for column in channel_columns if column in engineered.columns]
    if available_columns and spend_column in engineered.columns:
        channel_spend = engineered[available_columns].apply(pd.to_numeric, errors="coerce")
        total_spend = pd.to_numeric(engineered[spend_column], errors="coerce")
        engineered[output_column] = channel_spend.max(axis=1).div(total_spend.where(total_spend > 0))
    return engineered


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply optional date and advertising features supported by the dataset."""
    engineered = add_campaign_length(df, "campaign_start", "campaign_end")
    engineered = add_total_ad_spend(engineered, ["tv_spend", "social_spend", "search_spend"])
    return add_spend_concentration(engineered)
