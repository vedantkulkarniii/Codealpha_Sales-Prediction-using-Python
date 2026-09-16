"""Preprocessing utilities for sales prediction."""

from __future__ import annotations

import pandas as pd


def clean_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values."""
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def standardize_text(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Trim and standardize category text values."""
    cleaned = df.copy()
    for column in columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).str.strip()
    return cleaned
