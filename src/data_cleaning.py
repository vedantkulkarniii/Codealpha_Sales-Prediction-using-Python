"""Data cleaning utilities for the sales prediction project."""

from __future__ import annotations

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return a simple summary of the dataset structure."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
