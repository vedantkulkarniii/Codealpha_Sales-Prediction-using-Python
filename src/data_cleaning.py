"""Data cleaning utilities for the sales prediction project."""

from __future__ import annotations

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without duplicate observations."""
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric gaps with medians and categorical gaps with a label."""
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns
    categorical_columns = cleaned.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in categorical_columns:
        cleaned[column] = cleaned[column].fillna("Unknown")
    return cleaned


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the standard cleaning steps used before feature engineering."""
    return fill_missing_values(drop_duplicate_rows(df))


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return a simple summary of the dataset structure."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
