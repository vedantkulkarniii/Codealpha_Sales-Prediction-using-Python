"""Validation helpers for tabular sales datasets."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    required_columns: Iterable[str] = (),
) -> None:
    """Raise a clear error when a dataset cannot enter the pipeline."""
    if df.empty:
        raise ValueError("Dataset must contain at least one row")

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if target_column and target_column not in df.columns:
        missing_columns.append(target_column)
    if missing_columns:
        missing = ", ".join(sorted(set(missing_columns)))
        raise ValueError(f"Missing required columns: {missing}")

    if target_column and not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(f"Target column must be numeric: {target_column}")
