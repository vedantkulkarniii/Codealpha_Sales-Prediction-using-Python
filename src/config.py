"""Typed configuration for repeatable model training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Settings that control a training run and its reproducibility."""

    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    min_samples_leaf: int = 1
    cv_folds: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")