"""Model training entry point for the sales prediction project."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import TrainingConfig
from .data_cleaning import clean_dataset, load_data
from .evaluation import evaluate_regression
from .feature_engineering import build_features
from .preprocessing import build_preprocessor
from .validation import validate_dataset


def train_model(
    data_path: str,
    target_column: str,
    model_path: str = "models/sales_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    """Train, evaluate, and persist a sales regression pipeline."""
    training_config = config or TrainingConfig(test_size=test_size, random_state=random_state)
    dataset = build_features(clean_dataset(load_data(data_path)))
    validate_dataset(dataset, target_column=target_column)
    if len(dataset) < 5:
        raise ValueError("At least five rows are required for training")

    features = dataset.drop(columns=[target_column])
    target = dataset[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=training_config.test_size,
        random_state=training_config.random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=training_config.n_estimators,
                    min_samples_leaf=training_config.min_samples_leaf,
                    random_state=training_config.random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    artifact = {
        "pipeline": pipeline,
        "target_column": target_column,
        "feature_columns": features.columns.tolist(),
    }
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return evaluate_regression(y_test, predictions)
