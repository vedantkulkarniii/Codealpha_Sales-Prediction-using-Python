"""Command-line interface for training and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .predict import predict_sales
from .train_model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a model")
    train_parser.add_argument("data_path")
    train_parser.add_argument("target_column")
    train_parser.add_argument("--model-path", default="models/sales_model.joblib")

    predict_parser = subparsers.add_parser("predict", help="Predict sales from a CSV")
    predict_parser.add_argument("data_path")
    predict_parser.add_argument("--model-path", default="models/sales_model.joblib")
    predict_parser.add_argument("--output-path", default="outputs/predictions/predictions.csv")

    arguments = parser.parse_args()
    if arguments.command == "train":
        print(train_model(arguments.data_path, arguments.target_column, arguments.model_path))
        return

    predictions = predict_sales(pd.read_csv(arguments.data_path), arguments.model_path)
    Path(arguments.output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(arguments.output_path, index=False)
    print(f"Saved predictions to {arguments.output_path}")


if __name__ == "__main__":
    main()
