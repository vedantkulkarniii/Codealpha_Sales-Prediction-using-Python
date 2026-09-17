import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.predict import predict_sales
from src.train_model import train_model


class PipelineTests(unittest.TestCase):
    def test_training_and_prediction_share_the_same_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "sales_model.joblib"
            metrics = train_model(
                "data/raw/sample_sales.csv",
                "sales",
                str(model_path),
            )
            inputs = pd.read_csv("data/raw/sample_sales.csv").drop(columns=["sales"]).head(2)

            predictions = predict_sales(inputs, str(model_path))

            self.assertTrue(model_path.exists())
            self.assertIn("mae", metrics)
            self.assertIn("r2", metrics)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(list(predictions.columns), ["predicted_sales"])


if __name__ == "__main__":
    unittest.main()
