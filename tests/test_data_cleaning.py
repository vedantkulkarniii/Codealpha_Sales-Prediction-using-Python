import unittest

import pandas as pd

from src.data_cleaning import clean_dataset, summarize_dataframe
from src.validation import validate_dataset


class DataCleaningTests(unittest.TestCase):
    def test_clean_dataset_removes_duplicates_and_fills_gaps(self):
        data = pd.DataFrame({"spend": [10.0, 10.0, None], "channel": ["TV", "TV", None]})

        cleaned = clean_dataset(data)

        self.assertEqual(len(cleaned), 2)
        self.assertFalse(cleaned.isna().any().any())

    def test_summary_reports_shape_and_duplicates(self):
        data = pd.DataFrame({"sales": [10, 10]})

        summary = summarize_dataframe(data)

        self.assertEqual(summary["shape"], (2, 1))
        self.assertEqual(summary["duplicate_rows"], 1)

    def test_validation_checks_target_and_required_columns(self):
        data = pd.DataFrame({"sales": [10], "channel": ["TV"]})

        validate_dataset(data, target_column="sales", required_columns=["channel"])
        with self.assertRaises(ValueError):
            validate_dataset(data, required_columns=["missing"])

    def test_validation_rejects_empty_and_non_numeric_targets(self):
        with self.assertRaises(ValueError):
            validate_dataset(pd.DataFrame(), target_column="sales")
        with self.assertRaises(TypeError):
            validate_dataset(pd.DataFrame({"sales": ["unknown"]}), target_column="sales")


if __name__ == "__main__":
    unittest.main()
