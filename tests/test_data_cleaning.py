import unittest

import pandas as pd

from src.data_cleaning import clean_dataset, summarize_dataframe


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


if __name__ == "__main__":
    unittest.main()
