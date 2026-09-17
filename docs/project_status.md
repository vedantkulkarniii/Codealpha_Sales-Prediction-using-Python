# Project status

## Verified completion

The sales prediction project is now validated in the local environment.

### Confirmed working pieces
- dataset cleaning and duplicate handling
- missing-value imputation
- feature engineering for campaign duration and ad spend
- preprocessing with scaling and one-hot encoding
- Random Forest regression training
- evaluation metrics for MAE and R2
- saved model artifact loading for repeat predictions
- CLI-based training and prediction flow

### Validation status

The repository passes its unit tests with the project virtual environment:

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This produced 4 passing tests.

### End-to-end run

The pipeline was executed successfully with the sample CSV and generated a prediction output file:

```bash
.\.venv\Scripts\python.exe -m src.cli train data/raw/sample_sales.csv sales --model-path models/sales_model.joblib
.\.venv\Scripts\python.exe -m src.cli predict data/raw/sample_sales.csv --model-path models/sales_model.joblib --output-path outputs/predictions/cli_predictions.csv
```

### Notes

The repository is ready for a real dataset, but the current sample pipeline has already been validated locally.
