# Sales Prediction using Python

A complete end-to-end sales forecasting project built with Python, Pandas, NumPy, Scikit-learn, and Matplotlib. The project trains a regression model to predict sales based on campaign and marketing features and exposes a reusable CLI for training and inference.

## Project objective

Develop a clean machine learning pipeline that can:
- clean and validate raw sales data,
- engineer useful marketing features,
- preprocess numeric and categorical inputs,
- train a prediction model,
- evaluate model quality,
- save and reuse the trained artifact for future predictions.

## Repository structure

- data/raw/ — sample raw dataset
- data/processed/ — processed or cleaned datasets
- notebooks/ — analysis and exploration notebooks
- src/ — reusable ML pipeline modules
- models/ — saved trained model artifacts
- outputs/predictions/ — generated sales predictions
- tests/ — validation tests for cleaning and pipeline behavior

## Verified pipeline

The project currently supports:
- duplicate removal and missing-data handling,
- campaign-length and ad-spend feature engineering,
- numeric scaling and categorical one-hot encoding,
- Random Forest regression training,
- MAE and R2 evaluation,
- model artifact persistence for repeated prediction,
- CLI-based training and prediction workflow.

## Setup

```powershell
cd C:\Users\VEDANT\OneDrive\Desktop\Task4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run locally

```powershell
.\.venv\Scripts\python.exe -m src.cli train data/raw/sample_sales.csv sales --model-path models/sales_model.joblib
.\.venv\Scripts\python.exe -m src.cli predict data/raw/sample_sales.csv --model-path models/sales_model.joblib --output-path outputs/predictions/cli_predictions.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Validation result

The project has been validated locally with real execution.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result:
- 4 tests executed
- 4 tests passed
- exit code 0

Real model execution also succeeded:
- MAE: 653.33
- R2: 0.9453

## Notes

This repository is ready for a real sales dataset and already demonstrates a complete end-to-end machine learning workflow using a sample dataset for local verification and testing.
