# Sales Prediction using Python

This repository is being developed as a complete end-to-end sales prediction project using Python, Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Project objective

Build a professional sales forecasting pipeline that predicts future sales from marketing inputs such as advertising spend, platform, and target segment while following clean machine learning practices.

## Current project structure

- `data/raw/` - raw dataset files
- `data/processed/` - cleaned and transformed datasets
- `notebooks/` - analysis and modeling notebooks
- `src/` - reusable Python modules for cleaning, preprocessing, feature engineering, training, and prediction
- `models/` - trained model artifacts
- `outputs/figures/` - generated charts and visualizations
- `outputs/predictions/` - prediction outputs

## Current pipeline

The reusable pipeline now supports:

- duplicate removal and missing-value handling
- campaign duration and advertising-spend features
- numeric scaling and categorical one-hot encoding
- random-forest training with MAE and R2 evaluation
- persisted model loading for repeatable predictions

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

The repository includes a small sample dataset for local verification. Replace it with the project dataset before interpreting model quality.

## Run the pipeline

```bash
python -m src.cli train data/raw/sample_sales.csv sales --model-path models/sales_model.joblib
python -m src.cli predict data/raw/sample_sales.csv --model-path models/sales_model.joblib --output-path outputs/predictions/predictions.csv
python -m unittest discover -s tests -v
```
