# House Prices ML Pipeline

This project builds a complete machine learning workflow for Kaggle's **House Prices: Advanced Regression Techniques** dataset.

It includes:

- EDA charts and missing-value summaries
- Feature extraction / feature engineering
- Preprocessing for numeric and categorical columns
- Linear Regression
- Decision Tree
- Gradient Boosting
- Random Forest
- K-Means clustering analysis
- Ensemble learning with Voting and Stacking regressors
- Optional neural network with dense layers, dropout, and a freeze/fine-tune step
- Separate model reports plus one final comparison table

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Place `train.csv` and, optionally, `test.csv` in this folder:

```text
c:\Users\hp\Downloads\ml
```

You can also pass a direct zip/csv URL:

```powershell
python house_prices_ml.py --data-url "PASTE_DIRECT_DOWNLOAD_LINK_HERE"
```

Kaggle competition downloads usually require login, so the most reliable way is to download the zip from the Kaggle page and put it in this folder. The script can also unzip `house-prices-advanced-regression-techniques.zip` if it is present.

## Install

```powershell
python -m pip install -r requirements.txt
```

If TensorFlow is hard to install on your machine, the main sklearn models still work. Run:

```powershell
python house_prices_ml.py --skip-neural-net
```

## Run

```powershell
python house_prices_ml.py
```

Outputs are written to:

```text
outputs/
```

Important files:

- `outputs/model_comparison.csv`
- `outputs/classification_reports.txt`
- `outputs/regression_reports.txt`
- `outputs/kmeans_report.txt`
- `outputs/submission_ensemble.csv`, if `test.csv` is available
- `outputs/plots/*.png`

## Note About Accuracy

House price prediction is a regression problem, not a classification problem. Normal classification accuracy does not directly apply.

This project reports:

- `RMSE`, `MAE`, and `R2` for correct regression evaluation
- `Accuracy_Within_10pct` and `Accuracy_Within_20pct`, meaning the percent of predictions close to the real price
- A classification report by converting prices into `Low`, `Medium`, and `High` bins

