import argparse
import math
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="House Prices ML end-to-end pipeline")
    parser.add_argument("--train-path", default=str(ROOT / "train.csv"))
    parser.add_argument("--test-path", default=str(ROOT / "test.csv"))
    parser.add_argument("--data-url", default=None, help="Direct URL to train.csv or a zip containing train.csv.")
    parser.add_argument("--skip-neural-net", action="store_true", help="Skip TensorFlow neural network model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(data_url):
    if not data_url:
        return

    target = ROOT / Path(data_url.split("?")[0]).name
    if not target.suffix:
        target = ROOT / "downloaded_dataset"

    print(f"Downloading dataset from {data_url}")
    urllib.request.urlretrieve(data_url, target)

    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as archive:
            archive.extractall(ROOT)
        print(f"Extracted {target.name}")
    elif target.name != "train.csv" and target.suffix.lower() == ".csv":
        shutil.copy2(target, ROOT / "train.csv")
        print("Copied downloaded csv to train.csv")


def maybe_unzip_local_dataset():
    zip_candidates = [
        ROOT / "house-prices-advanced-regression-techniques.zip",
        ROOT / "house-prices.zip",
    ]
    for candidate in zip_candidates:
        if candidate.exists() and zipfile.is_zipfile(candidate):
            with zipfile.ZipFile(candidate) as archive:
                archive.extractall(ROOT)
            print(f"Extracted {candidate.name}")
            return


def load_data(args):
    download_dataset(args.data_url)
    maybe_unzip_local_dataset()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)

    if not train_path.exists():
        raise FileNotFoundError(
            "train.csv was not found. Download the Kaggle dataset from "
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data "
            "and place train.csv in this folder, or pass --data-url with a direct download link."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path.exists() else None
    return train, test


def engineer_features(df):
    data = df.copy()

    fill_zero_columns = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "GarageArea",
        "GarageCars",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
    ]
    for col in fill_zero_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(data.columns):
        data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(data.columns):
        data["TotalBath"] = (
            data["FullBath"]
            + 0.5 * data["HalfBath"]
            + data["BsmtFullBath"]
            + 0.5 * data["BsmtHalfBath"]
        )
    if {"WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}.issubset(data.columns):
        data["TotalPorchSF"] = (
            data["WoodDeckSF"]
            + data["OpenPorchSF"]
            + data["EnclosedPorch"]
            + data["3SsnPorch"]
            + data["ScreenPorch"]
        )
    if {"YrSold", "YearBuilt"}.issubset(data.columns):
        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    if {"YrSold", "YearRemodAdd"}.issubset(data.columns):
        data["RemodelAge"] = data["YrSold"] - data["YearRemodAdd"]
    if "GarageArea" in data.columns:
        data["HasGarage"] = (data["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in data.columns:
        data["HasBasement"] = (data["TotalBsmtSF"] > 0).astype(int)
    if "PoolArea" in data.columns:
        data["HasPool"] = (data["PoolArea"] > 0).astype(int)
    if "2ndFlrSF" in data.columns:
        data["HasSecondFloor"] = (data["2ndFlrSF"] > 0).astype(int)

    return data


def run_eda(train):
    missing = (
        train.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )
    missing["missing_percent"] = missing["missing_count"] / len(train) * 100
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    numeric = train.select_dtypes(include=[np.number])
    if "SalePrice" in train.columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(train["SalePrice"], kde=True)
        plt.title("SalePrice Distribution")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "saleprice_distribution.png")
        plt.close()

    if "SalePrice" in numeric.columns:
        corr = numeric.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False).head(15)
        corr.to_csv(OUTPUT_DIR / "top_correlations.csv")
        plt.figure(figsize=(10, 7))
        sns.barplot(x=corr.values, y=corr.index)
        plt.title("Top Numeric Correlations With SalePrice")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "top_correlations.png")
        plt.close()

        top_cols = corr.index[:10].tolist()
        plt.figure(figsize=(10, 8))
        sns.heatmap(train[top_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
        plt.close()


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def price_bins(values, thresholds=None):
    values = np.asarray(values)
    labels = np.array(["Low", "Medium", "High"], dtype=object)
    q1, q2 = thresholds if thresholds is not None else np.quantile(values, [0.33, 0.66])
    return labels[np.digitize(values, bins=[q1, q2], right=True)]


def regression_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)
    abs_pct_error = np.abs(y_true - y_pred) / np.maximum(y_true, 1)
    return {
        "RMSE_Log": math.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "RMSE_Dollars": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE_Dollars": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true_log, y_pred_log),
        "Accuracy_Within_10pct": float(np.mean(abs_pct_error <= 0.10)),
        "Accuracy_Within_20pct": float(np.mean(abs_pct_error <= 0.20)),
    }


def classification_summary(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    thresholds = np.quantile(y_true, [0.33, 0.66])
    true_bins = price_bins(y_true, thresholds)
    pred_bins = price_bins(y_pred, thresholds)
    return {
        "accuracy": accuracy_score(true_bins, pred_bins),
        "report": classification_report(true_bins, pred_bins, zero_division=0),
    }


def build_models():
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def run_sklearn_models(X_train, X_valid, y_train, y_valid, preprocessor):
    models = build_models()
    fitted = {}
    rows = []
    regression_text = []
    classification_text = []

    for name, model in models.items():
        print(f"Training {name}")
        pipeline = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        metrics = regression_metrics(y_valid, preds)
        class_info = classification_summary(y_valid, preds)

        rows.append({"Model": name, **metrics, "Binned_Price_Accuracy": class_info["accuracy"]})
        regression_text.append(f"\n{name}\n{pd.Series(metrics).to_string()}\n")
        classification_text.append(f"\n{name}\nAccuracy: {class_info['accuracy']:.4f}\n{class_info['report']}\n")
        fitted[name] = pipeline

    return fitted, rows, regression_text, classification_text


def run_kmeans(X_train, X_valid, y_valid, preprocessor):
    print("Running K-Means clustering")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    kmeans.fit(X_train_processed)
    clusters = kmeans.predict(X_valid_processed)
    true_prices = np.expm1(y_valid)
    true_bins = price_bins(true_prices, np.quantile(true_prices, [0.33, 0.66]))

    cluster_to_label = {}
    for cluster_id in np.unique(clusters):
        labels, counts = np.unique(true_bins[clusters == cluster_id], return_counts=True)
        cluster_to_label[cluster_id] = labels[np.argmax(counts)]

    pred_bins = np.array([cluster_to_label[cluster] for cluster in clusters])
    acc = accuracy_score(true_bins, pred_bins)
    report = classification_report(true_bins, pred_bins, zero_division=0)

    try:
        silhouette = silhouette_score(X_valid_processed, clusters)
    except ValueError:
        silhouette = np.nan

    text = (
        "KMeans clustering report\n"
        f"Cluster-to-label mapping: {cluster_to_label}\n"
        f"Binned price accuracy: {acc:.4f}\n"
        f"Silhouette score: {silhouette:.4f}\n\n"
        f"{report}\n"
    )
    (OUTPUT_DIR / "kmeans_report.txt").write_text(text, encoding="utf-8")
    joblib.dump(Pipeline(steps=[("preprocess", preprocessor), ("kmeans", kmeans)]), OUTPUT_DIR / "kmeans_pipeline.joblib")
    return {
        "Model": "KMeans",
        "RMSE_Log": np.nan,
        "RMSE_Dollars": np.nan,
        "MAE_Dollars": np.nan,
        "R2": np.nan,
        "Accuracy_Within_10pct": np.nan,
        "Accuracy_Within_20pct": np.nan,
        "Binned_Price_Accuracy": acc,
    }


def run_ensemble_models(X_train, X_valid, y_train, y_valid, preprocessor):
    base_models = build_models()
    estimators = [
        ("linear", base_models["LinearRegression"]),
        ("tree", base_models["DecisionTree"]),
        ("gbr", base_models["GradientBoosting"]),
        ("rf", base_models["RandomForest"]),
    ]
    ensemble_models = {
        "VotingEnsemble": VotingRegressor(estimators=estimators, n_jobs=1),
        "StackingEnsemble": StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0),
            cv=3,
            n_jobs=1,
            passthrough=False,
        ),
    }

    fitted = {}
    rows = []
    regression_text = []
    classification_text = []

    for name, model in ensemble_models.items():
        print(f"Training {name}")
        pipeline = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        metrics = regression_metrics(y_valid, preds)
        class_info = classification_summary(y_valid, preds)
        rows.append({"Model": name, **metrics, "Binned_Price_Accuracy": class_info["accuracy"]})
        regression_text.append(f"\n{name}\n{pd.Series(metrics).to_string()}\n")
        classification_text.append(f"\n{name}\nAccuracy: {class_info['accuracy']:.4f}\n{class_info['report']}\n")
        fitted[name] = pipeline

    return fitted, rows, regression_text, classification_text


def run_neural_net(X_train, X_valid, y_train, y_valid, preprocessor):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as exc:
        print(f"Skipping NeuralNetwork because TensorFlow could not be imported: {exc}")
        return None, [], [], []

    print("Training NeuralNetwork with dense layers, dropout, and freeze/fine-tune step")
    nn_preprocessor = clone(preprocessor)
    X_train_processed = nn_preprocessor.fit_transform(X_train).astype("float32")
    X_valid_processed = nn_preprocessor.transform(X_valid).astype("float32")

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = keras.Sequential(
        [
            layers.Input(shape=(X_train_processed.shape[1],)),
            layers.Dense(256, activation="relu", name="feature_layer_1"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu", name="feature_layer_2"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5),
    ]
    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_valid_processed, y_valid),
        epochs=250,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    model.get_layer("feature_layer_1").trainable = False
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss="mse", metrics=["mae"])
    model.fit(
        X_train_processed,
        y_train,
        validation_data=(X_valid_processed, y_valid),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    preds = model.predict(X_valid_processed, verbose=0).reshape(-1)
    metrics = regression_metrics(y_valid, preds)
    class_info = classification_summary(y_valid, preds)
    model.save(OUTPUT_DIR / "neural_network.keras")
    joblib.dump(nn_preprocessor, OUTPUT_DIR / "neural_network_preprocessor.joblib")

    row = {"Model": "NeuralNetwork_Dropout_FrozenLayer", **metrics, "Binned_Price_Accuracy": class_info["accuracy"]}
    regression_text = [f"\nNeuralNetwork_Dropout_FrozenLayer\n{pd.Series(metrics).to_string()}\n"]
    classification_text = [
        f"\nNeuralNetwork_Dropout_FrozenLayer\nAccuracy: {class_info['accuracy']:.4f}\n{class_info['report']}\n"
    ]
    return model, [row], regression_text, classification_text


def make_submission(best_pipeline, test_df):
    if test_df is None:
        return

    test_features = engineer_features(test_df)
    ids = test_features["Id"] if "Id" in test_features.columns else pd.Series(np.arange(1, len(test_features) + 1))
    if "SalePrice" in test_features.columns:
        test_features = test_features.drop(columns=["SalePrice"])
    preds = best_pipeline.predict(test_features)
    submission = pd.DataFrame({"Id": ids, "SalePrice": np.maximum(np.expm1(preds), 0)})
    submission.to_csv(OUTPUT_DIR / "submission_ensemble.csv", index=False)


def main():
    args = parse_args()
    ensure_dirs()

    train, test = load_data(args)
    train = engineer_features(train)
    test = engineer_features(test) if test is not None else None
    run_eda(train)

    if "SalePrice" not in train.columns:
        raise ValueError("train.csv must include SalePrice.")

    X = train.drop(columns=["SalePrice"])
    y = np.log1p(train["SalePrice"])
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )
    preprocessor = build_preprocessor(X_train)

    fitted_models, rows, regression_text, classification_text = run_sklearn_models(
        X_train, X_valid, y_train, y_valid, preprocessor
    )
    kmeans_row = run_kmeans(X_train, X_valid, y_valid, clone(preprocessor))
    rows.append(kmeans_row)

    ensemble_models, ensemble_rows, ensemble_reg_text, ensemble_class_text = run_ensemble_models(
        X_train, X_valid, y_train, y_valid, preprocessor
    )
    fitted_models.update(ensemble_models)
    rows.extend(ensemble_rows)
    regression_text.extend(ensemble_reg_text)
    classification_text.extend(ensemble_class_text)

    if not args.skip_neural_net:
        _, nn_rows, nn_reg_text, nn_class_text = run_neural_net(
            X_train, X_valid, y_train.to_numpy(), y_valid.to_numpy(), preprocessor
        )
        rows.extend(nn_rows)
        regression_text.extend(nn_reg_text)
        classification_text.extend(nn_class_text)

    comparison = pd.DataFrame(rows).sort_values(
        by=["RMSE_Log", "R2"],
        ascending=[True, False],
        na_position="last",
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "regression_reports.txt").write_text("\n".join(regression_text), encoding="utf-8")
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n".join(classification_text), encoding="utf-8")

    best_model_name = comparison.dropna(subset=["RMSE_Log"]).iloc[0]["Model"]
    best_pipeline = fitted_models.get(best_model_name)
    if best_pipeline is not None:
        joblib.dump(best_pipeline, OUTPUT_DIR / f"{best_model_name}_pipeline.joblib")
        make_submission(best_pipeline, test)

    print("\nFinal model comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model by RMSE_Log: {best_model_name}")
    print(f"Outputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
