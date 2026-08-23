"""
model_training.py
------------------
Trains and evaluates two regression models to predict a movie's
vote_average (rating):

    1. Linear Regression
    2. Random Forest Regressor

Both models use the following input features:

    - popularity
    - vote_count
    - release_year
    - release_month
    - Popularity_per_Vote

Models are evaluated with MAE, MSE, RMSE, and R^2 Score, compared in a
formatted table, and saved to disk using Joblib.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import project_paths, ensure_dir, format_metric

FEATURES = ["popularity", "vote_count", "release_year",
            "release_month", "Popularity_per_Vote"]
TARGET = "vote_average"


def load_cleaned_data(csv_path=None):
    """Load the cleaned dataset produced by preprocessing.py."""
    paths = project_paths()
    csv_path = csv_path or paths["clean_csv"]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{csv_path}'. "
            f"Run preprocessing.py first."
        )

    return pd.read_csv(csv_path)


def split_data(df, test_size=0.2, random_state=42):
    """Split the dataset into train/test sets using the model features."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained regression model.

    Returns
    -------
    dict
        Dictionary with MAE, MSE, RMSE, and R2 Score.
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": format_metric(mae),
        "MSE": format_metric(mse),
        "RMSE": format_metric(rmse),
        "R2 Score": format_metric(r2),
    }, predictions


def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train a Random Forest Regressor model."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_all(save_models=True):
    """
    Run the complete model training + evaluation pipeline for both
    Linear Regression and Random Forest, save trained models with
    Joblib, and return a comparison table plus predictions for
    diagnostic plots (actual vs predicted, residuals).

    Returns
    -------
    dict
        {
            "results": {"Linear Regression": {...}, "Random Forest": {...}},
            "comparison_df": pandas.DataFrame,
            "y_test": pandas.Series,
            "predictions": {"Linear Regression": np.ndarray, "Random Forest": np.ndarray},
            "feature_importance": pandas.Series (Random Forest only),
        }
    """
    paths = project_paths()
    df = load_cleaned_data()
    X_train, X_test, y_train, y_test = split_data(df)

    results = {}
    predictions = {}

    # ----- Linear Regression -----
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics, lr_preds = evaluate_model(lr_model, X_test, y_test)
    results["Linear Regression"] = lr_metrics
    predictions["Linear Regression"] = lr_preds

    # ----- Random Forest -----
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics, rf_preds = evaluate_model(rf_model, X_test, y_test)
    results["Random Forest"] = rf_metrics
    predictions["Random Forest"] = rf_preds

    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = "Model"

    feature_importance = pd.Series(
        rf_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    if save_models:
        ensure_dir(paths["models"])
        joblib.dump(lr_model, os.path.join(paths["models"], "linear_regression.pkl"))
        joblib.dump(rf_model, os.path.join(paths["models"], "random_forest.pkl"))

        # Save metrics + feature importance alongside models for the dashboard
        metadata = {
            "features": FEATURES,
            "target": TARGET,
            "metrics": results,
            "feature_importance": feature_importance.to_dict(),
        }
        with open(os.path.join(paths["models"], "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[model_training] Models and metadata saved to {paths['models']}")

    return {
        "results": results,
        "comparison_df": comparison_df,
        "y_test": y_test,
        "predictions": predictions,
        "feature_importance": feature_importance,
    }


def print_comparison_table(comparison_df):
    """Pretty-print the model comparison table to the console."""
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string())
    print("=" * 60 + "\n")


if __name__ == "__main__":
    output = train_and_evaluate_all()
    print_comparison_table(output["comparison_df"])
    print("Feature Importance (Random Forest):")
    print(output["feature_importance"].to_string())
