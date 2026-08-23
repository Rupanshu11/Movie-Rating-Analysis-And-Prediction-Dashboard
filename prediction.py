"""
prediction.py
--------------
Loads a trained model (Linear Regression or Random Forest) and uses
it to predict a movie's rating from user-supplied inputs:

    - Popularity
    - Vote Count
    - Release Year
    - Release Month

Popularity_per_Vote is calculated automatically from popularity and
vote_count, matching the feature engineering done in preprocessing.py.
"""

import os

import joblib
import pandas as pd

from utils import project_paths, rating_category

MODEL_FILENAMES = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
}


def load_model(model_name):
    """
    Load a trained model from disk by its display name.

    Parameters
    ----------
    model_name : str
        Either "Linear Regression" or "Random Forest".

    Returns
    -------
    sklearn estimator
        The loaded, trained model.

    Raises
    ------
    ValueError
        If model_name is not recognized.
    FileNotFoundError
        If the model file has not been trained/saved yet.
    """
    if model_name not in MODEL_FILENAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_FILENAMES)}."
        )

    paths = project_paths()
    model_path = os.path.join(paths["models"], MODEL_FILENAMES[model_name])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Run model_training.py first to train and save models."
        )

    return joblib.load(model_path)


def build_feature_row(popularity, vote_count, release_year, release_month):
    """
    Build a single-row DataFrame matching the feature schema used
    during training, automatically calculating Popularity_per_Vote.

    Parameters
    ----------
    popularity : float
    vote_count : int
    release_year : int
    release_month : int

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame ready to be passed to model.predict().
    """
    popularity_per_vote = popularity / (vote_count + 1)

    row = pd.DataFrame([{
        "popularity": popularity,
        "vote_count": vote_count,
        "release_year": release_year,
        "release_month": release_month,
        "Popularity_per_Vote": popularity_per_vote,
    }])
    return row


def predict_rating(model_name, popularity, vote_count, release_year, release_month):
    """
    Predict a movie's rating given raw user inputs.

    Returns
    -------
    dict
        {
            "predicted_rating": float,
            "model_used": str,
            "category": str,        # Poor / Average / Good / Excellent
            "popularity_per_vote": float,
        }
    """
    model = load_model(model_name)
    features = build_feature_row(popularity, vote_count, release_year, release_month)

    predicted_rating = float(model.predict(features)[0])
    predicted_rating = max(0.0, min(10.0, predicted_rating))  # clamp to valid rating range

    return {
        "predicted_rating": round(predicted_rating, 2),
        "model_used": model_name,
        "category": rating_category(predicted_rating),
        "popularity_per_vote": round(features["Popularity_per_Vote"].iloc[0], 4),
    }


if __name__ == "__main__":
    # Simple manual test of the prediction pipeline
    result = predict_rating(
        model_name="Random Forest",
        popularity=25.0,
        vote_count=1500,
        release_year=2023,
        release_month=6,
    )
    print(result)
