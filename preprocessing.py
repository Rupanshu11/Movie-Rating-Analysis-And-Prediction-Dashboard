"""
preprocessing.py
----------------
Handles all data cleaning and feature engineering for the movie
dataset:

    * Removing duplicate records
    * Handling missing values
    * Converting release_date to a proper datetime type
    * Deriving release_year and release_month columns
    * Engineering the Popularity_per_Vote feature
    * Dropping columns that are not useful for analysis / modelling
    * Saving the cleaned dataset to disk

Running this file directly will read the raw dataset, clean it, and
write the cleaned CSV to data/cleaned_dataset.csv.
"""

import pandas as pd

from utils import project_paths, ensure_dir


REQUIRED_COLUMNS = [
    "title", "release_date", "popularity",
    "vote_average", "vote_count", "overview", "id",
]


def load_raw_data(csv_path):
    """
    Load the raw movie dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Raw, unprocessed dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing from the dataset.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Please place the raw "
            f"CSV file there before running preprocessing."
        ) from exc

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}"
        )

    return df


def remove_duplicates(df):
    """Remove exact duplicate rows and duplicate movie ids/titles."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["id"], keep="first")
    after = len(df)
    print(f"[preprocessing] Removed {before - after} duplicate rows.")
    return df.reset_index(drop=True)


def handle_missing_values(df):
    """
    Handle missing values in a sensible, column-appropriate way:

    * overview        -> fill with a generic placeholder string
    * vote_average     -> fill with the column median
    * vote_count       -> fill with 0 (assume no votes recorded)
    * popularity       -> fill with the column median
    * release_date     -> drop rows where it's missing (can't derive
                           year/month without it)
    """
    df = df.copy()

    df["overview"] = df["overview"].fillna("No overview available.")

    if df["vote_average"].isna().any():
        median_rating = df["vote_average"].median()
        df["vote_average"] = df["vote_average"].fillna(median_rating)

    if df["vote_count"].isna().any():
        df["vote_count"] = df["vote_count"].fillna(0)

    if df["popularity"].isna().any():
        median_pop = df["popularity"].median()
        df["popularity"] = df["popularity"].fillna(median_pop)

    before = len(df)
    df = df.dropna(subset=["release_date"])
    after = len(df)
    if before != after:
        print(f"[preprocessing] Dropped {before - after} rows with missing release_date.")

    return df.reset_index(drop=True)


def convert_dates(df):
    """
    Convert release_date to a proper datetime dtype and derive
    release_year and release_month columns from it.
    """
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # Drop any rows that failed to parse into a valid date
    before = len(df)
    df = df.dropna(subset=["release_date"])
    after = len(df)
    if before != after:
        print(f"[preprocessing] Dropped {before - after} rows with invalid release_date.")

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    return df.reset_index(drop=True)


def engineer_features(df):
    """
    Engineer additional features used for analysis and modelling:

    Popularity_per_Vote = popularity / (vote_count + 1)

    Adding 1 to vote_count avoids division-by-zero errors for movies
    that have not yet received any votes.
    """
    df = df.copy()
    df["Popularity_per_Vote"] = df["popularity"] / (df["vote_count"] + 1)
    return df


def drop_unnecessary_columns(df):
    """
    Drop columns that are not useful for downstream analysis or
    modelling. Currently this keeps the 'overview' column (useful for
    display in the app) but removes any leftover index columns.
    """
    df = df.copy()
    cols_to_drop = [c for c in df.columns if c.lower().startswith("unnamed")]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def clean_dataset(raw_csv_path=None, save_path=None):
    """
    Run the full preprocessing pipeline end-to-end and optionally save
    the result to disk.

    Parameters
    ----------
    raw_csv_path : str, optional
        Path to the raw dataset CSV. Defaults to the project's
        standard data/Movie-Dataset-Latest.csv path.
    save_path : str, optional
        Where to save the cleaned dataset. Defaults to the project's
        standard data/cleaned_dataset.csv path.

    Returns
    -------
    pandas.DataFrame
        The cleaned, feature-engineered dataset.
    """
    paths = project_paths()
    raw_csv_path = raw_csv_path or paths["raw_csv"]
    save_path = save_path or paths["clean_csv"]

    print(f"[preprocessing] Loading raw dataset from {raw_csv_path}")
    df = load_raw_data(raw_csv_path)
    print(f"[preprocessing] Raw dataset shape: {df.shape}")

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = convert_dates(df)
    df = engineer_features(df)
    df = drop_unnecessary_columns(df)

    print(f"[preprocessing] Cleaned dataset shape: {df.shape}")

    ensure_dir(paths["data"])
    df.to_csv(save_path, index=False)
    print(f"[preprocessing] Cleaned dataset saved to {save_path}")

    return df


if __name__ == "__main__":
    clean_dataset()
