
import os


def ensure_dir(path):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)


def rating_category(rating):
    """
    Convert a numeric movie rating (0-10 scale) into a human-readable
    category label.

    Parameters
    ----------
    rating : float
        Predicted or actual movie rating.

    Returns
    -------
    str
        One of "Poor", "Average", "Good", or "Excellent".
    """
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return "Unknown"

    if rating < 4.0:
        return "Poor"
    elif rating < 6.0:
        return "Average"
    elif rating < 8.0:
        return "Good"
    else:
        return "Excellent"


def format_metric(value, decimals=3):
    """
    Round a numeric metric to a fixed number of decimal places for
    clean display in tables and dashboards.

    Parameters
    ----------
    value : float
        The metric value (e.g. MAE, MSE, R2).
    decimals : int
        Number of decimal places to keep.

    Returns
    -------
    float
        Rounded value.
    """
    return round(float(value), decimals)


def project_paths(base_dir=None):
    """
    Return a dictionary of standard project paths, resolved relative
    to the project root so scripts work regardless of the current
    working directory they are launched from.

    Parameters
    ----------
    base_dir : str, optional
        Root directory of the project. Defaults to the parent folder
        of this file's location (i.e. Movie_Rating_Analysis/).

    Returns
    -------
    dict
        Dictionary with keys: data, models, graphs, raw_csv, clean_csv.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = {
        "root": base_dir,
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
        "graphs": os.path.join(base_dir, "reports", "graphs"),
        "raw_csv": os.path.join(base_dir, "data", "Movie-Dataset-Latest.csv"),
        "clean_csv": os.path.join(base_dir, "data", "cleaned_dataset.csv"),
    }
    return paths
