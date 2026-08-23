"""
eda.py
------
Generates the full suite of exploratory data analysis (EDA) charts
for the cleaned movie dataset and saves each one as a PNG file inside
reports/graphs/.

Running this file directly will generate all 15 required graphs.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import project_paths, ensure_dir

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11

ACCENT = "#4C6EF5"
ACCENT_2 = "#F76707"


def _save(fig, graphs_dir, filename):
    """Save a matplotlib figure to the graphs directory and close it."""
    path = os.path.join(graphs_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eda] Saved {filename}")


def plot_rating_distribution(df, graphs_dir):
    """1. Movie Rating Distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["vote_average"].dropna(), bins=30, color=ACCENT, edgecolor="white")
    ax.set_title("Movie Rating Distribution")
    ax.set_xlabel("Vote Average")
    ax.set_ylabel("Number of Movies")
    _save(fig, graphs_dir, "01_rating_distribution.png")


def plot_popularity_distribution(df, graphs_dir):
    """2. Popularity Distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["popularity"].dropna(), bins=40, color=ACCENT_2, edgecolor="white")
    ax.set_title("Popularity Distribution")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Number of Movies")
    _save(fig, graphs_dir, "02_popularity_distribution.png")


def plot_vote_count_distribution(df, graphs_dir):
    """3. Vote Count Distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["vote_count"].dropna(), bins=40, color="#2F9E44", edgecolor="white")
    ax.set_title("Vote Count Distribution")
    ax.set_xlabel("Vote Count")
    ax.set_ylabel("Number of Movies")
    _save(fig, graphs_dir, "03_vote_count_distribution.png")


def plot_movies_per_year(df, graphs_dir):
    """4. Movies Released Per Year"""
    counts = df["release_year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color=ACCENT)
    ax.set_title("Movies Released Per Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Movies")
    _save(fig, graphs_dir, "04_movies_per_year.png")


def plot_popularity_vs_rating(df, graphs_dir):
    """5. Popularity vs Rating Scatter Plot"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["popularity"], df["vote_average"], alpha=0.4, color=ACCENT, s=18)
    ax.set_title("Popularity vs Rating")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Vote Average")
    _save(fig, graphs_dir, "05_popularity_vs_rating.png")


def plot_correlation_heatmap(df, graphs_dir):
    """6. Correlation Heatmap"""
    numeric_cols = ["popularity", "vote_average", "vote_count",
                     "release_year", "release_month", "Popularity_per_Vote"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=9)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, graphs_dir, "06_correlation_heatmap.png")


def plot_top_rated_movies(df, graphs_dir, n=10):
    """7. Top 10 Highest Rated Movies (with a minimum vote threshold for fairness)"""
    threshold = df["vote_count"].quantile(0.5)
    qualified = df[df["vote_count"] >= threshold]
    top = qualified.nlargest(n, "vote_average").sort_values("vote_average")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["title"], top["vote_average"], color=ACCENT)
    ax.set_title(f"Top {n} Highest Rated Movies")
    ax.set_xlabel("Vote Average")
    _save(fig, graphs_dir, "07_top_rated_movies.png")


def plot_top_popular_movies(df, graphs_dir, n=10):
    """8. Top 10 Most Popular Movies"""
    top = df.nlargest(n, "popularity").sort_values("popularity")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["title"], top["popularity"], color=ACCENT_2)
    ax.set_title(f"Top {n} Most Popular Movies")
    ax.set_xlabel("Popularity")
    _save(fig, graphs_dir, "08_top_popular_movies.png")


def plot_avg_rating_by_year(df, graphs_dir):
    """9. Average Rating by Release Year"""
    avg = df.groupby("release_year")["vote_average"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg.index, avg.values, marker="o", color=ACCENT, linewidth=2)
    ax.set_title("Average Rating by Release Year")
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Average Vote Rating")
    ax.grid(alpha=0.3)
    _save(fig, graphs_dir, "09_avg_rating_by_year.png")


def plot_vote_count_vs_rating(df, graphs_dir):
    """10. Vote Count vs Rating Scatter Plot"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["vote_count"], df["vote_average"], alpha=0.4, color="#2F9E44", s=18)
    ax.set_title("Vote Count vs Rating")
    ax.set_xlabel("Vote Count")
    ax.set_ylabel("Vote Average")
    _save(fig, graphs_dir, "10_vote_count_vs_rating.png")


def plot_rating_boxplot(df, graphs_dir):
    """11. Rating Box Plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(df["vote_average"].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=ACCENT, color="black"),
               medianprops=dict(color="black"))
    ax.set_title("Rating Box Plot")
    ax.set_ylabel("Vote Average")
    _save(fig, graphs_dir, "11_rating_boxplot.png")


def plot_popularity_boxplot(df, graphs_dir):
    """12. Popularity Box Plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(df["popularity"].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=ACCENT_2, color="black"),
               medianprops=dict(color="black"))
    ax.set_title("Popularity Box Plot")
    ax.set_ylabel("Popularity")
    _save(fig, graphs_dir, "12_popularity_boxplot.png")


def plot_popularity_rank_curve(df, graphs_dir):
    """13. Popularity Rank Curve (sorted popularity values, log scale)"""
    sorted_pop = df["popularity"].sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_pop.index + 1, sorted_pop.values, color=ACCENT, linewidth=2)
    ax.set_yscale("log")
    ax.set_title("Popularity Rank Curve")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Popularity (log scale)")
    ax.grid(alpha=0.3)
    _save(fig, graphs_dir, "13_popularity_rank_curve.png")


def plot_movies_by_month(df, graphs_dir):
    """14. Movie Count by Release Month"""
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    counts = df["release_month"].value_counts().sort_index()
    counts = counts.reindex(range(1, 13), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(month_names, counts.values, color=ACCENT_2)
    ax.set_title("Movie Count by Release Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Movies")
    _save(fig, graphs_dir, "14_movies_by_month.png")


def plot_feature_correlation_matrix(df, graphs_dir):
    """15. Feature Correlation Matrix (scatter matrix of key numeric features)"""
    cols = ["popularity", "vote_average", "vote_count", "Popularity_per_Vote"]
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(11, 10))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[cols[i]].dropna(), bins=20, color=ACCENT)
            else:
                ax.scatter(df[cols[j]], df[cols[i]], alpha=0.3, s=8, color=ACCENT_2)
            if i == n - 1:
                ax.set_xlabel(cols[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(cols[i], fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Feature Correlation Matrix", y=1.0)
    _save(fig, graphs_dir, "15_feature_correlation_matrix.png")


def generate_all_graphs(df=None):
    """
    Generate and save all 15 EDA graphs to reports/graphs/.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Cleaned dataset. If not provided, it is loaded from the
        project's standard cleaned_dataset.csv path.
    """
    paths = project_paths()
    ensure_dir(paths["graphs"])

    if df is None:
        df = pd.read_csv(paths["clean_csv"])
        df["release_date"] = pd.to_datetime(df["release_date"])

    plot_functions = [
        plot_rating_distribution,
        plot_popularity_distribution,
        plot_vote_count_distribution,
        plot_movies_per_year,
        plot_popularity_vs_rating,
        plot_correlation_heatmap,
        plot_top_rated_movies,
        plot_top_popular_movies,
        plot_avg_rating_by_year,
        plot_vote_count_vs_rating,
        plot_rating_boxplot,
        plot_popularity_boxplot,
        plot_popularity_rank_curve,
        plot_movies_by_month,
        plot_feature_correlation_matrix,
    ]

    for func in plot_functions:
        try:
            func(df, paths["graphs"])
        except Exception as exc:  # noqa: BLE001 - want to continue on single-plot failure
            print(f"[eda] Failed to generate plot '{func.__name__}': {exc}")

    print(f"[eda] All graphs saved to {paths['graphs']}")


if __name__ == "__main__":
    generate_all_graphs()
