# Project Report: Movie Rating Analysis and Prediction using Machine Learning

**Degree Program:** B.Tech Computer Science (Artificial Intelligence & Machine Learning)
**Project Type:** Final Year Project
**Domain:** Data Analytics & Machine Learning

---

## 1. Abstract

This project analyzes a dataset of movies to uncover patterns in audience
ratings, popularity, vote counts, and release trends, and applies supervised
Machine Learning regression techniques to predict a movie's rating from its
metadata. The complete pipeline — data preprocessing, exploratory data
analysis (EDA), model training/evaluation, and rating prediction — is
delivered through a professional, multi-page **Streamlit** dashboard.

## 2. Objective

- Clean and prepare a real-world-style movie dataset for analysis.
- Perform exploratory data analysis to understand distributions, trends,
  and relationships among key movie attributes.
- Train and compare two regression models — **Linear Regression** and
  **Random Forest Regressor** — to predict `vote_average` (movie rating).
- Package the entire workflow into an interactive dashboard suitable for
  demonstration, portfolio showcasing, and academic evaluation.

## 3. Dataset Description

The dataset contains the following columns: `id`, `title`, `release_date`,
`popularity`, `vote_average`, `vote_count`, and `overview`.

The data was sourced from **"The Movies Dataset"** (a public TMDB movie
metadata export distributed on Kaggle), containing **45,460 movies** after
extracting the relevant columns from the original `movies_metadata.csv` and
dropping a small number of rows (6) with corrupted `id`/`title` values — a
known parsing artifact in that source file caused by a small number of
malformed rows shifting column alignment.

## 4. Methodology

### 4.1 Data Preprocessing (`src/preprocessing.py`)
- Removed exact duplicate rows and duplicate movie IDs.
- Handled missing values: median imputation for numeric fields
  (`popularity`, `vote_average`), zero-fill for `vote_count`, placeholder
  text for missing `overview`, and row-drop for unrecoverable missing
  `release_date` values.
- Converted `release_date` to a proper datetime type and derived
  `release_year` and `release_month`.
- Engineered a new feature: `Popularity_per_Vote = popularity / (vote_count + 1)`.
- Saved the cleaned dataset to `data/cleaned_dataset.csv`.

### 4.2 Exploratory Data Analysis (`src/eda.py`)
Fifteen visualizations were generated and saved to `reports/graphs/`,
covering:
1. Movie Rating Distribution
2. Popularity Distribution
3. Vote Count Distribution
4. Movies Released Per Year
5. Popularity vs Rating Scatter Plot
6. Correlation Heatmap
7. Top 10 Highest Rated Movies
8. Top 10 Most Popular Movies
9. Average Rating by Release Year
10. Vote Count vs Rating Scatter Plot
11. Rating Box Plot
12. Popularity Box Plot
13. Popularity Rank Curve
14. Movie Count by Release Month
15. Feature Correlation Matrix

### 4.3 Machine Learning (`src/model_training.py`)
Two regression algorithms were trained on an 80/20 train-test split using
features `popularity`, `vote_count`, `release_year`, `release_month`, and
`Popularity_per_Vote`, with `vote_average` as the target:

1. **Linear Regression** — a simple, interpretable baseline.
2. **Random Forest Regressor** (200 trees, max depth 12) — an ensemble
   method capturing non-linear feature interactions.

Models were evaluated using **MAE**, **MSE**, **RMSE**, and **R² Score**,
then persisted with **Joblib** to `models/linear_regression.pkl` and
`models/random_forest.pkl`.

#### Results (on the real "Movies Dataset" — 45,346 movies after cleaning)

| Model              | MAE   | MSE   | RMSE  | R² Score |
|---------------------|-------|-------|-------|----------|
| Linear Regression   | 1.330 | 3.726 | 1.930 | 0.011    |
| Random Forest        | 0.857 | 1.401 | 1.184 | 0.628    |

**Feature Importance (Random Forest):**

| Feature              | Importance |
|----------------------|------------|
| vote_count           | 0.890      |
| Popularity_per_Vote  | 0.035      |
| release_year         | 0.035      |
| popularity           | 0.028      |
| release_month        | 0.012      |

> **Interpretation:** Random Forest substantially outperforms Linear
> Regression (R² of 0.628 vs 0.011), showing that the rating relationship is
> strongly non-linear. `vote_count` dominates feature importance — this
> reflects a real statistical effect in review data: movies with very few
> votes have highly volatile, often extreme ratings (a single enthusiastic
> or critical reviewer swings the average), while movies with large vote
> counts converge toward a stable, moderate rating. Linear Regression cannot
> capture this non-linear "regression to the mean" pattern, which is why
> Random Forest performs so much better here.

### 4.4 Rating Prediction (`src/prediction.py`)
Given user-supplied `popularity`, `vote_count`, `release_year`, and
`release_month`, the module automatically computes `Popularity_per_Vote`,
loads the selected trained model, and returns the predicted rating along
with a qualitative category: **Poor** (<4), **Average** (4–6), **Good**
(6–8), or **Excellent** (8+).

### 4.5 Streamlit Dashboard (`app.py`)
A seven-page dashboard was built with a custom dark, cinema-inspired theme:
**Home**, **Dataset Explorer**, **EDA Dashboard**, **Top Movies**,
**Rating Prediction**, **Model Performance**, and **About** — using metric
cards, expanders, tabs, columns, a searchable/filterable data explorer with
CSV export, and interactive model diagnostics (feature importance,
actual-vs-predicted, residual plots).

## 5. Tools & Technologies

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, Streamlit,
streamlit-option-menu.

## 6. Conclusion

This project demonstrates an end-to-end, production-style data science
workflow: from raw data ingestion and cleaning, through exploratory
analysis and model building, to a polished, interactive dashboard for
non-technical stakeholders. The modular code structure (separate
preprocessing, EDA, training, and prediction modules) makes it easy to
extend — for example, by swapping in a real-world dataset, adding richer
features (genre, cast, budget), or trying more advanced models.

## 7. Future Scope

- Incorporate genre, cast, crew, and budget data for richer, more
  predictive features.
- Apply NLP techniques (e.g. sentiment analysis, embeddings) to the
  `overview` text field.
- Experiment with gradient boosting methods (XGBoost, LightGBM) and neural
  network regressors.
- Deploy the dashboard publicly via Streamlit Community Cloud or a cloud
  VM for portfolio access.
- Add authentication and personalized movie recommendations.

## 8. References

- Scikit-learn documentation — https://scikit-learn.org
- Streamlit documentation — https://docs.streamlit.io
- Pandas documentation — https://pandas.pydata.org
