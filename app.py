"""
app.py
------
Movie Rating Analysis and Prediction — Streamlit Dashboard

A professional, multi-page dashboard for exploring a movie dataset,
visualizing insights, and predicting movie ratings using trained
Machine Learning models (Linear Regression & Random Forest).

Run with:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Make the src/ package importable regardless of working directory
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import project_paths, rating_category            # noqa: E402
from preprocessing import clean_dataset                       # noqa: E402
from eda import generate_all_graphs                            # noqa: E402
from model_training import train_and_evaluate_all              # noqa: E402
from prediction import predict_rating                          # noqa: E402
from visualization import (                                    # noqa: E402
    feature_importance_chart,
    actual_vs_predicted_chart,
    residual_plot,
)

PATHS = project_paths()

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Rating Analysis & Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS — cinematic dark dashboard theme
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-deep: #0E1117;
    --bg-panel: #161A23;
    --bg-panel-light: #1D222D;
    --accent-gold: #E8B923;
    --accent-teal: #2FB6A6;
    --text-primary: #F2F3F5;
    --text-muted: #9AA1AD;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Archivo', sans-serif !important;
    letter-spacing: 0.2px;
}

.stApp {
    background-color: var(--bg-deep);
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid #262B36;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--bg-panel-light), var(--bg-panel));
    border: 1px solid #262B36;
    padding: 18px 16px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}
div[data-testid="stMetricValue"] {
    color: var(--accent-gold) !important;
}

/* Headline banner */
.hero-banner {
    background: linear-gradient(120deg, #1B1F2A 0%, #262B3A 60%, #1B1F2A 100%);
    border: 1px solid #2C3140;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 22px;
}
.hero-title {
    font-family: 'Archivo', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.02rem;
}
.accent-gold { color: var(--accent-gold); }
.accent-teal { color: var(--accent-teal); }

/* Section chip */
.chip {
    display: inline-block;
    background: rgba(232, 185, 35, 0.12);
    color: var(--accent-gold);
    border: 1px solid rgba(232, 185, 35, 0.35);
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Prediction result card */
.result-card {
    border-radius: 16px;
    padding: 26px;
    text-align: center;
    border: 1px solid #2C3140;
    background: linear-gradient(160deg, var(--bg-panel-light), var(--bg-panel));
}
.result-rating {
    font-family: 'Archivo', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--accent-gold);
}
.result-tag {
    display: inline-block;
    margin-top: 6px;
    padding: 4px 16px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# DATA / MODEL LOADING (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cleaned_data():
    """Load the cleaned dataset, running preprocessing if needed."""
    if not os.path.exists(PATHS["clean_csv"]):
        clean_dataset()
    df = pd.read_csv(PATHS["clean_csv"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


@st.cache_data(show_spinner=False)
def get_raw_data():
    """Load the raw dataset for the Home page summary."""
    return pd.read_csv(PATHS["raw_csv"])


@st.cache_resource(show_spinner=False)
def get_training_results():
    """Train (or retrain) both models and cache results for the session."""
    return train_and_evaluate_all(save_models=True)


def ensure_graphs_exist(df):
    """Generate EDA graphs on first run if they don't already exist."""
    graphs_dir = PATHS["graphs"]
    if not os.path.exists(graphs_dir) or len(os.listdir(graphs_dir)) < 15:
        with st.spinner("Generating EDA visualizations for the first time..."):
            generate_all_graphs(df)


def category_color(category):
    """Return a (background, text) color pair for a rating category chip."""
    mapping = {
        "Excellent": ("rgba(47, 182, 106, 0.18)", "#2FB66A"),
        "Good": ("rgba(47, 182, 166, 0.18)", "#2FB6A6"),
        "Average": ("rgba(232, 185, 35, 0.18)", "#E8B923"),
        "Poor": ("rgba(230, 80, 80, 0.18)", "#E65050"),
    }
    return mapping.get(category, ("rgba(154,161,173,0.18)", "#9AA1AD"))


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 6px 0 16px 0;'>"
        "<span style='font-family:Archivo, sans-serif; font-size:1.4rem; font-weight:800; color:#F2F3F5;'>🎬 CineMetrics</span>"
        "<br><span style='color:#9AA1AD; font-size:0.82rem;'>Movie Rating Analytics</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    page = option_menu(
        menu_title=None,
        options=[
            "Home", "Dataset Explorer", "EDA Dashboard",
            "Top Movies", "Rating Prediction", "Model Performance", "About",
        ],
        icons=["house", "table", "bar-chart-line", "star", "cpu", "graph-up", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#E8B923", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "3px 0",
                "color": "#C3C8D1", "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#262B3A", "color": "#F2F3F5"},
        },
    )

    st.markdown("---")
    
    st.caption("Movie Rating Analysis & Prediction")


# --------------------------------------------------------------------------
# LOAD DATA (with error handling / loading spinner)
# --------------------------------------------------------------------------
try:
    with st.spinner("Loading dataset..."):
        raw_df = get_raw_data()
        clean_df = get_cleaned_data()
except FileNotFoundError:
    st.error(
        "⚠️ Dataset not found. Please place 'Movie-Dataset-Latest.csv' inside the "
        "`data/` folder before running the app."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"⚠️ An unexpected error occurred while loading data: {exc}")
    st.stop()


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="chip">B.Tech AI &amp; ML Final Project</div>
            <div class="hero-title">🎬 Movie Rating Analysis <span class="accent-gold">&amp; Prediction</span></div>
            <div class="hero-subtitle">
                Exploring rating patterns, popularity trends, and building Machine Learning
                models that predict how audiences will rate a movie.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(clean_df):,}")
    col2.metric("Average Rating", f"{clean_df['vote_average'].mean():.2f} / 10")
    col3.metric("Avg. Popularity", f"{clean_df['popularity'].mean():.1f}")
    col4.metric("Year Range", f"{int(clean_df['release_year'].min())}–{int(clean_df['release_year'].max())}")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📖 Project Overview")
        st.markdown("""
This project analyzes a dataset of movies to uncover insights about **ratings**,
**popularity**, **vote counts**, and **release trends**, then applies supervised
Machine Learning to **predict a movie's rating** from its metadata.

**Pipeline:**
1. **Preprocessing** — cleaning, missing-value handling, feature engineering
2. **EDA** — 15 visualizations covering distributions, trends, and correlations
3. **Modelling** — Linear Regression & Random Forest Regressor
4. **Prediction** — interactive rating predictor with model selection
5. **Dashboard** — this Streamlit app, for exploration & presentation
        """)

    with right:
        st.subheader("📂 Dataset Summary")
        st.dataframe(
            pd.DataFrame({
                "Metric": ["Rows", "Columns", "Missing (raw)", "Duplicates removed",
                           "Earliest release", "Latest release"],
                "Value": [
                    len(raw_df), raw_df.shape[1],
                    int(raw_df.isna().sum().sum()),
                    int(raw_df.duplicated().sum()),
                    str(clean_df["release_date"].min().date()),
                    str(clean_df["release_date"].max().date()),
                ],
            }),
            hide_index=True, use_container_width=True,
        )

    st.write("")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(clean_df.head(15), use_container_width=True)


# ==========================================================================
# PAGE: DATASET EXPLORER
# ==========================================================================
elif page == "Dataset Explorer":
    st.markdown("## 📂 Dataset Explorer")
    st.caption("Search, filter, and export the cleaned movie dataset.")

    with st.expander("🔎 Filters", expanded=True):
        c1, c2, c3 = st.columns([1.4, 1, 1])

        with c1:
            search_term = st.text_input("Search by movie title", "")

        with c2:
            rating_range = st.slider(
                "Filter by rating", 0.0, 10.0, (0.0, 10.0), step=0.1
            )

        with c3:
            year_min, year_max = int(clean_df["release_year"].min()), int(clean_df["release_year"].max())
            year_range = st.slider("Filter by release year", year_min, year_max, (year_min, year_max))

    filtered = clean_df.copy()
    if search_term.strip():
        filtered = filtered[filtered["title"].str.contains(search_term, case=False, na=False)]
    filtered = filtered[
        (filtered["vote_average"] >= rating_range[0]) & (filtered["vote_average"] <= rating_range[1])
        & (filtered["release_year"] >= year_range[0]) & (filtered["release_year"] <= year_range[1])
    ]

    st.success(f"✅ {len(filtered):,} movies match your filters.")
    st.dataframe(
        filtered[["title", "release_date", "popularity", "vote_average", "vote_count", "overview"]],
        use_container_width=True, height=440,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered CSV", data=csv_bytes,
        file_name="filtered_movies.csv", mime="text/csv",
    )


# ==========================================================================
# PAGE: EDA DASHBOARD
# ==========================================================================
elif page == "EDA Dashboard":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.caption("15 visualizations covering distributions, trends, and correlations.")

    ensure_graphs_exist(clean_df)
    graphs_dir = PATHS["graphs"]

    sections = [
        ("Distributions", [
            ("01_rating_distribution.png", "Movie Rating Distribution"),
            ("02_popularity_distribution.png", "Popularity Distribution"),
            ("03_vote_count_distribution.png", "Vote Count Distribution"),
        ]),
        ("Trends Over Time", [
            ("04_movies_per_year.png", "Movies Released Per Year"),
            ("09_avg_rating_by_year.png", "Average Rating by Release Year"),
            ("14_movies_by_month.png", "Movie Count by Release Month"),
        ]),
        ("Relationships", [
            ("05_popularity_vs_rating.png", "Popularity vs Rating"),
            ("10_vote_count_vs_rating.png", "Vote Count vs Rating"),
            ("06_correlation_heatmap.png", "Correlation Heatmap"),
            ("15_feature_correlation_matrix.png", "Feature Correlation Matrix"),
        ]),
        ("Rankings", [
            ("07_top_rated_movies.png", "Top 10 Highest Rated Movies"),
            ("08_top_popular_movies.png", "Top 10 Most Popular Movies"),
        ]),
        ("Spread & Outliers", [
            ("11_rating_boxplot.png", "Rating Box Plot"),
            ("12_popularity_boxplot.png", "Popularity Box Plot"),
            ("13_popularity_rank_curve.png", "Popularity Rank Curve"),
        ]),
    ]

    for section_name, images in sections:
        with st.expander(f"📁 {section_name}", expanded=(section_name == "Distributions")):
            cols = st.columns(2)
            for idx, (filename, caption) in enumerate(images):
                img_path = os.path.join(graphs_dir, filename)
                with cols[idx % 2]:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=caption, use_container_width=True)
                    else:
                        st.warning(f"Graph not found: {filename}")


# ==========================================================================
# PAGE: TOP MOVIES
# ==========================================================================
elif page == "Top Movies":
    st.markdown("## ⭐ Top Movies")

    tab1, tab2 = st.tabs(["🏆 Top Rated", "🔥 Most Popular"])

    with tab1:
        n = st.slider("Number of movies to show", 5, 25, 10, key="top_rated_n")
        vote_threshold = clean_df["vote_count"].quantile(0.5)
        qualified = clean_df[clean_df["vote_count"] >= vote_threshold]
        top_rated = qualified.nlargest(n, "vote_average")[
            ["title", "release_date", "vote_average", "vote_count", "popularity"]
        ].reset_index(drop=True)
        top_rated.index += 1
        st.dataframe(top_rated, use_container_width=True)

    with tab2:
        n2 = st.slider("Number of movies to show", 5, 25, 10, key="top_popular_n")
        top_popular = clean_df.nlargest(n2, "popularity")[
            ["title", "release_date", "popularity", "vote_average", "vote_count"]
        ].reset_index(drop=True)
        top_popular.index += 1
        st.dataframe(top_popular, use_container_width=True)


# ==========================================================================
# PAGE: RATING PREDICTION
# ==========================================================================
elif page == "Rating Prediction":
    st.markdown("## 🤖 Movie Rating Prediction")
    st.caption("Enter movie metadata below to predict its audience rating.")

    models_ready = os.path.exists(os.path.join(PATHS["models"], "linear_regression.pkl")) and \
        os.path.exists(os.path.join(PATHS["models"], "random_forest.pkl"))

    if not models_ready:
        st.warning("Models not found — training them now (this happens only once)...")
        get_training_results()
        st.rerun()

    left, right = st.columns([1.1, 1])

    with left:
        with st.form("prediction_form"):
            st.markdown("#### 🎛️ Movie Details")

            model_choice = st.selectbox("Model", ["Random Forest", "Linear Regression"])

            popularity_input = st.number_input(
                "Popularity", min_value=0.0, max_value=1000.0, value=25.0, step=0.5
            )
            vote_count_input = st.number_input(
                "Vote Count", min_value=0, max_value=100000, value=1000, step=10
            )
            release_year_input = st.number_input(
                "Release Year", min_value=1900, max_value=2030, value=2023, step=1
            )
            release_month_input = st.selectbox(
                "Release Month", list(range(1, 13)),
                format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
            )

            submitted = st.form_submit_button("🔮 Predict Rating", use_container_width=True)

    with right:
        if submitted:
            try:
                with st.spinner("Predicting rating..."):
                    result = predict_rating(
                        model_name=model_choice,
                        popularity=popularity_input,
                        vote_count=vote_count_input,
                        release_year=release_year_input,
                        release_month=release_month_input,
                    )

                bg, fg = category_color(result["category"])
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="color:#9AA1AD; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">Predicted Rating</div>
                        <div class="result-rating">{result['predicted_rating']} / 10</div>
                        <div class="result-tag" style="background:{bg}; color:{fg};">{result['category']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                c1, c2 = st.columns(2)
                c1.metric("Model Used", result["model_used"])
                c2.metric("Popularity per Vote", result["popularity_per_vote"])
                st.success("✅ Prediction generated successfully.")

            except FileNotFoundError as exc:
                st.error(f"⚠️ {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"⚠️ Prediction failed: {exc}")
        else:
            st.info("👈 Fill in the movie details and click **Predict Rating** to see the result here.")


# ==========================================================================
# PAGE: MODEL PERFORMANCE
# ==========================================================================
elif page == "Model Performance":
    st.markdown("## 📈 Model Performance")
    st.caption("Comparing Linear Regression and Random Forest on held-out test data.")

    with st.spinner("Training and evaluating models..."):
        training_output = get_training_results()

    comparison_df = training_output["comparison_df"]
    st.markdown("#### 🧮 Comparison Table")
    st.dataframe(comparison_df.style.format("{:.3f}"), use_container_width=True)

    best_model = comparison_df["R2 Score"].idxmax()
    st.success(f"✅ **{best_model}** achieved the highest R² Score on the test set.")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["🌲 Feature Importance", "🎯 Actual vs Predicted", "📉 Residual Plot"])

    with tab1:
        fig = feature_importance_chart(training_output["feature_importance"])
        st.pyplot(fig, use_container_width=True)

    with tab2:
        model_for_diag = st.selectbox(
            "Select model", ["Random Forest", "Linear Regression"], key="diag_model_1"
        )
        fig = actual_vs_predicted_chart(
            training_output["y_test"], training_output["predictions"][model_for_diag], model_for_diag
        )
        st.pyplot(fig, use_container_width=True)

    with tab3:
        model_for_diag2 = st.selectbox(
            "Select model", ["Random Forest", "Linear Regression"], key="diag_model_2"
        )
        fig = residual_plot(
            training_output["y_test"], training_output["predictions"][model_for_diag2], model_for_diag2
        )
        st.pyplot(fig, use_container_width=True)


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
elif page == "About":
    st.markdown("## ℹ️ About This Project")

    st.markdown("""
### 🎯 Objective
Analyze a movie dataset to discover insights about ratings, popularity, vote counts,
and release trends — then build Machine Learning models to predict movie ratings,
presented through an interactive Streamlit dashboard.

### 📂 Dataset Information
The dataset contains metadata for movies including `title`, `release_date`,
`popularity`, `vote_average`, `vote_count`, `overview`, and `id`. It was cleaned to
remove duplicates and missing values, and enriched with `release_year`,
`release_month`, and `Popularity_per_Vote` features.

### 🛠️ Technologies Used
- **Python** — core programming language
- **Pandas / NumPy** — data manipulation & numerical computation
- **Matplotlib** — static visualizations
- **Scikit-learn** — machine learning models & evaluation
- **Joblib** — model persistence
- **Streamlit** — interactive dashboard

### 🤖 Machine Learning Models
- **Linear Regression** — a simple, interpretable baseline model
- **Random Forest Regressor** — an ensemble model capturing non-linear relationships

Both models are evaluated using MAE, MSE, RMSE, and R² Score.

### 🚀 Future Scope
- Incorporate genre, cast, and crew data for richer features
- Add NLP-based sentiment analysis on movie overviews
- Experiment with gradient boosting (XGBoost / LightGBM) and neural networks
- Deploy the dashboard publicly via Streamlit Community Cloud
- Add user authentication and personalized recommendations

---
**Project Type:** B.Tech Computer Science (AI & ML) Final Year Project
    """)
