
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#4C6EF5"
ACCENT_2 = "#F76707"


def feature_importance_chart(feature_importance):
   
    fi_sorted = feature_importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(fi_sorted.index, fi_sorted.values, color=ACCENT)
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def actual_vs_predicted_chart(y_test, predictions, model_name):
  
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, predictions, alpha=0.4, color=ACCENT, s=20)

    lims = [min(min(y_test), min(predictions)), max(max(y_test), max(predictions))]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, label="Perfect Prediction")

    ax.set_title(f"Actual vs Predicted Rating ({model_name})")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.legend()
    fig.tight_layout()
    return fig


def residual_plot(y_test, predictions, model_name):
   
    residuals = np.array(y_test) - np.array(predictions)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predictions, residuals, alpha=0.4, color=ACCENT_2, s=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Plot ({model_name})")
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("Residual (Actual - Predicted)")
    fig.tight_layout()
    return fig
