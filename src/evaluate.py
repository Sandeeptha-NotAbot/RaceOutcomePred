"""
evaluate.py

Loads trained models from models/ and evaluates them on the
validation and test sets, producing:
    - Full metrics table  (accuracy, precision, recall, F1, ROC-AUC)
    - Confusion matrices
    - Feature importance  (Decision Tree + Logistic Regression coefficients)
    - Results saved to results/

Usage:
    python src/evaluate.py
"""
__author__ = "Sandeeptha Madan, Evan Sivets"

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

from src.data_loader import load_dataset
from src.features import build_features, split_by_season, get_Xy, FEATURE_COLS
from src.train import load_models

# Config 
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# Metrics 

def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a dict of evaluation metrics for one model on one split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y, y_prob), 4) if y_prob is not None else "N/A",
    }
    return metrics


def evaluate_all(models: dict,
                 X_val: pd.DataFrame,  y_val: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all models on both val and test sets.
    Returns a tidy DataFrame with columns:
        model, split, accuracy, precision, recall, f1, roc_auc
    """
    rows = []
    for name, model in sorted(models.items()):
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            m = compute_metrics(model, X, y)
            rows.append({"model": name, "split": split_name, **m})

    return pd.DataFrame(rows)


#  Confusion matrices 

def plot_confusion_matrices(models: dict,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> None:
    """Save a confusion matrix plot for each model to results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, model in sorted(models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(
                     confusion_matrix=cm,
                     display_labels=["No Podium", "Podium"]
                 )
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"cm_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved confusion matrix → {path}")


#  Feature importance 

def plot_feature_importance(models: dict) -> None:
    """
    Save feature importance plots for:
        - Decision Tree   (tree-based importances)
        - Logistic Regression L2  (absolute coefficient magnitudes)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    feature_names = FEATURE_COLS

    # Decision Tree
    if "decision_tree" in models:
        clf = models["decision_tree"].named_steps["clf"]
        importances = clf.feature_importances_
        _save_importance_plot(
            importances, feature_names,
            title="Decision Tree — Feature Importances",
            filename="importance_decision_tree.png",
        )

    # Logistic Regression L2 coefficients
    if "logreg_l2" in models:
        clf = models["logreg_l2"].named_steps["clf"]
        importances = np.abs(clf.coef_[0])
        _save_importance_plot(
            importances, feature_names,
            title="Logistic Regression (L2) — |Coefficients|",
            filename="importance_logreg_l2.png",
        )


def _save_importance_plot(importances: np.ndarray,
                          feature_names: list[str],
                          title: str,
                          filename: str) -> None:
    indices = np.argsort(importances)[::-1]
    sorted_names   = [feature_names[i] for i in indices]
    sorted_values  = importances[indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sorted_names[::-1], sorted_values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved feature importance → {path}")


#  Summary table 

def print_and_save_metrics(df: pd.DataFrame) -> None:
    """Pretty-print the metrics table and save it as CSV and JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Print
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    for split in ["val", "test"]:
        print(f"\n  {split.upper()} SET")
        print("  " + "-" * 60)
        subset = df[df["split"] == split].drop(columns="split")
        print(subset.to_string(index=False))
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved metrics table → {csv_path}")

    # Save JSON (useful for pasting numbers into the LaTeX report)
    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"  Saved metrics JSON  → {json_path}")


#  Main 

if __name__ == "__main__":
    # Load data
    raw_df     = load_dataset()
    feature_df = build_features(raw_df)
    _, val_df, test_df = split_by_season(feature_df)

    X_val,  y_val  = get_Xy(val_df)
    X_test, y_test = get_Xy(test_df)

    # Load trained models
    print("\nLoading models...")
    models = load_models()

    # Evaluate
    metrics_df = evaluate_all(models, X_val, y_val, X_test, y_test)
    print_and_save_metrics(metrics_df)

    # Plots
    print("\nGenerating plots...")
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models)

    print("\nDone. All outputs saved to results/")