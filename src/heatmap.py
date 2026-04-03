"""
heatmap.py

Generates a feature correlation heatmap and saves it to results/.

Usage:
    python -m src.heatmap
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.features import build_features, split_by_season, get_Xy, FEATURE_COLS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_LABELS = {
    "grid":                          "Grid Position",
    "quali_position":                "Qualifying Position",
    "driver_season_podium_rate":     "Driver Podium Rate\n(Season)",
    "constructor_season_podium_rate":"Constructor Podium Rate\n(Season)",
    "driver_season_avg_grid":        "Driver Avg Grid\n(Season)",
    "teammate_podium_rate_diff":     "Teammate Podium\nRate Diff",
}

if __name__ == "__main__":
    raw_df     = load_dataset()
    feature_df = build_features(raw_df)
    train_df, _, _ = split_by_season(feature_df)
    X_train, y_train = get_Xy(train_df)

    # Include label in correlation matrix
    df_corr = X_train.copy()
    df_corr["podium"] = y_train.values
    df_corr = df_corr.rename(columns=FEATURE_LABELS)
    df_corr = df_corr.rename(columns={"podium": "Podium (Label)"})

    corr = df_corr.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Features", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "feature_correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")