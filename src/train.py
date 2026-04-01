"""
train.py

Trains three ML models on the engineered feature matrix and saves
them to the models/ directory.

Models:
    1. Logistic Regression  (baseline, L1 + L2 comparison)
    2. Support Vector Machine (linear + RBF kernel comparison)
    3. Decision Tree         (nonlinear, interpretable)

Usage:
    python src/train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset
from src.features import build_features, split_by_season, get_Xy

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RANDOM_SEED = 42


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict[str, Pipeline]:
    """
    Returns a dict of named sklearn Pipelines.

    Each pipeline: StandardScaler → Classifier
    Scaling is important for Logistic Regression and SVM which are
    sensitive to feature magnitude. Decision Tree is scale-invariant
    but included in a pipeline for consistency.
    """
    models = {
        # Logistic Regression — L2 regularization (default, good baseline)
        "logreg_l2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                l1_ratio=0,
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )),
        ]),

        # Logistic Regression — L1 regularization (sparse, feature selection)
        "logreg_l1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                l1_ratio=1,
                solver="saga",
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )),
        ]),

        # SVM — linear kernel
        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="linear",
                C=1.0,
                probability=True,          # needed for predict_proba in evaluate.py
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )),
        ]),

        # SVM — RBF kernel (nonlinear)
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            )),
        ]),

        # Decision Tree
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                max_depth=5,               # limit depth to reduce overfitting
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
    }
    return models


# ── Training ──────────────────────────────────────────────────────────────────

def train_all(X_train: pd.DataFrame,
              y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit all models on the training set and return fitted pipelines."""
    models = get_models()
    fitted = {}

    print("\nTraining models...")
    print("-" * 40)
    for name, pipeline in models.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        print("done")

    return fitted


# ── Saving / loading ──────────────────────────────────────────────────────────

def save_models(fitted: dict[str, Pipeline]) -> None:
    """Pickle each fitted pipeline to models/<name>.pkl"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, pipeline in fitted.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  Saved → {path}")


def load_models() -> dict[str, Pipeline]:
    """Load all .pkl files from models/ and return as a dict."""
    fitted = {}
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            path = os.path.join(MODELS_DIR, filename)
            with open(path, "rb") as f:
                fitted[name] = pickle.load(f)
    print(f"  Loaded {len(fitted)} models from {MODELS_DIR}")
    return fitted


# ── Quick val check ───────────────────────────────────────────────────────────

def quick_val_accuracy(fitted: dict[str, Pipeline],
                       X_val: pd.DataFrame,
                       y_val: pd.Series) -> None:
    """Print a quick accuracy snapshot on the validation set."""
    print("\nValidation accuracy (quick check):")
    print("-" * 40)
    for name, pipeline in sorted(fitted.items()):
        acc = pipeline.score(X_val, y_val)
        print(f"  {name:<20} {acc:.4f}")
    print("\nRun evaluate.py for full metrics (precision, recall, F1, ROC-AUC).")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load and prepare data
    raw_df     = load_dataset()
    feature_df = build_features(raw_df)
    train_df, val_df, test_df = split_by_season(feature_df)

    X_train, y_train = get_Xy(train_df)
    X_val,   y_val   = get_Xy(val_df)

    print(f"\n  Training on {len(X_train)} examples, "
          f"validating on {len(X_val)} examples")
    print(f"  Features: {list(X_train.columns)}")

    # Train
    fitted = train_all(X_train, y_train)

    # Save
    print("\nSaving models...")
    save_models(fitted)

    # Quick sanity check
    quick_val_accuracy(fitted, X_val, y_val)