"""
features.py

Transforms the raw merged DataFrame from data_loader.py into a
feature matrix ready for model training.

Features engineered:
    - grid                        : starting grid position
    - quali_position              : qualifying classification position
    - driver_season_podium_rate   : driver's podium rate in all prior races THIS season
    - constructor_season_podium_rate : constructor's podium rate in all prior races THIS season
    - driver_season_avg_grid      : driver's avg grid position in all prior races THIS season
    - teammate_podium_rate_diff   : driver's podium rate minus their teammate's (same team, same season)

Label:
    - podium (1 = P1-P3, 0 = otherwise)
"""
__author__ = "Sandeeptha Madan, Evan Sivets"

import pandas as pd
import numpy as np
from src.data_loader import load_dataset


#  Helpers 

def _expanding_season_stat(df: pd.DataFrame,
                           group_cols: list[str],
                             value_col: str,
                             new_col: str,
                             agg: str = "mean") -> pd.DataFrame:
    """
    For each row, compute an aggregate of `value_col` over all PRIOR
    races in the same season for the given group (driver or constructor).

    Uses shift(1) so the current race is never included (no leakage).
    Rows with no prior data in the season are filled with the overall
    season mean to avoid NaNs propagating into the model.
    """
    df = df.sort_values(["year", "round"]).copy()

    if agg == "mean":
        df[new_col] = (
            df.groupby(group_cols)[value_col]
            .transform(lambda s: s.shift(1).expanding().mean())
        )
    elif agg == "sum":
        df[new_col] = (
            df.groupby(group_cols)[value_col]
            .transform(lambda s: s.shift(1).expanding().sum())
        )

    # Fill first-race-of-season NaNs with that season's overall mean
    season_means = df.groupby("year")[value_col].transform("mean")
    df[new_col] = df[new_col].fillna(season_means)

    return df


#  Feature engineering 

def add_driver_season_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of races THIS season (before this round) where driver finished podium."""
    return _expanding_season_stat(
        df,
        group_cols=["year", "driverId"],
        value_col="podium",
        new_col="driver_season_podium_rate",
        agg="mean",
    )


def add_constructor_season_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of race entries THIS season (before this round) where constructor finished podium."""
    return _expanding_season_stat(
        df,
        group_cols=["year", "constructorId"],
        value_col="podium",
        new_col="constructor_season_podium_rate",
        agg="mean",
    )


def add_driver_season_avg_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Driver's average grid position across all prior races this season."""
    return _expanding_season_stat(
        df,
        group_cols=["year", "driverId"],
        value_col="grid",
        new_col="driver_season_avg_grid",
        agg="mean",
    )


def add_teammate_podium_rate_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each driver, compute:
        driver_season_podium_rate  -  teammate's driver_season_podium_rate

    Requires add_driver_season_podium_rate() to have been called first.
    A positive value means the driver is outperforming their teammate so far
    this season. Drivers without a teammate in the data get a diff of 0.
    """
    if "driver_season_podium_rate" not in df.columns:
        raise ValueError("Run add_driver_season_podium_rate() before add_teammate_podium_rate_diff().")

    # One row per (race, constructor) with mean podium rate of *other* drivers
    # at the same constructor
    teammate_mean = (
        df.groupby(["raceId", "constructorId", "driverId"])["driver_season_podium_rate"]
        .first()
        .reset_index()
    )

    # For each (raceId, constructorId), compute the mean rate excluding each driver
    constructor_total = teammate_mean.groupby(["raceId", "constructorId"])["driver_season_podium_rate"].transform("sum")
    constructor_count = teammate_mean.groupby(["raceId", "constructorId"])["driver_season_podium_rate"].transform("count")

    teammate_mean["teammate_avg_podium_rate"] = (
        (constructor_total - teammate_mean["driver_season_podium_rate"])
        / (constructor_count - 1).replace(0, np.nan)
    ).fillna(teammate_mean["driver_season_podium_rate"])  # solo driver → diff = 0

    df = df.merge(
        teammate_mean[["raceId", "driverId", "teammate_avg_podium_rate"]],
        on=["raceId", "driverId"],
        how="left",
    )

    df["teammate_podium_rate_diff"] = (
        df["driver_season_podium_rate"] - df["teammate_avg_podium_rate"]
    ).fillna(0)

    df = df.drop(columns=["teammate_avg_podium_rate"])
    return df


#  Quali position fill 

def clean_grid_and_quali(df: pd.DataFrame) -> pd.DataFrame:
    """
    grid = 0 means the driver started from the pit lane — replace with 20
    (back of a typical grid) so it's treated as a disadvantage, not noise.
    Fill missing quali_position with grid position as a fallback.
    """
    df = df.copy()
    df["grid"] = df["grid"].replace(0, 20)
    df["quali_position"] = df["quali_position"].fillna(df["grid"])
    return df


#  Pipeline 

FEATURE_COLS = [
    "grid",
    "quali_position",
    "driver_season_podium_rate",
    "constructor_season_podium_rate",
    "driver_season_avg_grid",
    "teammate_podium_rate_diff",
]

LABEL_COL = "podium"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Returns a DataFrame with FEATURE_COLS + LABEL_COL + identifier columns
    (raceId, year, round, driverId, driver_name, constructorId, constructor_name).
    """
    print("Engineering features...")

    df = clean_grid_and_quali(df)
    df = add_driver_season_podium_rate(df)
    df = add_constructor_season_podium_rate(df)
    df = add_driver_season_avg_grid(df)
    df = add_teammate_podium_rate_diff(df)

    id_cols = ["raceId", "year", "round", "driverId", "driver_name",
               "constructorId", "constructor_name"]

    keep = id_cols + FEATURE_COLS + [LABEL_COL]
    df   = df[keep].reset_index(drop=True)

    # Drop any rows still missing a feature (e.g. drivers with no grid data)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing feature values")

    print(f"  Feature matrix: {len(df)} rows × {len(FEATURE_COLS)} features")
    print(f"  Podium share:   {df[LABEL_COL].mean():.1%}")
    return df


# Train / val / test split 

def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split to avoid data leakage:
        Train : 2014 – 2021  (~70 %)
        Val   : 2022         (~15 %)
        Test  : 2023 – 2024  (~15 %)

    Returns (train_df, val_df, test_df).
    """
    train = df[df["year"] <= 2021].copy()
    val   = df[df["year"] == 2022].copy()
    test  = df[df["year"] >= 2023].copy()

    print(f"\n  Train : {len(train):>5} rows  ({train['year'].min()}–{train['year'].max()})")
    print(f"  Val   : {len(val):>5} rows  ({val['year'].min()}–{val['year'].max()})")
    print(f"  Test  : {len(test):>5} rows  ({test['year'].min()}–{test['year'].max()})")

    return train, val, test


def get_Xy(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from a split DataFrame."""
    return split_df[FEATURE_COLS], split_df[LABEL_COL]


# Main 

if __name__ == "__main__":
    raw_df      = load_dataset()
    feature_df  = build_features(raw_df)
    train, val, test = split_by_season(feature_df)

    X_train, y_train = get_Xy(train)
    X_val,   y_val   = get_Xy(val)
    X_test,  y_test  = get_Xy(test)

    print("\nFeature matrix preview (train):")
    print(X_train.head(10).to_string(index=False))
    print("\nClass balance across splits:")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name}: {y.mean():.1%} podium")