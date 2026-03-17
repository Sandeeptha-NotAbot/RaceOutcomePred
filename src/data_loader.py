"""
data_loader.py

Loads, filters, and merges the raw Kaggle F1 CSVs into a single
clean DataFrame ready for feature engineering.

Expected files in DATA_RAW_DIR:
    races.csv, results.csv, drivers.csv,
    constructors.csv, qualifying.csv, circuits.csv
"""

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ERA_START    = 2014   # first year of the hybrid-turbo era
ERA_END      = 2024   # last complete season before 2026 reg changes


# ── Loaders ───────────────────────────────────────────────────────────────────

def _path(filename: str) -> str:
    return os.path.join(DATA_RAW_DIR, filename)


def load_raw() -> dict[str, pd.DataFrame]:
    """Load all six CSVs and return them as a dict of DataFrames."""
    files = {
        "races":        "races.csv",
        "results":      "results.csv",
        "drivers":      "drivers.csv",
        "constructors": "constructors.csv",
        "qualifying":   "qualifying.csv",
        "circuits":     "circuits.csv",
    }

    dfs = {}
    for key, filename in files.items():
        filepath = _path(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing file: {filepath}\n"
                f"Download the Kaggle dataset and place all CSVs in {DATA_RAW_DIR}"
            )
        dfs[key] = pd.read_csv(filepath, na_values=["\\N", "NA", ""])
        print(f"  Loaded {filename:<25} {len(dfs[key]):>6} rows")

    return dfs


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_era(races: pd.DataFrame) -> pd.DataFrame:
    """Keep only races within the hybrid-turbo era (ERA_START – ERA_END)."""
    races["year"] = pd.to_numeric(races["year"], errors="coerce")
    mask = races["year"].between(ERA_START, ERA_END)
    filtered = races[mask].copy()
    print(f"  Era filter ({ERA_START}–{ERA_END}): "
          f"{len(races)} → {len(filtered)} races")
    return filtered


# ── Merging ───────────────────────────────────────────────────────────────────

def build_dataset(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all tables into one flat DataFrame, one row per driver per race.

    Columns in output:
        raceId, year, round, circuitId, circuit_name, country,
        driverId, driver_name, constructorId, constructor_name,
        grid, position, positionOrder,
        q1, q2, q3,          ← qualifying lap times (raw strings)
        podium               ← LABEL: 1 if finished P1–P3, else 0
    """
    races        = filter_era(dfs["races"])
    results      = dfs["results"]
    drivers      = dfs["drivers"]
    constructors = dfs["constructors"]
    qualifying   = dfs["qualifying"]
    circuits     = dfs["circuits"]

    # ── 1. Keep only results for era races ────────────────────────────────────
    era_race_ids = set(races["raceId"])
    results = results[results["raceId"].isin(era_race_ids)].copy()

    # ── 2. Numeric coercions ──────────────────────────────────────────────────
    results["grid"]          = pd.to_numeric(results["grid"],          errors="coerce")
    results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
    results["position"]      = pd.to_numeric(results["position"],      errors="coerce")

    # ── 3. Build label: podium = finished P1, P2, or P3 ──────────────────────
    results["podium"] = (results["positionOrder"].between(1, 3)).astype(int)

    # ── 4. Merge races (year, round, circuitId) ───────────────────────────────
    race_cols = ["raceId", "year", "round", "circuitId", "name"]
    df = results.merge(races[race_cols].rename(columns={"name": "race_name"}),
                       on="raceId", how="left")

    # ── 5. Merge circuits ─────────────────────────────────────────────────────
    circuit_cols = ["circuitId", "name", "country"]
    df = df.merge(circuits[circuit_cols].rename(columns={"name": "circuit_name"}),
                  on="circuitId", how="left")

    # ── 6. Merge drivers ──────────────────────────────────────────────────────
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    driver_cols = ["driverId", "driver_name", "nationality"]
    df = df.merge(drivers[driver_cols].rename(columns={"nationality": "driver_nationality"}),
                  on="driverId", how="left")

    # ── 7. Merge constructors ─────────────────────────────────────────────────
    constructor_cols = ["constructorId", "name"]
    df = df.merge(constructors[constructor_cols].rename(columns={"name": "constructor_name"}),
                  on="constructorId", how="left")

    # ── 8. Merge qualifying (best lap time per driver per race) ───────────────
    qual_cols = ["raceId", "driverId", "position", "q1", "q2", "q3"]
    qualifying_era = qualifying[qualifying["raceId"].isin(era_race_ids)][qual_cols].copy()
    qualifying_era = qualifying_era.rename(columns={"position": "quali_position"})
    df = df.merge(qualifying_era, on=["raceId", "driverId"], how="left")

    # ── 9. Select & reorder final columns ─────────────────────────────────────
    final_cols = [
        "raceId", "year", "round", "circuitId", "circuit_name", "country",
        "driverId", "driver_name", "driver_nationality",
        "constructorId", "constructor_name",
        "grid", "quali_position", "position", "positionOrder",
        "q1", "q2", "q3",
        "podium",
    ]
    df = df[final_cols].reset_index(drop=True)

    print(f"\n  Final dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"  Podium share:  {df['podium'].mean():.1%}  "
          f"({df['podium'].sum()} podium finishes)")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Single entry point for the rest of the pipeline.

    Usage:
        from src.data_loader import load_dataset
        df = load_dataset()
    """
    print(f"\nLoading F1 dataset ({ERA_START}–{ERA_END})...")
    print("-" * 45)
    dfs = load_raw()
    print("-" * 45)
    df  = build_dataset(dfs)
    print("-" * 45)
    return df


if __name__ == "__main__":
    df = load_dataset()
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))
    print("\nColumn dtypes:")
    print(df.dtypes)