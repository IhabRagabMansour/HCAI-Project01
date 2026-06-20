"""Palmer Penguins data loading, cleaning, splitting, and feature metadata.

Project 2 (Explainability) trains models to predict penguin `species` from the
seven input features. This module centralizes:

- loading + cleaning the dataset (drop missing rows, per the project sheet),
- a deterministic stratified train/test split,
- the canonical feature groupings used everywhere downstream,
- per-numeric-feature MAD (median absolute deviation) for the counterfactual
  MAD-weighted L1 distance (Task 4).

Everything is cached by random seed so repeated requests are instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Canonical feature groups (single source of truth) ───────────────────────

TARGET = "species"

# All numeric input features. `year` is numeric but discrete (a calendar year).
NUMERIC_FEATURES = [
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

CATEGORICAL_FEATURES = ["island", "sex"]

# Only these four biometric measurements are selectable for PDP/ALE (Task 5).
# `year` is intentionally excluded (project sheet, common mistake #21).
BIOMETRIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Discrete numeric feature(s) — perturbed differently in counterfactuals.
DISCRETE_NUMERIC = ["year"]

INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fixed class order so probability vectors line up across the whole app.
SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]

MAD_EPSILON = 1e-6


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class PenguinData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_all: pd.DataFrame          # all clean rows (features only) — used by PDP/ALE
    y_all: pd.Series
    mad: dict                    # numeric feature name -> MAD (>= epsilon)
    numeric_std: dict            # numeric feature name -> std (for CF Gaussian noise)
    categories: dict             # categorical feature name -> sorted list of values
    numeric_ranges: dict         # numeric feature name -> (min, max) observed
    observed_years: list         # sorted unique observed years


# ── Loading + cleaning ──────────────────────────────────────────────────────

def load_clean_penguins() -> pd.DataFrame:
    """Load Palmer Penguins and drop rows with any missing value.

    The project sheet recommends ``dropna()`` for simplicity (§8.1). We keep
    column order stable and reset the index so positional row selection in the
    counterfactual UI is well-defined.
    """
    from palmerpenguins import load_penguins

    df = load_penguins()
    df = df.dropna().reset_index(drop=True)
    # Ensure year is an integer (it loads as float once NaNs are dropped)
    df["year"] = df["year"].astype(int)
    return df


def compute_mad(df: pd.DataFrame, numeric_features=NUMERIC_FEATURES) -> dict:
    """Per-feature MAD = median_i |x_ij - median_j|, floored at MAD_EPSILON
    to avoid division by zero in the counterfactual distance (sheet §4.5).
    """
    mad = {}
    for col in numeric_features:
        median = df[col].median()
        mad_val = float((df[col] - median).abs().median())
        mad[col] = max(mad_val, MAD_EPSILON)
    return mad


# ── Public, cached entry point ──────────────────────────────────────────────

@lru_cache(maxsize=8)
def get_penguin_data(seed: int = 42, test_size: float = 0.2) -> PenguinData:
    """Load, clean, split, and compute all metadata. Cached per (seed, test_size)."""
    df = load_clean_penguins()

    X = df[INPUT_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,                       # 3 classes — stratify (sheet §8.2)
    )

    mad = compute_mad(df)
    numeric_std = {col: float(df[col].std()) for col in NUMERIC_FEATURES}
    categories = {
        col: sorted(df[col].astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    numeric_ranges = {
        col: (float(df[col].min()), float(df[col].max()))
        for col in NUMERIC_FEATURES
    }
    observed_years = sorted(int(v) for v in df["year"].unique())

    return PenguinData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        X_all=X.reset_index(drop=True),
        y_all=y.reset_index(drop=True),
        mad=mad,
        numeric_std=numeric_std,
        categories=categories,
        numeric_ranges=numeric_ranges,
        observed_years=observed_years,
    )
