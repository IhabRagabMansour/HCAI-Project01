"""Preprocessing pipeline for Palmer Penguins models.

A single ColumnTransformer builder shared by both model families:

- numeric features: median imputation, optionally StandardScaler
- categorical features: most-frequent imputation + one-hot encoding

Scaling is **off** for decision trees (they don't need it, and unscaled
thresholds keep the tree plot human-readable, e.g. "flipper_length_mm <= 206.5")
and **on** for logistic regression (which requires standardized inputs).

``verbose_feature_names_out=False`` keeps post-transform names clean
("island_Biscoe" instead of "cat__island_Biscoe"), which matters for the
tree visualization and the logistic-regression coefficient table.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Build an un-fit ColumnTransformer. ``scale`` toggles StandardScaler on
    the numeric branch (True for logistic regression, False for trees)."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names_out(fitted_pipeline) -> list[str]:
    """Post-preprocessing feature names from a fitted full pipeline."""
    return list(fitted_pipeline.named_steps["preprocessor"].get_feature_names_out())
