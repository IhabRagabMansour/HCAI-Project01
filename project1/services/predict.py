"""Single-row prediction helpers.

The Stage 8 Pipeline refactor stored a complete (preprocessor + estimator)
sklearn Pipeline on each TrainedModel, which is what makes this possible:
``pipeline.predict(single_row_dataframe)`` handles all preprocessing identically
to training, with no manual encoding/scaling required at predict time.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def build_input_form_spec(df: pd.DataFrame, columns_meta: list[dict], excluded: set[str]) -> list[dict]:
    """For each feature column (target + excluded skipped), return a form-spec
    dict with the name, humanized dtype, input type, default value, and (for
    categorical columns) the list of distinct choices observed in the dataset.

    - Numeric → ``input_type='number'`` with default = column median
    - Categorical → ``input_type='select'`` with default = column mode
    """
    specs: list[dict] = []
    for col_meta in columns_meta:
        name = col_meta["name"]
        if name in excluded or name not in df.columns:
            continue
        series = df[name].dropna()
        if len(series) == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            specs.append({
                "name": name,
                "dtype": col_meta["dtype"],
                "input_type": "number",
                "default": float(series.median()),
            })
        else:
            choices = sorted(series.astype(str).unique().tolist())
            mode = series.astype(str).mode()
            default = str(mode.iloc[0]) if len(mode) else (choices[0] if choices else "")
            specs.append({
                "name": name,
                "dtype": col_meta["dtype"],
                "input_type": "select",
                "default": default,
                "choices": choices,
            })
    return specs


def pick_random_row_values(df: pd.DataFrame, specs: list[dict]) -> dict:
    """Pick a random row from df, return a {column_name: value} dict matching
    the form spec types. NaN cells fall back to the spec's default."""
    if len(df) == 0:
        return {s["name"]: s["default"] for s in specs}
    idx = random.randrange(len(df))
    sample = df.iloc[idx]
    values: dict = {}
    for spec in specs:
        v = sample[spec["name"]]
        if pd.isna(v):
            values[spec["name"]] = spec["default"]
        elif spec["input_type"] == "number":
            values[spec["name"]] = float(v)
        else:
            values[spec["name"]] = str(v)
    return values


def predict_single(pipeline, row: dict, feature_order: list[str]) -> dict:
    """Run pipeline.predict on a single-row DataFrame and (when supported)
    pipeline.predict_proba. Returns a dict with the raw prediction and the
    per-class probability list (or None for regression / non-probabilistic).
    """
    X = pd.DataFrame([{k: row[k] for k in feature_order}])
    pred = pipeline.predict(X)

    probabilities: list[float] | None = None
    estimator = (
        pipeline.named_steps.get("estimator")
        if hasattr(pipeline, "named_steps") else None
    )
    if estimator is not None and hasattr(estimator, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            probabilities = [float(p) for p in np.asarray(proba)[0]]
        except Exception:
            probabilities = None

    return {
        "prediction": pred[0],
        "probabilities": probabilities,
    }
