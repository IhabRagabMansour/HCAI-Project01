"""Counterfactual explanations via local random sampling (Task 4).

Given a selected model, an example x, and a target species, we:

1. sample N candidate points locally around x (type-aware perturbation),
2. predict each with the *currently selected* model,
3. keep only candidates predicted as the target class,
4. rank the survivors by MAD-weighted L1 distance to x,
5. return the best k.

If none are found, we widen the search (more candidates, larger numeric noise,
higher categorical-switch probability) and retry up to ``max_attempts``.

Feature-type handling:
- biometric decimals  -> Gaussian noise  z = x + N(0, alpha * std), clipped to range
- year (discrete)     -> rounded Gaussian, clipped to observed year range
- island, sex (categ) -> keep with prob (1 - switch), else sample another category

The MAD-weighted L1 distance is computed over the numeric features only
(the standard Wachter et al. definition); categorical changes are used as a
tie-break so a counterfactual that flips fewer categories ranks higher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
    INPUT_FEATURES,
)


@dataclass
class CounterfactualConfig:
    n_candidates: int = 500
    k: int = 5
    alpha: float = 0.5            # numeric noise scale = alpha * feature std
    cat_switch_prob: float = 0.2  # probability of switching a categorical value
    max_attempts: int = 6
    seed: int = 0


@dataclass
class Counterfactual:
    row: dict                    # raw feature values of the candidate
    predicted_class: str
    target_proba: float
    distance: float              # MAD-weighted L1 over numeric features
    changed_features: list       # names of features that differ from x
    n_changed: int
    n_cat_changed: int = field(default=0)


# ── Distance ────────────────────────────────────────────────────────────────

def mad_weighted_l1(x: dict, z: dict, mad: dict, numeric_features=NUMERIC_FEATURES) -> float:
    """sum_j |x_j - z_j| / MAD_j over numeric features."""
    return float(sum(abs(float(x[j]) - float(z[j])) / mad[j] for j in numeric_features))


# ── Sampling ────────────────────────────────────────────────────────────────

def sample_candidate(x: dict, data, alpha: float, cat_switch: float, rng) -> dict:
    """Perturb x once, respecting each feature's type."""
    z = dict(x)

    # Biometric decimals: Gaussian noise scaled by std, clipped to observed range.
    for feat in BIOMETRIC_FEATURES:
        std = data.numeric_std[feat]
        lo, hi = data.numeric_ranges[feat]
        z[feat] = float(np.clip(float(x[feat]) + rng.normal(0.0, alpha * std), lo, hi))

    # Year: discrete — rounded small Gaussian, clipped to observed year range.
    lo_y, hi_y = data.numeric_ranges["year"]
    z["year"] = int(np.clip(round(float(x["year"]) + rng.normal(0.0, alpha)), lo_y, hi_y))

    # Categoricals: keep with high probability, occasionally switch.
    for feat in CATEGORICAL_FEATURES:
        cats = data.categories[feat]
        if len(cats) > 1 and rng.random() < cat_switch:
            others = [c for c in cats if c != x[feat]]
            if others:
                z[feat] = others[int(rng.integers(len(others)))]

    return z


def changed_features(x: dict, z: dict, tol: float = 1e-9) -> list:
    """Features whose value differs from x (categoricals compared as strings)."""
    changed = []
    for feat in INPUT_FEATURES:
        if feat in CATEGORICAL_FEATURES:
            if str(x[feat]) != str(z[feat]):
                changed.append(feat)
        else:
            if abs(float(x[feat]) - float(z[feat])) > tol:
                changed.append(feat)
    return changed


def _n_categorical_changed(x: dict, z: dict) -> int:
    return sum(1 for f in CATEGORICAL_FEATURES if str(x[f]) != str(z[f]))


# ── Generation ──────────────────────────────────────────────────────────────

def generate_counterfactuals(pipeline, x: dict, target_class: str, data,
                             config: CounterfactualConfig | None = None) -> list:
    """Return up to k counterfactuals of the target class, ranked by distance.
    Widens the search and retries if none are found."""
    config = config or CounterfactualConfig()
    rng = np.random.default_rng(config.seed)

    classes = list(pipeline.classes_)
    if target_class not in classes:
        return []
    class_index = classes.index(target_class)

    alpha = config.alpha
    cat_switch = config.cat_switch_prob
    n = config.n_candidates
    found: list = []

    for _attempt in range(config.max_attempts):
        candidates = [sample_candidate(x, data, alpha, cat_switch, rng) for _ in range(n)]
        df = pd.DataFrame(candidates, columns=INPUT_FEATURES)
        preds = pipeline.predict(df)
        probas = pipeline.predict_proba(df)

        for i, cand in enumerate(candidates):
            if preds[i] == target_class:
                changed = changed_features(x, cand)
                found.append(Counterfactual(
                    row=cand,
                    predicted_class=str(preds[i]),
                    target_proba=float(probas[i, class_index]),
                    distance=mad_weighted_l1(x, cand, data.mad),
                    changed_features=changed,
                    n_changed=len(changed),
                    n_cat_changed=_n_categorical_changed(x, cand),
                ))

        if found:
            break

        # Widen the search for the next attempt.
        alpha *= 1.5
        cat_switch = min(1.0, cat_switch * 1.5)
        n = int(n * 1.5)

    # Rank: primary = MAD distance; tie-break = fewer categorical changes.
    found.sort(key=lambda cf: (cf.distance, cf.n_cat_changed))
    return found[: config.k]
