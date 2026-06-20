"""Manual PDP and ALE for the four biometric features (Task 5).

Implemented from scratch using only the model's ``predict_proba`` and numpy —
no library PDP/ALE function is used (project sheet §5-6, §17-18). Each method
returns three curves, one per species probability.

PDP  (Lecture 3, slide 29):  PD_f(x_A) = E_{x_B ~ marginal}[f(x_A, x_B)]
    estimated by replacing the feature with each grid value for ALL rows and
    averaging predict_proba. (Marginal expectation — may probe unrealistic
    feature combinations under correlation.)

ALE  (Lecture 3, slide 31):  ALE_f(x_A) = ∫ E_{x_B|z_A}[∂f/∂z_A] dz_A − C
    estimated by finite differences inside equal-population bins (conditional
    distribution — only rows whose feature lies in the bin), accumulated, then
    centered to zero data-weighted mean. Finite differences work for both the
    differentiable logistic regression and the piecewise-constant decision tree.

Three corrections over the project sheet's ALE pseudocode:
  1. half-open bins [lo, hi) (last bin closed) so edge points are not double-counted,
  2. data-weighted centering Σ(n_k·ale_k)/N (zero mean over the data, not over bins),
  3. empty bins contribute a zero local effect and carry the accumulator.
"""

from __future__ import annotations

import numpy as np


def _class_names(pipeline) -> list[str]:
    return [str(c) for c in pipeline.classes_]


# ── Partial Dependence Plot ─────────────────────────────────────────────────

def compute_pdp(pipeline, X, feature: str, n_grid: int = 25) -> dict:
    """Average predicted probability per class as the feature sweeps a grid.

    Returns {"feature", "grid": [n_grid], "classes": [k], "curves": [n_grid][k]}.
    Each grid row sums to ~1 (averaged probability vectors).
    """
    values = X[feature].to_numpy(dtype=float)
    lo, hi = float(values.min()), float(values.max())
    grid = np.linspace(lo, hi, n_grid)

    curves = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        proba = pipeline.predict_proba(Xv)          # (n_rows, n_classes)
        curves.append(np.asarray(proba).mean(axis=0))
    curves = np.vstack(curves)                       # (n_grid, n_classes)

    return {
        "feature": feature,
        "grid": [float(g) for g in grid],
        "classes": _class_names(pipeline),
        "curves": curves.tolist(),
    }


# ── Accumulated Local Effects ───────────────────────────────────────────────

def compute_ale(pipeline, X, feature: str, n_bins: int = 20) -> dict:
    """Accumulated local effects per class via finite differences.

    Returns {"feature", "centers": [B], "classes": [k], "curves": [B][k]}.
    Curves are centered so the data-weighted mean is ~0 per class.
    """
    classes = _class_names(pipeline)
    n_classes = len(classes)
    values = X[feature].to_numpy(dtype=float)

    # Equal-population (quantile) bin edges; dedupe in case of repeated values.
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([values.min(), values.max()], dtype=float)
    n_actual = len(edges) - 1

    deltas = np.zeros((n_actual, n_classes))
    counts = np.zeros(n_actual)
    centers = []

    for k in range(n_actual):
        lo, hi = edges[k], edges[k + 1]
        centers.append(float((lo + hi) / 2.0))

        if k == n_actual - 1:
            mask = (values >= lo) & (values <= hi)      # last bin closed
        else:
            mask = (values >= lo) & (values < hi)       # half-open (fix #1)

        n_k = int(mask.sum())
        counts[k] = n_k
        if n_k == 0:
            continue                                    # zero effect, carry accumulator (fix #3)

        X_sub = X.loc[mask]
        X_lo = X_sub.copy(); X_lo[feature] = lo
        X_hi = X_sub.copy(); X_hi[feature] = hi
        diff = np.asarray(pipeline.predict_proba(X_hi)) - np.asarray(pipeline.predict_proba(X_lo))
        deltas[k] = diff.mean(axis=0)                   # mean local secant in the bin

    ale = np.cumsum(deltas, axis=0)                     # accumulate (the integral)

    # Data-weighted centering: zero mean over the data, not over bins (fix #2).
    total = counts.sum()
    if total > 0:
        weighted_mean = (counts[:, None] * ale).sum(axis=0) / total
    else:
        weighted_mean = ale.mean(axis=0)
    ale_centered = ale - weighted_mean

    return {
        "feature": feature,
        "centers": centers,
        "classes": classes,
        "curves": ale_centered.tolist(),
        "bin_counts": [int(c) for c in counts],
    }
