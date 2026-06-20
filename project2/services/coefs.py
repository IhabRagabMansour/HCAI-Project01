"""Logistic-regression coefficient extraction (Task 3).

Builds a per-class coefficient table from a fitted multinomial logistic
regression pipeline, flagging nonzero entries so the L1-induced sparsity is
visible. Coefficients are on STANDARDIZED features (the logreg pipeline scales
numeric inputs), so their magnitudes are directly comparable across features.
"""

from __future__ import annotations

import numpy as np

from .complexity import NONZERO_THRESHOLD
from .pipeline import feature_names_out


def coefficient_table(pipeline, threshold: float = NONZERO_THRESHOLD) -> dict:
    """Return a template-ready coefficient table for a fitted logreg pipeline.

    Structure:
        {
          "classes": [class names...],
          "rows": [{"feature", "cells": [{"value", "nonzero"} per class],
                    "any_nonzero"}],
          "intercepts": [per class],
          "n_nonzero", "n_features", "n_total_coefs"
        }
    """
    estimator = pipeline.named_steps["estimator"]
    names = feature_names_out(pipeline)
    classes = [str(c) for c in estimator.classes_]

    coef = np.asarray(estimator.coef_)                       # (n_classes, n_features)
    intercept = np.asarray(estimator.intercept_).ravel()     # (n_classes,)

    rows = []
    for j, fname in enumerate(names):
        cells = []
        any_nz = False
        for ci in range(len(classes)):
            v = float(coef[ci, j])
            nz = abs(v) > threshold
            any_nz = any_nz or nz
            cells.append({"value": v, "nonzero": nz})
        rows.append({"feature": fname, "cells": cells, "any_nonzero": any_nz})

    return {
        "classes": classes,
        "rows": rows,
        "intercepts": [float(v) for v in intercept],
        "n_nonzero": int(np.sum(np.abs(coef) > threshold)),
        "n_features": len(names),
        "n_total_coefs": int(coef.size),
    }
