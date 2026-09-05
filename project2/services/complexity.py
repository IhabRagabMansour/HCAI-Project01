"""Model complexity measures Omega(f) for the interpretability tradeoff.

- Decision tree:  Omega = number of leaves
- Logistic reg.:  Omega = number of nonzero coefficients

The logistic-regression measure counts coefficients whose absolute value
exceeds a small threshold, because L1 regularization drives coefficients close
to — but not exactly — zero numerically. This is the most human-readable
sparsity measure and pairs naturally with the L1 penalty.
"""

from __future__ import annotations

import numpy as np

NONZERO_THRESHOLD = 1e-6


def tree_n_leaves(pipeline) -> int:
    """Number of leaves of the fitted DecisionTreeClassifier inside the pipeline."""
    return int(pipeline.named_steps["estimator"].get_n_leaves())


def logreg_n_nonzero(pipeline, threshold: float = NONZERO_THRESHOLD) -> int:
    """Number of nonzero coefficients across all classes of the fitted
    LogisticRegression inside the pipeline."""
    coef = np.asarray(pipeline.named_steps["estimator"].coef_)
    return int(np.sum(np.abs(coef) > threshold))
