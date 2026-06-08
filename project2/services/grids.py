"""Train grids of models spanning the accuracy/complexity frontier.

Each grid trains the same model family at several regularization strengths.
The interface's lambda slider later selects the best entry post-hoc
(see selection.py). Grids are cached by seed; the penguins dataset is tiny
(~333 rows) so a full grid trains in well under a second.

IMPORTANT distinction (project sheet §4.3): the per-model fitting parameter
(`max_leaf_nodes` for trees, `C` for logistic regression) is NOT the interface
lambda. lambda is applied after training to choose among these entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .complexity import logreg_n_nonzero, tree_n_leaves
from .data import get_penguin_data
from .pipeline import build_preprocessor


# sklearn 1.8 deprecated penalty="l1" in favor of l1_ratio. Build pure-L1
# multinomial logistic regression in a version-compatible way.
_SKL_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def _build_l1_logreg(c: float, seed: int) -> LogisticRegression:
    kw = dict(solver="saga", C=c, max_iter=5000, random_state=seed)
    if _SKL_VERSION >= (1, 8):
        kw["l1_ratio"] = 1.0          # pure L1 under the unified elasticnet API
    else:
        kw["penalty"] = "l1"          # classic API (sklearn < 1.8)
    return LogisticRegression(**kw)


# Fitting-parameter grids (sheet §4.2, §4.3)
TREE_MAX_LEAF_NODES_GRID = [2, 3, 4, 5, 6, 8, 10, 15, 20, None]
LOGREG_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


@dataclass
class ModelEntry:
    model_class: str        # "tree" | "logreg"
    param_name: str         # "max_leaf_nodes" | "C"
    param_value: object     # the fitting-parameter value (None for unconstrained tree)
    pipeline: object        # fitted sklearn Pipeline (preprocessor + estimator)
    test_accuracy: float
    complexity: int

    @property
    def param_display(self) -> str:
        if self.param_value is None:
            return "None (unconstrained)"
        return str(self.param_value)


@lru_cache(maxsize=8)
def train_tree_grid(seed: int = 42) -> tuple:
    """Train decision trees across the max_leaf_nodes grid. Trees use an
    UNSCALED preprocessor so plotted thresholds stay in real units."""
    data = get_penguin_data(seed)
    entries = []
    for mln in TREE_MAX_LEAF_NODES_GRID:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(scale=False)),
            ("estimator", DecisionTreeClassifier(max_leaf_nodes=mln, random_state=seed)),
        ])
        pipe.fit(data.X_train, data.y_train)
        acc = float(accuracy_score(data.y_test, pipe.predict(data.X_test)))
        entries.append(ModelEntry(
            model_class="tree",
            param_name="max_leaf_nodes",
            param_value=mln,
            pipeline=pipe,
            test_accuracy=acc,
            complexity=tree_n_leaves(pipe),
        ))
    return tuple(entries)


@lru_cache(maxsize=8)
def train_logreg_grid(seed: int = 42) -> tuple:
    """Train multinomial L1 logistic regressions across the C grid. Uses a
    SCALED preprocessor (logistic regression needs standardized inputs)."""
    data = get_penguin_data(seed)
    entries = []
    for c in LOGREG_C_GRID:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(scale=True)),
            ("estimator", _build_l1_logreg(c, seed)),
        ])
        pipe.fit(data.X_train, data.y_train)
        acc = float(accuracy_score(data.y_test, pipe.predict(data.X_test)))
        entries.append(ModelEntry(
            model_class="logreg",
            param_name="C",
            param_value=c,
            pipeline=pipe,
            test_accuracy=acc,
            complexity=logreg_n_nonzero(pipe),
        ))
    return tuple(entries)


def get_grid(model_class: str, seed: int = 42) -> tuple:
    if model_class == "tree":
        return train_tree_grid(seed)
    if model_class == "logreg":
        return train_logreg_grid(seed)
    raise ValueError(f"Unknown model class: {model_class!r}")


def unconstrained_tree_entry(seed: int = 42) -> ModelEntry:
    """The fully-grown tree (max_leaf_nodes=None) — the Task 1 baseline
    'fit a decision tree' before any lambda-based selection."""
    for entry in train_tree_grid(seed):
        if entry.param_value is None:
            return entry
    # Fallback: the largest-leaf-count entry
    return max(train_tree_grid(seed), key=lambda e: e.complexity)
