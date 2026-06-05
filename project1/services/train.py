from __future__ import annotations

import io
import time
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .preprocess import PreparedData


CLASSIFICATION_ALGOS = ("logreg", "rf_clf", "svm", "knn_clf", "dt_clf")
REGRESSION_ALGOS     = ("linreg", "rf_reg", "svr", "knn_reg", "dt_reg")
CLASSIFICATION_METRICS = ("accuracy", "f1", "precision", "recall")
REGRESSION_METRICS     = ("r2", "rmse", "mae")


# ── Hyperparameter spec registry ────────────────────────────────────────────
#
# Each algorithm lists tunable hyperparameters with type, default, and the
# bounds / choices the form should enforce. The view consumes this both to
# render the form and to validate POST values.
#
# Special translations applied at estimator-construction time:
#   max_depth     == 0         → None  (sklearn convention for "unlimited")
#   max_features  == "none"    → None
#   penalty       == "none"    → None
#   p (KNN)       cast to int
# These keep the form values clean (no nullable fields needed in JSON).

HYPERPARAM_SPECS: dict[str, list[dict]] = {
    "logreg": [
        {"key": "class_weight", "label": "Class weight", "type": "choice",
         "default": "none",
         "choices": [["none", "None"], ["balanced", "Balanced (auto by frequency)"]],
         "hint": "Use 'balanced' for imbalanced classes — adjusts weights inversely proportional to frequencies."},
        {"key": "C", "label": "Regularization (C)", "type": "float",
         "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1,
         "hint": "Inverse of regularization strength. Smaller = stronger regularization."},
        {"key": "penalty", "label": "Penalty", "type": "choice",
         "default": "l2",
         "choices": [["l2", "L2 (Ridge)"], ["l1", "L1 (Lasso)"],
                     ["elasticnet", "ElasticNet"], ["none", "None"]],
         "hint": "L1 and ElasticNet require the saga solver."},
        {"key": "solver", "label": "Solver", "type": "choice",
         "default": "lbfgs",
         "choices": [["lbfgs", "lbfgs"], ["liblinear", "liblinear"],
                     ["saga", "saga"], ["newton-cg", "newton-cg"]],
         "hint": "Optimization algorithm. lbfgs is a strong default."},
        {"key": "fit_intercept", "label": "Fit intercept", "type": "bool",
         "default": True,
         "hint": "Include a bias term in the model."},
        {"key": "max_iter", "label": "Max iterations", "type": "int",
         "default": 1000, "min": 50, "max": 10000, "step": 50,
         "hint": "Convergence cap for the solver."},
    ],
    "rf_clf": [
        {"key": "class_weight", "label": "Class weight", "type": "choice",
         "default": "none",
         "choices": [["none", "None"], ["balanced", "Balanced (auto by frequency)"]],
         "hint": "Use 'balanced' for imbalanced classes."},
        {"key": "n_estimators", "label": "Number of trees", "type": "int",
         "default": 100, "min": 10, "max": 500, "step": 10,
         "hint": "More trees = more accurate but slower."},
        {"key": "max_depth", "label": "Max tree depth (0 = unlimited)", "type": "int",
         "default": 0, "min": 0, "max": 50, "step": 1,
         "hint": "Limit depth to control overfitting."},
        {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1,
         "hint": "Minimum samples required to split an internal node."},
        {"key": "min_samples_leaf", "label": "Min samples per leaf", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1,
         "hint": "Minimum samples required at each leaf."},
        {"key": "max_features", "label": "Max features per split", "type": "choice",
         "default": "sqrt",
         "choices": [["sqrt", "sqrt"], ["log2", "log2"], ["none", "All features"]],
         "hint": "Features to consider at each split."},
        {"key": "criterion", "label": "Split criterion", "type": "choice",
         "default": "gini",
         "choices": [["gini", "Gini"], ["entropy", "Entropy"], ["log_loss", "Log loss"]],
         "hint": "Function to measure split quality."},
        {"key": "bootstrap", "label": "Bootstrap samples", "type": "bool",
         "default": True,
         "hint": "Use bootstrap samples when building trees."},
    ],
    "svm": [
        {"key": "class_weight", "label": "Class weight", "type": "choice",
         "default": "none",
         "choices": [["none", "None"], ["balanced", "Balanced (auto by frequency)"]],
         "hint": "Use 'balanced' for imbalanced classes."},
        {"key": "C", "label": "Regularization (C)", "type": "float",
         "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1,
         "hint": "Larger C = less regularization."},
        {"key": "kernel", "label": "Kernel", "type": "choice",
         "default": "rbf",
         "choices": [["rbf", "RBF"], ["linear", "Linear"],
                     ["poly", "Polynomial"], ["sigmoid", "Sigmoid"]],
         "hint": "RBF is a strong default; Linear is fast on high-dim data."},
        {"key": "gamma", "label": "Gamma", "type": "choice",
         "default": "scale",
         "choices": [["scale", "Scale"], ["auto", "Auto"]],
         "hint": "Kernel coefficient for rbf, poly, sigmoid."},
        {"key": "degree", "label": "Polynomial degree", "type": "int",
         "default": 3, "min": 1, "max": 10, "step": 1,
         "hint": "Only used by polynomial kernel."},
        {"key": "coef0", "label": "coef0", "type": "float",
         "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
         "hint": "Independent term for poly and sigmoid kernels."},
    ],
    "knn_clf": [
        {"key": "n_neighbors", "label": "Number of neighbors (k)", "type": "int",
         "default": 5, "min": 1, "max": 50, "step": 1,
         "hint": "Smaller k = more flexible; larger k = smoother."},
        {"key": "weights", "label": "Weight by distance?", "type": "choice",
         "default": "uniform",
         "choices": [["uniform", "Uniform"], ["distance", "By distance"]],
         "hint": "Distance weighting gives closer neighbors more influence."},
        {"key": "algorithm", "label": "Search algorithm", "type": "choice",
         "default": "auto",
         "choices": [["auto", "Auto"], ["ball_tree", "Ball tree"],
                     ["kd_tree", "KD tree"], ["brute", "Brute force"]],
         "hint": "Algorithm used to compute nearest neighbors."},
        {"key": "p", "label": "Distance metric", "type": "choice",
         "default": "2",
         "choices": [["1", "Manhattan (p=1)"], ["2", "Euclidean (p=2)"]],
         "hint": "Power parameter for the Minkowski metric."},
    ],
    "dt_clf": [
        {"key": "class_weight", "label": "Class weight", "type": "choice",
         "default": "none",
         "choices": [["none", "None"], ["balanced", "Balanced (auto by frequency)"]],
         "hint": "Use 'balanced' for imbalanced classes."},
        {"key": "max_depth", "label": "Max tree depth (0 = unlimited)", "type": "int",
         "default": 0, "min": 0, "max": 50, "step": 1,
         "hint": "Limit depth to control overfitting."},
        {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1,
         "hint": "Minimum samples to split an internal node."},
        {"key": "min_samples_leaf", "label": "Min samples per leaf", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1,
         "hint": "Minimum samples at a leaf."},
        {"key": "criterion", "label": "Split criterion", "type": "choice",
         "default": "gini",
         "choices": [["gini", "Gini"], ["entropy", "Entropy"], ["log_loss", "Log loss"]],
         "hint": "Function to measure split quality."},
        {"key": "splitter", "label": "Splitter", "type": "choice",
         "default": "best",
         "choices": [["best", "Best"], ["random", "Random"]],
         "hint": "Strategy to choose the split at each node."},
    ],
    "linreg": [
        {"key": "fit_intercept", "label": "Fit intercept", "type": "bool",
         "default": True,
         "hint": "Include a bias term in the model."},
        {"key": "positive", "label": "Positive coefficients only", "type": "bool",
         "default": False,
         "hint": "Constrain coefficients to be non-negative."},
    ],
    "rf_reg": [
        {"key": "n_estimators", "label": "Number of trees", "type": "int",
         "default": 100, "min": 10, "max": 500, "step": 10,
         "hint": "More trees = more accurate but slower."},
        {"key": "max_depth", "label": "Max tree depth (0 = unlimited)", "type": "int",
         "default": 0, "min": 0, "max": 50, "step": 1,
         "hint": "Limit depth to control overfitting."},
        {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1,
         "hint": "Minimum samples to split an internal node."},
        {"key": "min_samples_leaf", "label": "Min samples per leaf", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1,
         "hint": "Minimum samples at a leaf."},
        {"key": "max_features", "label": "Max features per split", "type": "choice",
         "default": "1.0",
         "choices": [["1.0", "All features"], ["sqrt", "sqrt"], ["log2", "log2"]],
         "hint": "Features to consider at each split."},
        {"key": "criterion", "label": "Split criterion", "type": "choice",
         "default": "squared_error",
         "choices": [["squared_error", "Squared error"], ["absolute_error", "Absolute error"],
                     ["friedman_mse", "Friedman MSE"], ["poisson", "Poisson"]],
         "hint": "Function to measure split quality."},
        {"key": "bootstrap", "label": "Bootstrap samples", "type": "bool",
         "default": True,
         "hint": "Use bootstrap samples when building trees."},
    ],
    "svr": [
        {"key": "C", "label": "Regularization (C)", "type": "float",
         "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1,
         "hint": "Larger C = less regularization."},
        {"key": "kernel", "label": "Kernel", "type": "choice",
         "default": "rbf",
         "choices": [["rbf", "RBF"], ["linear", "Linear"],
                     ["poly", "Polynomial"], ["sigmoid", "Sigmoid"]],
         "hint": "RBF is a strong default."},
        {"key": "gamma", "label": "Gamma", "type": "choice",
         "default": "scale",
         "choices": [["scale", "Scale"], ["auto", "Auto"]],
         "hint": "Kernel coefficient for rbf, poly, sigmoid."},
        {"key": "degree", "label": "Polynomial degree", "type": "int",
         "default": 3, "min": 1, "max": 10, "step": 1,
         "hint": "Only used by polynomial kernel."},
        {"key": "coef0", "label": "coef0", "type": "float",
         "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
         "hint": "Independent term for poly and sigmoid kernels."},
        {"key": "epsilon", "label": "Epsilon", "type": "float",
         "default": 0.1, "min": 0.0, "max": 100.0, "step": 0.01,
         "hint": "Width of the epsilon-tube; no penalty for points within it."},
    ],
    "knn_reg": [
        {"key": "n_neighbors", "label": "Number of neighbors (k)", "type": "int",
         "default": 5, "min": 1, "max": 50, "step": 1,
         "hint": "Smaller k = more flexible; larger k = smoother."},
        {"key": "weights", "label": "Weight by distance?", "type": "choice",
         "default": "uniform",
         "choices": [["uniform", "Uniform"], ["distance", "By distance"]],
         "hint": "Distance weighting gives closer neighbors more influence."},
        {"key": "algorithm", "label": "Search algorithm", "type": "choice",
         "default": "auto",
         "choices": [["auto", "Auto"], ["ball_tree", "Ball tree"],
                     ["kd_tree", "KD tree"], ["brute", "Brute force"]],
         "hint": "Algorithm used to compute nearest neighbors."},
        {"key": "p", "label": "Distance metric", "type": "choice",
         "default": "2",
         "choices": [["1", "Manhattan (p=1)"], ["2", "Euclidean (p=2)"]],
         "hint": "Power parameter for the Minkowski metric."},
    ],
    "dt_reg": [
        {"key": "max_depth", "label": "Max tree depth (0 = unlimited)", "type": "int",
         "default": 0, "min": 0, "max": 50, "step": 1,
         "hint": "Limit depth to control overfitting."},
        {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
         "default": 2, "min": 2, "max": 50, "step": 1,
         "hint": "Minimum samples to split an internal node."},
        {"key": "min_samples_leaf", "label": "Min samples per leaf", "type": "int",
         "default": 1, "min": 1, "max": 50, "step": 1,
         "hint": "Minimum samples at a leaf."},
        {"key": "criterion", "label": "Split criterion", "type": "choice",
         "default": "squared_error",
         "choices": [["squared_error", "Squared error"], ["absolute_error", "Absolute error"],
                     ["friedman_mse", "Friedman MSE"], ["poisson", "Poisson"]],
         "hint": "Function to measure split quality."},
        {"key": "splitter", "label": "Splitter", "type": "choice",
         "default": "best",
         "choices": [["best", "Best"], ["random", "Random"]],
         "hint": "Strategy to choose the split at each node."},
    ],
}


def _resolve_hyperparameters(algorithm: str, raw: dict) -> dict:
    """Filter raw hyperparameters to keys known to this algorithm and apply
    the special-case translations sklearn expects (None instead of 0/"none",
    int casting for KNN's p, etc.)."""
    spec_keys = {s["key"] for s in HYPERPARAM_SPECS.get(algorithm, [])}
    out: dict = {}
    for key, val in (raw or {}).items():
        if key not in spec_keys:
            continue
        if key == "max_depth" and val == 0:
            out[key] = None
            continue
        if key == "max_features" and val == "none":
            out[key] = None
            continue
        if key == "penalty" and val == "none":
            out[key] = None
            continue
        if key == "class_weight" and val == "none":
            out[key] = None
            continue
        if key == "p":
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                continue
            continue
        out[key] = val
    return out


@dataclass
class TrainResult:
    pipeline: object            # in-memory fitted sklearn Pipeline (preprocessor + estimator)
    pipeline_bytes: bytes       # joblib-serialized version of the same pipeline
    train_score: float
    test_score: float
    train_duration_ms: int


# ── Estimator factory ───────────────────────────────────────────────────────

def build_estimator(algorithm: str, random_seed: int = 42, hyperparameters: dict | None = None):
    """Return a fresh sklearn estimator. Seed is passed where supported.
    Hyperparameters override sklearn defaults (after spec-based resolution)."""
    kw = _resolve_hyperparameters(algorithm, hyperparameters or {})

    # Classification
    if algorithm == "logreg":
        return LogisticRegression(random_state=random_seed, **{"max_iter": 1000, **kw})
    if algorithm == "rf_clf":
        return RandomForestClassifier(random_state=random_seed, **kw)
    if algorithm == "svm":
        return SVC(random_state=random_seed, **kw)
    if algorithm == "knn_clf":
        return KNeighborsClassifier(**kw)
    if algorithm == "dt_clf":
        return DecisionTreeClassifier(random_state=random_seed, **kw)
    # Regression
    if algorithm == "linreg":
        return LinearRegression(**kw)
    if algorithm == "rf_reg":
        return RandomForestRegressor(random_state=random_seed, **kw)
    if algorithm == "svr":
        return SVR(**kw)
    if algorithm == "knn_reg":
        return KNeighborsRegressor(**kw)
    if algorithm == "dt_reg":
        return DecisionTreeRegressor(random_state=random_seed, **kw)

    raise ValueError(f"Unknown algorithm: {algorithm!r}")


# ── Metric dispatch ─────────────────────────────────────────────────────────

def compute_score(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric!r}")


# ── Train + score ───────────────────────────────────────────────────────────

def train_and_score(
    prepared: PreparedData,
    algorithm: str,
    metric: str,
    random_seed: int = 42,
    hyperparameters: dict | None = None,
) -> TrainResult:
    """Build a full sklearn Pipeline (preprocessor + fresh estimator), fit it on
    raw X_train, score on train + test using the chosen metric, return the
    fitted pipeline (both in-memory and joblib-serialized).
    """
    from .pipeline import build_full_pipeline, build_sampler

    estimator = build_estimator(algorithm, random_seed, hyperparameters=hyperparameters)
    oversampling = getattr(prepared, "oversampling", "none") or "none"
    sampler = build_sampler(oversampling, random_seed)
    pipeline = build_full_pipeline(prepared.preprocessing, estimator, sampler=sampler)

    t0 = time.perf_counter()
    pipeline.fit(prepared.X_train, prepared.y_train)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    y_train_pred = pipeline.predict(prepared.X_train)
    y_test_pred  = pipeline.predict(prepared.X_test)

    train_score = compute_score(prepared.y_train, y_train_pred, metric)
    test_score  = compute_score(prepared.y_test,  y_test_pred,  metric)

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)

    return TrainResult(
        pipeline=pipeline,
        pipeline_bytes=buf.getvalue(),
        train_score=train_score,
        test_score=test_score,
        train_duration_ms=elapsed_ms,
    )
