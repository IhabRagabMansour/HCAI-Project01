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


@dataclass
class TrainResult:
    pipeline: object            # in-memory fitted sklearn Pipeline (preprocessor + estimator)
    pipeline_bytes: bytes       # joblib-serialized version of the same pipeline
    train_score: float
    test_score: float
    train_duration_ms: int


# ── Estimator factory ───────────────────────────────────────────────────────

def build_estimator(algorithm: str, random_seed: int = 42):
    """Return a fresh sklearn estimator. Seed is passed where the estimator supports it."""
    # Classification
    if algorithm == "logreg":
        return LogisticRegression(random_state=random_seed, max_iter=1000)
    if algorithm == "rf_clf":
        return RandomForestClassifier(random_state=random_seed)
    if algorithm == "svm":
        return SVC(random_state=random_seed)
    if algorithm == "knn_clf":
        return KNeighborsClassifier()
    if algorithm == "dt_clf":
        return DecisionTreeClassifier(random_state=random_seed)
    # Regression
    if algorithm == "linreg":
        return LinearRegression()
    if algorithm == "rf_reg":
        return RandomForestRegressor(random_state=random_seed)
    if algorithm == "svr":
        return SVR()
    if algorithm == "knn_reg":
        return KNeighborsRegressor()
    if algorithm == "dt_reg":
        return DecisionTreeRegressor(random_state=random_seed)

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
) -> TrainResult:
    """Build a full sklearn Pipeline (preprocessor + fresh estimator), fit it on
    raw X_train, score on train + test using the chosen metric, return the
    fitted pipeline (both in-memory and joblib-serialized).
    """
    from .pipeline import build_full_pipeline

    estimator = build_estimator(algorithm, random_seed)
    pipeline = build_full_pipeline(prepared.preprocessing, estimator)

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
