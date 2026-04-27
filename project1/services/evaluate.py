from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score,
)

from .preprocess import PreparedData


MAX_PREDICTIONS_SAMPLE = 500


# ── Helpers ─────────────────────────────────────────────────────────────────

def _classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
    }


def _decode_labels(values, label_map):
    """Map integer-encoded class indices back to original (string) class names.
    If label_map is None, just stringify."""
    if label_map is None:
        return [str(v) for v in values]
    inverse = {v: k for k, v in label_map.items()}
    return [inverse.get(int(v), str(v)) for v in values]


def _subsample_indices(n: int, max_size: int = MAX_PREDICTIONS_SAMPLE):
    if n <= max_size:
        return np.arange(n)
    rng = np.random.default_rng(42)
    return rng.choice(n, size=max_size, replace=False)


# ── Classification evaluation ──────────────────────────────────────────────

def evaluate_classification(estimator, prepared: PreparedData, label_map: dict | None) -> dict:
    y_train_pred = estimator.predict(prepared.X_train)
    y_test_pred  = estimator.predict(prepared.X_test)

    train_metrics = _classification_metrics(prepared.y_train, y_train_pred)
    test_metrics  = _classification_metrics(prepared.y_test,  y_test_pred)

    # Class set from union of true + predicted (test) so we don't miss anything
    classes = sorted(set(np.asarray(prepared.y_test).tolist()) |
                     set(np.asarray(y_test_pred).tolist()))
    labels_display = _decode_labels(classes, label_map)

    cm = confusion_matrix(prepared.y_test, y_test_pred, labels=classes).tolist()

    # Per-class precision/recall/F1
    p_arr = precision_score(prepared.y_test, y_test_pred,
                            labels=classes, average=None, zero_division=0)
    r_arr = recall_score(prepared.y_test, y_test_pred,
                         labels=classes, average=None, zero_division=0)
    f_arr = f1_score(prepared.y_test, y_test_pred,
                     labels=classes, average=None, zero_division=0)

    y_test_arr = np.asarray(prepared.y_test)
    per_class = [
        {
            "label":     labels_display[i],
            "precision": float(p_arr[i]),
            "recall":    float(r_arr[i]),
            "f1":        float(f_arr[i]),
            "support":   int(np.sum(y_test_arr == cls)),
        }
        for i, cls in enumerate(classes)
    ]

    # Sampled predictions (test set), decoded
    idx = _subsample_indices(len(prepared.y_test))
    y_true_decoded = _decode_labels(np.asarray(prepared.y_test)[idx], label_map)
    y_pred_decoded = _decode_labels(np.asarray(y_test_pred)[idx], label_map)
    predictions_sample = [
        {"y_true": yt, "y_pred": yp}
        for yt, yp in zip(y_true_decoded, y_pred_decoded)
    ]

    return {
        "problem_type": "classification",
        "labels": labels_display,
        "train": train_metrics,
        "test":  test_metrics,
        "confusion_matrix": cm,
        "per_class": per_class,
        "predictions_sample": predictions_sample,
    }


# ── Regression evaluation ──────────────────────────────────────────────────

def evaluate_regression(estimator, prepared: PreparedData) -> dict:
    y_train_pred = estimator.predict(prepared.X_train)
    y_test_pred  = estimator.predict(prepared.X_test)

    train_metrics = _regression_metrics(prepared.y_train, y_train_pred)
    test_metrics  = _regression_metrics(prepared.y_test,  y_test_pred)

    idx = _subsample_indices(len(prepared.y_test))
    y_true_arr = np.asarray(prepared.y_test, dtype=float)[idx]
    y_pred_arr = np.asarray(y_test_pred,    dtype=float)[idx]

    predictions_sample = [
        {"y_true": float(yt), "y_pred": float(yp)}
        for yt, yp in zip(y_true_arr, y_pred_arr)
    ]
    residuals_sample = [
        {"y_pred": float(yp), "residual": float(yt - yp)}
        for yt, yp in zip(y_true_arr, y_pred_arr)
    ]

    return {
        "problem_type": "regression",
        "train": train_metrics,
        "test":  test_metrics,
        "predictions_sample": predictions_sample,
        "residuals_sample":   residuals_sample,
    }


# ── Feature importance ─────────────────────────────────────────────────────

def compute_feature_importance(estimator_or_pipeline, feature_names: list[str]) -> list[dict] | None:
    """Return [{name, value}, ...] sorted desc, or None when not supported.

    Accepts either a bare estimator or a sklearn Pipeline (in which case the
    final 'estimator' step is unwrapped automatically).
    """
    # Unwrap if it's a Pipeline with the standard "estimator" step name.
    if hasattr(estimator_or_pipeline, "named_steps") and "estimator" in estimator_or_pipeline.named_steps:
        estimator = estimator_or_pipeline.named_steps["estimator"]
    else:
        estimator = estimator_or_pipeline

    importances = None

    # Tree-based and ensemble: feature_importances_
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        # Linear models: coef_ (raises AttributeError for SVC with non-linear kernels)
        try:
            coef = np.asarray(estimator.coef_)
            if coef.ndim == 2:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
        except (AttributeError, ValueError):
            return None

    if importances is None or len(importances) != len(feature_names):
        return None

    items = [
        {"name": n, "value": float(v)}
        for n, v in zip(feature_names, importances)
    ]
    items.sort(key=lambda x: x["value"], reverse=True)
    return items


# ── Orchestrator ───────────────────────────────────────────────────────────

def build_evaluation(
    estimator,
    prepared: PreparedData,
    problem_type: str,
    label_map: dict | None,
    feature_names: list[str],
) -> dict:
    if problem_type == "classification":
        result = evaluate_classification(estimator, prepared, label_map)
    elif problem_type == "regression":
        result = evaluate_regression(estimator, prepared)
    else:
        raise ValueError(f"Unknown problem type: {problem_type!r}")

    result["feature_importance"] = compute_feature_importance(estimator, feature_names)
    return result
