"""Baseline text classifier for AG News (Task 1).

A TF-IDF vectorizer + multinomial logistic regression, wrapped in a single
sklearn Pipeline so downstream code (deferral, active learning) can call
``predict`` / ``predict_proba`` directly on raw article text. Logistic
regression is chosen over a linear SVM because it provides class-probability
estimates via predict_proba() which are needed both for confidence-based deferral (Task 3) and
for uncertainty-sampling active learning (Task 4).

The fitted pipeline and its test-set evaluation are cached to disk (joblib), so
the ~minute-long fit on the full 120k training set happens only once.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from .data import CLASS_NAMES, N_CLASSES, get_agnews

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "project3", "artifacts")
BASELINE_FILE = os.path.join(ARTIFACT_DIR, "baseline.joblib")
BASELINE_EVAL_FILE = os.path.join(ARTIFACT_DIR, "baseline_eval.joblib")

# Model description shown in the interface / report.
MODEL_DESCRIPTION = "TF-IDF (1–2 grams) + multinomial logistic regression"
SEED = 42


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=10.0,
            max_iter=200,
            random_state=SEED,
        )),
    ])


def build_baseline() -> Pipeline:
    """Fit the baseline on the full training set and cache it to disk."""
    import joblib

    data = get_agnews()
    pipe = _build_pipeline()
    pipe.fit(data.X_train, data.y_train)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipe, BASELINE_FILE)
    return pipe


@lru_cache(maxsize=1)
def get_baseline() -> Pipeline:
    """Return the fitted baseline pipeline (disk-cached; built on first use)."""
    import joblib

    if os.path.exists(BASELINE_FILE):
        return joblib.load(BASELINE_FILE)
    return build_baseline()


def build_baseline_eval() -> dict:
    """Evaluate the baseline on the test set and cache predictions + metrics."""
    import joblib

    data = get_agnews()
    pipe = get_baseline()

    y_pred = pipe.predict(data.X_test)
    acc = float(accuracy_score(data.y_test, y_pred))
    cm = confusion_matrix(data.y_test, y_pred, labels=list(range(N_CLASSES)))
    row_sums = cm.sum(axis=1)
    per_class_acc = [
        float(cm[i, i] / row_sums[i]) if row_sums[i] else 0.0
        for i in range(N_CLASSES)
    ]

    result = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
        "n_train": data.n_train,
        "n_test": data.n_test,
        "class_names": list(CLASS_NAMES),
        "y_pred": y_pred.astype(int).tolist(),
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(result, BASELINE_EVAL_FILE)
    return result


@lru_cache(maxsize=1)
def get_baseline_eval() -> dict:
    """Return cached baseline test evaluation (built on first use)."""
    import joblib

    if os.path.exists(BASELINE_EVAL_FILE):
        return joblib.load(BASELINE_EVAL_FILE)
    return build_baseline_eval()
