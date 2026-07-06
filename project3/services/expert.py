"""Simulated human expert for AG News (Task 2).

A **topic-specialist expert**: highly accurate on its competence topics and
weak (near-guessing) elsewhere. Its competence region is chosen to *complement*
the baseline classifier — the classifier is weakest on Business and Sci/Tech
(which it confuses with each other), so the expert specializes in exactly those
two topics. This makes deferral genuinely useful: on Business/Sci-Tech the
expert beats the classifier, while on World/Sports the classifier is better and
should keep the decision.

The expert is imperfect (never an oracle) and region-specific. It is simulated
from the true label (the "region" is the true topic), which is the standard way
to model region-specific competence. Predictions are deterministic given a seed
and cached for the full train and test splits.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from django.conf import settings
from sklearn.metrics import accuracy_score

from .data import CLASS_NAMES, N_CLASSES, get_agnews

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "project3", "artifacts")
EXPERT_TEST_FILE = os.path.join(ARTIFACT_DIR, "expert_test.joblib")
EXPERT_TRAIN_FILE = os.path.join(ARTIFACT_DIR, "expert_train.joblib")
EXPERT_EVAL_FILE = os.path.join(ARTIFACT_DIR, "expert_eval.joblib")

# Competence region: the two topics the baseline classifier struggles with.
COMPETENCE_TOPICS = ["Business", "Sci/Tech"]
COMPETENCE_IDS = frozenset(CLASS_NAMES.index(t) for t in COMPETENCE_TOPICS)

P_HIGH = 0.95   # accuracy inside the competence region
P_LOW = 0.45    # accuracy outside it (better than random 0.25, but weak)

EXPERT_SEED = 7
EXPERT_DESCRIPTION = (
    "Topic-specialist expert: ~95% accurate on Business and Sci/Tech "
    "(its competence region), ~45% elsewhere."
)


def simulate_expert(y_true, seed: int) -> np.ndarray:
    """Vectorized region-specific expert simulation.

    In its competence region the expert returns the true label with probability
    P_HIGH; outside it, with probability P_LOW. When wrong, it returns a label
    chosen uniformly among the other classes.
    """
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    in_comp = np.isin(y_true, list(COMPETENCE_IDS))
    p_correct = np.where(in_comp, P_HIGH, P_LOW)
    correct = rng.random(n) < p_correct

    # Uniform wrong label: shift by a random 1..K-1 offset (mod K).
    offsets = rng.integers(1, N_CLASSES, size=n)
    wrong = (y_true + offsets) % N_CLASSES

    return np.where(correct, y_true, wrong).astype(int)


def _cached_predictions(path: str, y_true, seed: int) -> np.ndarray:
    import joblib
    if os.path.exists(path):
        return joblib.load(path)
    preds = simulate_expert(y_true, seed)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(preds, path)
    return preds


@lru_cache(maxsize=1)
def get_expert_test_predictions() -> np.ndarray:
    return _cached_predictions(EXPERT_TEST_FILE, get_agnews().y_test, EXPERT_SEED)


@lru_cache(maxsize=1)
def get_expert_train_predictions() -> np.ndarray:
    return _cached_predictions(EXPERT_TRAIN_FILE, get_agnews().y_train, EXPERT_SEED + 1)


def _per_class_accuracy(y_true, y_pred) -> list:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = []
    for c in range(N_CLASSES):
        mask = y_true == c
        out.append(float((y_pred[mask] == c).mean()) if mask.any() else 0.0)
    return out


def build_expert_eval() -> dict:
    """Evaluate the expert on the test set and analyze strengths/weaknesses."""
    import joblib
    from .baseline import get_baseline_eval

    data = get_agnews()
    expert_pred = get_expert_test_predictions()

    overall = float(accuracy_score(data.y_test, expert_pred))
    per_class = _per_class_accuracy(data.y_test, expert_pred)
    clf_per_class = get_baseline_eval()["per_class_accuracy"]

    # Per-class comparison: where does the expert beat the classifier?
    comparison = []
    for i in range(N_CLASSES):
        comparison.append({
            "cls": CLASS_NAMES[i],
            "expert": per_class[i],
            "classifier": clf_per_class[i],
            "expert_better": per_class[i] > clf_per_class[i],
            "in_competence": i in COMPETENCE_IDS,
        })

    result = {
        "description": EXPERT_DESCRIPTION,
        "competence_topics": list(COMPETENCE_TOPICS),
        "overall_accuracy": overall,
        "per_class_accuracy": per_class,
        "classifier_per_class": clf_per_class,
        "comparison": comparison,
        "class_names": list(CLASS_NAMES),
        "p_high": P_HIGH,
        "p_low": P_LOW,
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(result, EXPERT_EVAL_FILE)
    return result


@lru_cache(maxsize=1)
def get_expert_eval() -> dict:
    import joblib
    if os.path.exists(EXPERT_EVAL_FILE):
        return joblib.load(EXPERT_EVAL_FILE)
    return build_expert_eval()
