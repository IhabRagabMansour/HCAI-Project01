"""Learning to defer (Task 3).

For each article the system chooses to either predict (classifier) or defer
(expert). Two strategies are implemented and compared:

1. Expert-advantage model using the Bayes-optimal deferral rule: defer when

       P(expert correct | x)  >  max_y P(y | x)

   i.e. when the expert is more likely correct than the classifier. We use the
   classifier's own softmax confidence max_y P(y|x) as its correctness estimate,
   and learn P(expert correct | x) with a logistic regression on the baseline's
   TF-IDF features. This uses the expert's *competence*, unlike confidence-based
   rejection.

2. Confidence threshold, a naïve rejection baseline:
   defer when the classifier's top probability is below a tuned threshold. The
    This ignores whether the expert is actually better.

Both are trained where expert labels are available (the Task-3 setting) and
evaluated on the test set with the full deferral-quality metric suite plus an
oracle upper bound.

Design note: we deliberately do NOT learn P(classifier correct | x) as a second
model. Individual classifier errors are nearly unpredictable from text, so such
a model regresses to the high base rate (~0.97 everywhere), swamping the real
expert advantage and preventing any deferral. The classifier's calibrated
softmax confidence is a far better estimate of its own correctness for this
decision rule.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from django.conf import settings
from sklearn.linear_model import LogisticRegression

from .baseline import get_baseline, get_baseline_eval
from .data import CLASS_NAMES, get_agnews
from .expert import get_expert_test_predictions, get_expert_train_predictions

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "project3", "artifacts")
DEFER_FILE = os.path.join(ARTIFACT_DIR, "defer.joblib")

DEFER_TRAIN_SIZE = 20000     # train subset used to fit the deferral models
DEFER_SEED = 11
QUERY_COST = 0.0             # optional margin: defer only if advantage > cost


def deferral_metrics(y_true, clf_pred, exp_pred, defer_mask) -> dict:
    """Full deferral-quality metric suite for a given deferral decision."""
    y_true = np.asarray(y_true)
    clf_pred = np.asarray(clf_pred)
    exp_pred = np.asarray(exp_pred)
    defer_mask = np.asarray(defer_mask, dtype=bool)

    team_pred = np.where(defer_mask, exp_pred, clf_pred)
    clf_correct = clf_pred == y_true
    exp_correct = exp_pred == y_true
    n_defer = int(defer_mask.sum())
    n_keep = int((~defer_mask).sum())

    return {
        "team_accuracy": float((team_pred == y_true).mean()),
        "classifier_accuracy": float(clf_correct.mean()),
        "expert_accuracy": float(exp_correct.mean()),
        "deferral_rate": float(defer_mask.mean()),
        "accuracy_deferred": float((team_pred[defer_mask] == y_true[defer_mask]).mean()) if n_defer else 0.0,
        "accuracy_kept": float((team_pred[~defer_mask] == y_true[~defer_mask]).mean()) if n_keep else 0.0,
        "useful_deferral_frac": float((defer_mask & exp_correct & ~clf_correct).sum() / n_defer) if n_defer else 0.0,
        "harmful_deferral_frac": float((defer_mask & ~exp_correct & clf_correct).sum() / n_defer) if n_defer else 0.0,
        "expert_correct_when_deferred": float(exp_correct[defer_mask].mean()) if n_defer else 0.0,
        "classifier_correct_when_kept": float(clf_correct[~defer_mask].mean()) if n_keep else 0.0,
        # Oracle: always route to whoever is right (upper bound on team accuracy).
        "oracle_accuracy": float((clf_correct | exp_correct).mean()),
        "n_deferred": n_defer,
        "n_kept": n_keep,
    }


def _train_expert_correctness(F_train, y_train_sub, exp_pred_train):
    """Fit c_exp = P(expert correct | x) on TF-IDF features F_train."""
    exp_correct = (exp_pred_train == y_train_sub).astype(int)
    return LogisticRegression(max_iter=1000).fit(F_train, exp_correct)


def _tune_confidence_threshold(conf_train, clf_pred_train, exp_pred_train, y_train_sub):
    """Pick the threshold maximizing team accuracy on the training subset."""
    best_tau, best_acc = 0.5, -1.0
    for tau in np.linspace(0.30, 0.99, 70):
        defer = conf_train < tau
        team = np.where(defer, exp_pred_train, clf_pred_train)
        acc = (team == y_train_sub).mean()
        if acc > best_acc:
            best_acc, best_tau = acc, float(tau)
    return best_tau


def build_deferral() -> dict:
    import joblib

    data = get_agnews()
    baseline = get_baseline()
    y_train = np.asarray(data.y_train)
    y_test = np.asarray(data.y_test)

    # ── Deferral-model training data (expert labels available in Task 3) ──
    rng = np.random.default_rng(DEFER_SEED)
    idx = rng.choice(len(y_train), size=DEFER_TRAIN_SIZE, replace=False)
    X_train_sub = [data.X_train[i] for i in idx]
    y_train_sub = y_train[idx]
    P_train = baseline.predict_proba(X_train_sub)
    clf_pred_train = P_train.argmax(axis=1)
    exp_pred_train = np.asarray(get_expert_train_predictions())[idx]

    # TF-IDF features (reuse the fitted baseline vectorizer) for the expert
    # correctness model — see the module docstring for why.
    tfidf = baseline.named_steps["tfidf"]
    F_train = tfidf.transform(X_train_sub)
    F_test = tfidf.transform(data.X_test)

    c_exp = _train_expert_correctness(F_train, y_train_sub, exp_pred_train)

    # ── Test-time signals ──
    P_test = baseline.predict_proba(data.X_test)
    clf_pred_test = np.asarray(get_baseline_eval()["y_pred"])
    exp_pred_test = np.asarray(get_expert_test_predictions())

    # Strategy 1: Bayes-optimal deferral — defer when the expert is more likely
    # correct than the classifier's own confidence.
    p_exp_correct = c_exp.predict_proba(F_test)[:, 1]
    p_clf_conf = P_test.max(axis=1)          # classifier's own correctness estimate
    advantage = p_exp_correct - p_clf_conf
    defer_adv = advantage > QUERY_COST

    # Strategy 2: confidence-threshold baseline (tuned on train, no test leak).
    conf_train = P_train.max(axis=1)
    tau = _tune_confidence_threshold(conf_train, clf_pred_train, exp_pred_train, y_train_sub)
    defer_conf = P_test.max(axis=1) < tau

    adv_metrics = deferral_metrics(y_test, clf_pred_test, exp_pred_test, defer_adv)
    conf_metrics = deferral_metrics(y_test, clf_pred_test, exp_pred_test, defer_conf)

    # Example deferred / kept articles under the expert-advantage strategy.
    def _examples(mask, want, limit=4):
        out = []
        for i in np.where(mask == want)[0][:limit]:
            out.append({
                "text": data.X_test[i],
                "true": CLASS_NAMES[int(y_test[i])],
                "expert": CLASS_NAMES[int(exp_pred_test[i])],
                "classifier": CLASS_NAMES[int(clf_pred_test[i])],
            })
        return out

    result = {
        "advantage": adv_metrics,
        "confidence": conf_metrics,
        "confidence_threshold": tau,
        "query_cost": QUERY_COST,
        "deferred_examples": _examples(defer_adv, True),
        "kept_examples": _examples(defer_adv, False),
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(result, DEFER_FILE)
    return result


@lru_cache(maxsize=1)
def get_deferral() -> dict:
    import joblib
    if os.path.exists(DEFER_FILE):
        return joblib.load(DEFER_FILE)
    return build_deferral()
