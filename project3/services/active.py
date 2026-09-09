"""Active learning for expert-competence discovery (Task 4).

From Task 4 the expert's labels are no longer available during training. The
classifier is already trained on the full labeled set, but we must *actively
query* the expert on selected training examples to learn the expert-correctness
model P(expert correct | x) that drives the deferral decision.

Pool-based loop:
    1. take a pool of unlabeled (no expert label) training examples,
    2. score each by a utility u(x),
    3. query the expert on the highest-utility example(s),
    4. add the (x, expert_correct) label and retrain the deferral model,
    5. repeat until the query budget is exhausted; evaluate on the test set.

Query strategies compared:
  - "uncertainty": margin sampling on the classifier — u = 1 - (p_top1 - p_top2)
    Classifier-uncertain articles are disproportionately
    the Business/Sci-Tech cases where deferral matters, so this concentrates the
    budget on the deferral-relevant region.
  - "random": baseline. Both are averaged over the same number of runs and
    share a random warm-up set per run, so the gap reflects the strategy.

The deferral rule is unchanged (Bayes-optimal: defer when
P(expert correct | x) > max_y P(y|x)); only the expert-correctness model is
learned from the queried labels.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from django.conf import settings
from sklearn.linear_model import LogisticRegression

from .baseline import get_baseline, get_baseline_eval
from .data import get_agnews
from .expert import get_expert_test_predictions, get_expert_train_predictions

ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "project3", "artifacts")
ACTIVE_FILE = os.path.join(ARTIFACT_DIR, "active.joblib")

POOL_SIZE = 3000
BUDGET = 500
CHECKPOINTS = [5, 10, 20, 40, 70, 100, 150, 200, 300, 400, 500]

N_RUNS = 3

# Random warm-up per run, matched to the first checkpoint so both strategies
# start level there.
SEED_SIZE = 5

# Must sit below the reachable ceiling (~0.939) or the queries-to-target
# metric reports "not reached" for both.
TARGET_ACCURACY = 0.93
AL_SEED = 23


def _margin_uncertainty(P) -> np.ndarray:
    """Margin sampling utility: 1 - (p_top1 - p_top2). Higher = more uncertain."""
    s = np.sort(P, axis=1)
    return 1.0 - (s[:, -1] - s[:, -2])


def _team_accuracy_from_cexp(p_exp_correct, conf_test, clf_test, exp_test, y_test):
    """Bayes-optimal deferral team accuracy given a P(expert correct) estimate."""
    defer = p_exp_correct > conf_test
    team = np.where(defer, exp_test, clf_test)
    return float((team == y_test).mean()), float(defer.mean())


def _fit_cexp_predict(F_lab, exp_correct_lab, F_test):
    """Fit P(expert correct|x) on labeled pool and predict on test.
    Falls back to the observed base rate when only one class is present."""
    classes = np.unique(exp_correct_lab)
    if len(classes) < 2:
        return np.full(F_test.shape[0], float(exp_correct_lab.mean()))
    model = LogisticRegression(max_iter=1000).fit(F_lab, exp_correct_lab)
    return model.predict_proba(F_test)[:, 1]


def _run_curve(order, F_pool, exp_correct_pool, F_test, conf_test, clf_test, exp_test, y_test):
    """Reveal pool expert labels in `order`; evaluate team accuracy at checkpoints."""
    curve = []
    for cp in CHECKPOINTS:
        labeled = order[:cp]
        p_exp = _fit_cexp_predict(F_pool[labeled], exp_correct_pool[labeled], F_test)
        acc, rate = _team_accuracy_from_cexp(p_exp, conf_test, clf_test, exp_test, y_test)
        curve.append({"n_queries": cp, "team_accuracy": acc, "deferral_rate": rate})
    return curve


def _seeded_uncertainty_order(perm, unc_order) -> np.ndarray:
    """A random warm-up set, then the rest of the pool most-uncertain first.

    Without a warm-up the first queries are the most ambiguous articles, which
    can all share one expert outcome; P(expert correct | x) then has a single
    class and collapses to deferring everything. The warm-up is the head of
    `perm`, the permutation the random strategy uses, so both start level.
    """
    warm_up = perm[:SEED_SIZE]
    seeded = set(warm_up.tolist())
    rest = [i for i in unc_order if i not in seeded]
    return np.concatenate([warm_up, np.asarray(rest, dtype=int)])


def _average_curves(runs) -> list:
    """Mean team accuracy and deferral rate at each checkpoint across runs."""
    return [
        {
            "n_queries": cp,
            "team_accuracy": float(np.mean([run[j]["team_accuracy"] for run in runs])),
            "deferral_rate": float(np.mean([run[j]["deferral_rate"] for run in runs])),
        }
        for j, cp in enumerate(CHECKPOINTS)
    ]


def _queries_to_target(curve, target) -> int | None:
    for pt in curve:
        if pt["team_accuracy"] >= target:
            return pt["n_queries"]
    return None


def build_active() -> dict:
    import joblib

    data = get_agnews()
    baseline = get_baseline()
    tfidf = baseline.named_steps["tfidf"]
    y_train = np.asarray(data.y_train)
    y_test = np.asarray(data.y_test)

    # ── Pool (no expert labels initially) ──
    rng = np.random.default_rng(AL_SEED)
    pool_idx = rng.choice(len(y_train), size=POOL_SIZE, replace=False)
    X_pool = [data.X_train[i] for i in pool_idx]
    y_pool = y_train[pool_idx]
    F_pool = tfidf.transform(X_pool)
    P_pool = baseline.predict_proba(X_pool)
    exp_pool = np.asarray(get_expert_train_predictions())[pool_idx]
    exp_correct_pool = (exp_pool == y_pool).astype(int)

    # ── Fixed test-time signals ──
    F_test = tfidf.transform(data.X_test)
    P_test = baseline.predict_proba(data.X_test)
    conf_test = P_test.max(axis=1)
    clf_test = np.asarray(get_baseline_eval()["y_pred"])
    exp_test = np.asarray(get_expert_test_predictions())

    # ── Paired comparison: same warm-up per run, same averaging ──
    unc = _margin_uncertainty(P_pool)
    unc_order = np.argsort(-unc)

    uncertainty_runs, random_runs = [], []
    for r in range(N_RUNS):
        perm = np.random.default_rng(AL_SEED + 100 + r).permutation(POOL_SIZE)
        random_runs.append(_run_curve(
            perm, F_pool, exp_correct_pool, F_test, conf_test, clf_test, exp_test, y_test
        ))
        uncertainty_runs.append(_run_curve(
            _seeded_uncertainty_order(perm, unc_order),
            F_pool, exp_correct_pool, F_test, conf_test, clf_test, exp_test, y_test
        ))

    uncertainty_curve = _average_curves(uncertainty_runs)
    random_curve = _average_curves(random_runs)

    # ── Reference lines ──
    classifier_only = float((clf_test == y_test).mean())
    p_exp_full = _fit_cexp_predict(F_pool, exp_correct_pool, F_test)   # all pool labels
    full_supervision, _ = _team_accuracy_from_cexp(p_exp_full, conf_test, clf_test, exp_test, y_test)

    result = {
        "pool_size": POOL_SIZE,
        "budget": BUDGET,
        "checkpoints": list(CHECKPOINTS),
        "uncertainty_curve": uncertainty_curve,
        "random_curve": random_curve,
        "classifier_only": classifier_only,
        "full_supervision": full_supervision,
        "target_accuracy": TARGET_ACCURACY,
        "uncertainty_queries_to_target": _queries_to_target(uncertainty_curve, TARGET_ACCURACY),
        "random_queries_to_target": _queries_to_target(random_curve, TARGET_ACCURACY),
        "n_runs": N_RUNS,
        "seed_size": SEED_SIZE,
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(result, ACTIVE_FILE)
    return result


@lru_cache(maxsize=1)
def get_active() -> dict:
    import joblib
    if os.path.exists(ACTIVE_FILE):
        return joblib.load(ACTIVE_FILE)
    return build_active()


DEMO_POOL_SIZE = 10


@lru_cache(maxsize=1)
def get_demo_query_pool() -> tuple:
    """The most-uncertain articles for the human labeling demo (Task 5).

    Returns a tuple of dicts with a stable 0-based ``position``, the article
    ``text``, and its ``true_label`` — the same uncertainty strategy used by the
    automated active-learning loop, applied to a small candidate set.
    """
    data = get_agnews()
    baseline = get_baseline()
    rng = np.random.default_rng(AL_SEED + 999)
    idx = rng.choice(len(data.y_train), size=400, replace=False)
    X = [data.X_train[i] for i in idx]
    unc = _margin_uncertainty(baseline.predict_proba(X))
    order = np.argsort(-unc)[:DEMO_POOL_SIZE]
    return tuple(
        {
            "position": pos,
            "text": data.X_train[int(idx[k])],
            "true_label": int(data.y_train[int(idx[k])]),
        }
        for pos, k in enumerate(order)
    )
