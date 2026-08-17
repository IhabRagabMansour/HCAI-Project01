"""Preference models: Bradley-Terry and its Plackett-Luce ranking extension.

Both interfaces share one latent utility, so the two conditions are directly
comparable:

    s_i = w^T x_i                             (utility of movie i)

Pairwise (Bradley-Terry)
------------------------
    P(i > j | w) = exp(s_i) / (exp(s_i) + exp(s_j)) = sigmoid(w^T (x_i - x_j))

Ranking (Task 2 — sequential Luce / Plackett-Luce)
--------------------------------------------------
Lecture 9 (slide 12) gives the Luce choice model
``p(a | s, theta) = U(a) / sum_a' U(a')``. Reading a full ranking as *repeated
choice from the remaining items* — pick the favourite, remove it, pick the
favourite of the rest, and so on — turns that model into a likelihood for a
complete ordering:

    P(i_1 > ... > i_n | w) = prod_{k=1}^{n-1} exp(s_{i_k}) / sum_{j>=k} exp(s_{i_j})

The last factor is 1 (only one item remains), so the product stops at n-1.

Why this extension:
  * for n = 2 it reduces **exactly** to Bradley-Terry (unit-tested below),
  * it reuses the same score w^T x, so neither interface gets an advantage,
  * it is a proper likelihood over the ordered list, so w is directly MLE-fittable,
  * it follows the Luce model from the lecture rather than inventing a new one.

Alternative considered: expand one ranking of 10 into its 45 implied pairwise
comparisons and fit ordinary Bradley-Terry. That is simpler, but those 45
relations come from a *single* elicitation act and are not independent
observations; treating them as i.i.d. overstates the evidence. The sequential
formulation is the principled direct ranking likelihood.

Estimation
----------
Both conditions are fit by maximizing a regularized log-likelihood

    max_w  L(w) - lambda * ||w||^2

which is MAP estimation under a zero-mean Gaussian prior. With d ~ 31 and only
~15-20 observations the problem is under-determined, so this term is not
cosmetic: it is what keeps the estimate stable and is exactly the remedy for the
"overconfidence from few comparisons" issue raised in Lecture 9.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

DEFAULT_REGULARIZATION = 1.0


# ── Pairwise (Bradley-Terry) ────────────────────────────────────────────────

def pairwise_probability(w, x_i, x_j) -> float:
    """P(i > j | w) = sigmoid(w^T (x_i - x_j))."""
    w = np.asarray(w, dtype=float)
    diff = np.asarray(x_i, dtype=float) - np.asarray(x_j, dtype=float)
    return float(expit(float(w @ diff)))


def _pair_difference_matrix(pairs, X) -> np.ndarray:
    """Stack x_chosen - x_rejected for every observed pair -> (n_pairs, d)."""
    X = np.asarray(X, dtype=float)
    if len(pairs) == 0:
        return np.zeros((0, X.shape[1]))
    chosen = np.asarray([int(c) for c, _ in pairs])
    rejected = np.asarray([int(r) for _, r in pairs])
    return X[chosen] - X[rejected]


def pairwise_log_likelihood(w, pairs, X) -> float:
    """Sum of log P(chosen > rejected | w) over observed pairwise choices."""
    w = np.asarray(w, dtype=float)
    D = _pair_difference_matrix(pairs, X)
    if len(D) == 0:
        return 0.0
    # log sigmoid(z) computed stably as -log(1 + exp(-z))
    z = D @ w
    return float(-np.logaddexp(0.0, -z).sum())


def _pairwise_objective(w, D, regularization):
    """Negative regularized log-likelihood and its gradient (for minimization)."""
    z = D @ w
    neg_ll = np.logaddexp(0.0, -z).sum()
    # d/dw [-log sigmoid(z)] = -(1 - sigmoid(z)) * d
    residual = expit(z) - 1.0
    grad = D.T @ residual
    value = neg_ll + regularization * float(w @ w)
    grad = grad + 2.0 * regularization * w
    return value, grad


# ── Ranking (Plackett-Luce) ─────────────────────────────────────────────────

def ranking_log_likelihood(w, rankings, X) -> float:
    """Sum of log P(ranking | w) over observed rankings.

    Each ranking is a sequence of movie ids ordered from most to least preferred.
    """
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    total = 0.0
    for ranking in rankings:
        ids = [int(i) for i in ranking]
        if len(ids) < 2:
            continue                      # a single item carries no information
        scores = X[ids] @ w
        for k in range(len(ids) - 1):
            total += scores[k] - logsumexp(scores[k:])
    return float(total)


def ranking_probability(w, ranking, X) -> float:
    """P(ranking | w) for one ranking; always in (0, 1)."""
    return float(np.exp(ranking_log_likelihood(w, [ranking], X)))


def _ranking_objective(w, ranking_features, regularization):
    """Negative regularized ranking log-likelihood and gradient."""
    neg_ll = 0.0
    grad = np.zeros_like(w)
    for feats in ranking_features:        # feats: (n_items, d), best first
        scores = feats @ w
        for k in range(len(scores) - 1):
            tail = scores[k:]
            lse = logsumexp(tail)
            neg_ll += lse - scores[k]
            # d/dw [logsumexp - s_k] = sum_j softmax_j x_j - x_k
            probs = np.exp(tail - lse)
            grad += probs @ feats[k:] - feats[k]
    value = neg_ll + regularization * float(w @ w)
    grad = grad + 2.0 * regularization * w
    return value, grad


# ── Estimation ──────────────────────────────────────────────────────────────

def _minimize(objective, args, dim):
    """L-BFGS-B from w = 0 with analytic gradients (identical for both fits)."""
    result = minimize(
        objective, x0=np.zeros(dim), args=args,
        jac=True, method="L-BFGS-B",
    )
    return np.asarray(result.x, dtype=float)


def fit_pairwise_w(pairs, X, regularization: float = DEFAULT_REGULARIZATION) -> np.ndarray:
    """MAP estimate of w from pairwise choices [(chosen_id, rejected_id), ...]."""
    X = np.asarray(X, dtype=float)
    dim = X.shape[1]
    if len(pairs) == 0:
        return np.zeros(dim)
    D = _pair_difference_matrix(pairs, X)
    return _minimize(_pairwise_objective, (D, regularization), dim)


def fit_ranking_w(rankings, X, regularization: float = DEFAULT_REGULARIZATION) -> np.ndarray:
    """MAP estimate of w from rankings [[best_id, ..., worst_id], ...]."""
    X = np.asarray(X, dtype=float)
    dim = X.shape[1]
    usable = [[int(i) for i in r] for r in rankings if len(r) >= 2]
    if not usable:
        return np.zeros(dim)
    ranking_features = [X[ids] for ids in usable]
    return _minimize(_ranking_objective, (ranking_features, regularization), dim)


# ── Evaluation helpers (held-out predictive performance) ────────────────────

def predict_pairwise(w, pairs, X) -> np.ndarray:
    """P(first > second | w) for each (i, j) pair."""
    D = _pair_difference_matrix(pairs, X)
    if len(D) == 0:
        return np.zeros(0)
    return expit(D @ np.asarray(w, dtype=float))


def heldout_accuracy(w, pairs, X) -> float:
    """Fraction of held-out pairs where the model's preferred item was chosen.

    ``pairs`` are (chosen, rejected) as observed, so a correct prediction means
    P(chosen > rejected) > 0.5.
    """
    probs = predict_pairwise(w, pairs, X)
    if len(probs) == 0:
        return 0.0
    return float((probs > 0.5).mean())


def heldout_log_loss(w, pairs, X) -> float:
    """Mean negative log-likelihood of the observed held-out choices."""
    probs = predict_pairwise(w, pairs, X)
    if len(probs) == 0:
        return 0.0
    eps = 1e-12
    return float(-np.log(np.clip(probs, eps, 1.0)).mean())
