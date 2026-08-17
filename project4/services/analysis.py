"""Turning one participant's elicited data into two comparable models.

This is where Task 2 meets Task 4. Each condition produces its own estimate of
the same latent utility vector w:

    pairwise   15 binary choices          -> Bradley-Terry MAP fit
    ranking     3 orderings of ten films  -> Plackett-Luce MAP fit

Both are then scored on the *same* ten held-out pairwise choices, which the
participant makes once, after both conditions. That shared yardstick is the
whole point of the design: the two methods differ in what the participant did,
not in what they are measured against, so the accuracy difference is
attributable to the elicitation method rather than to the test.

The held-out pairs use films neither condition showed (see `sampling`), so
this measures generalization to unseen movies rather than recall.
"""

from __future__ import annotations

import numpy as np

from ..models import PreferenceModelFit, RankingTrial, PairwiseTrial, BLOCK_HELDOUT, BLOCK_MAIN
from .features import get_features
from .preference_models import (
    fit_pairwise_w, fit_ranking_w, heldout_accuracy, heldout_log_loss,
)
from .study import PAIRWISE, RANKING

# How many liked/disliked features to show a participant in the debrief.
TOP_FEATURE_COUNT = 5


# ── Reading a session's elicited data ───────────────────────────────────────

def pairwise_pairs(session, block=BLOCK_MAIN) -> list:
    """Observed choices as (chosen_id, rejected_id), which is what the model wants."""
    return [
        (trial.chosen_movie_id, trial.rejected_movie_id)
        for trial in PairwiseTrial.objects.filter(session=session, block=block)
    ]


def rankings(session, block=BLOCK_MAIN) -> list:
    """Observed rankings, best film first."""
    return [
        [int(movie_id) for movie_id in trial.ranked_movie_ids]
        for trial in RankingTrial.objects.filter(session=session, block=block)
    ]


# ── Fitting ─────────────────────────────────────────────────────────────────

def fit_session_models(session) -> dict:
    """Fit w for both conditions, score both on the held-out pairs, store both.

    Idempotent: re-running replaces the stored fits rather than duplicating them,
    so a refresh or a re-entry cannot corrupt the record.
    """
    X, _ = get_features()
    heldout = pairwise_pairs(session, BLOCK_HELDOUT)

    observations = {
        PAIRWISE: pairwise_pairs(session, BLOCK_MAIN),
        RANKING: rankings(session, BLOCK_MAIN),
    }
    weights = {
        PAIRWISE: fit_pairwise_w(observations[PAIRWISE], X),
        RANKING: fit_ranking_w(observations[RANKING], X),
    }

    fits = {}
    for condition, w in weights.items():
        fit, _ = PreferenceModelFit.objects.update_or_create(
            session=session, condition=condition,
            defaults={
                "weights": [float(value) for value in w],
                "n_observations": len(observations[condition]),
                # No held-out data yet means no honest score to report.
                "heldout_accuracy": (
                    heldout_accuracy(w, heldout, X) if heldout else None),
                "heldout_log_loss": (
                    heldout_log_loss(w, heldout, X) if heldout else None),
            },
        )
        fits[condition] = fit
    return fits


# ── Interpreting w for the participant ──────────────────────────────────────

_RATING_LABELS = {
    "Family": "family-friendly films",
    "Teen": "teen-rated films",
    "Mature": "films rated for adults",
    "Other": "films with an unusual rating",
}


def humanize_feature(name: str) -> str:
    """A feature name a participant can read, e.g. 'genre:Sci-Fi' -> 'Sci-Fi films'."""
    kind, _, value = name.partition(":")
    if kind == "genre":
        return f"{value} films"
    if kind == "rating":
        return _RATING_LABELS.get(value, f"{value}-rated films")
    if kind == "country":
        return "films made in the USA"
    if name == "num:duration":
        return "longer films"
    if name == "num:title_year":
        return "more recent films"
    return name


def top_features(weights, encoder, count: int = TOP_FEATURE_COUNT) -> dict:
    """The most strongly liked and disliked features of a fitted w.

    Genre, rating and country features are 0/1, so their weight is the change in
    utility from having that property; the two numeric features are standardized,
    so theirs is the change per standard deviation. Both are therefore "per one
    unit as encoded" and can be read on the same scale.
    """
    w = np.asarray(weights, dtype=float)
    order = np.argsort(w)                     # most negative first

    def entries(indices):
        return [
            {
                "name": encoder.feature_names[i],
                "label": humanize_feature(encoder.feature_names[i]),
                "weight": float(w[i]),
            }
            for i in indices
            if abs(float(w[i])) > 1e-6        # a zero weight says nothing
        ]

    return {
        "liked": entries(reversed(order[-count:])),
        "disliked": entries(order[:count]),
    }


# ── The debrief summary ─────────────────────────────────────────────────────

_CONDITION_LABELS = {
    PAIRWISE: "Choosing between two films",
    RANKING: "Ranking ten films",
}


def session_summary(session) -> dict:
    """Everything the debrief page shows about this participant's two models."""
    _, encoder = get_features()
    fits = {fit.condition: fit for fit in PreferenceModelFit.objects.filter(session=session)}

    methods = []
    for condition in (PAIRWISE, RANKING):
        fit = fits.get(condition)
        if fit is None:
            continue
        methods.append({
            "condition": condition,
            "label": _CONDITION_LABELS[condition],
            "n_observations": fit.n_observations,
            "accuracy": fit.heldout_accuracy,
            "accuracy_percent": (
                None if fit.heldout_accuracy is None
                else round(100.0 * fit.heldout_accuracy)),
            "log_loss": fit.heldout_log_loss,
            "features": top_features(fit.weights, encoder),
        })

    scored = [m for m in methods if m["accuracy"] is not None]
    best, tied, on_log_loss = None, False, False
    if len(scored) == 2:
        first, second = scored
        if first["accuracy"] != second["accuracy"]:
            best = max(scored, key=lambda m: m["accuracy"])
        elif _both_have_log_loss(first, second) and first["log_loss"] != second["log_loss"]:
            # Ten held-out pairs give accuracy only eleven possible values, so
            # exact ties are common. Log loss still separates the two: it asks
            # not just whether a model was right but how confident it was.
            best = min(scored, key=lambda m: m["log_loss"])
            on_log_loss = True
        else:
            tied = True

    return {
        "methods": methods,
        "best": best,
        "tied": tied,
        "decided_on_log_loss": on_log_loss,
        "n_heldout": PairwiseTrial.objects.filter(
            session=session, block=BLOCK_HELDOUT).count(),
    }


def _both_have_log_loss(first, second) -> bool:
    return first["log_loss"] is not None and second["log_loss"] is not None
