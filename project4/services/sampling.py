"""Movie sampling for the study.

The project sheet explicitly permits sampling movies **uniformly at random**,
which is what we do. The important design work is not the sampling distribution
but the *allocation*: to control carry-over between conditions, every session
draws one pool of distinct movies and splits it into four disjoint parts —

    practice        movies used only in the (discarded) practice trials
    pairwise        movies shown only in the pairwise condition
    ranking         movies shown only in the ranking condition
    held-out        movies used only in the shared final evaluation

No movie is ever seen twice by the same participant. That removes memory and
recognition effects between the two conditions, and it means the shared held-out
evaluation is genuinely about generalization rather than recall.

The whole plan is regenerated deterministically from the session's seed, so the
database only has to store one integer rather than the full trial list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class StudyConfig:
    """Elicitation budget. Frozen so a running study cannot drift."""

    n_pairwise_trials: int = 15
    n_ranking_trials: int = 3
    ranking_size: int = 10          # the sheet requires exactly ten movies
    n_heldout_pairs: int = 10
    n_practice_pairwise: int = 1
    n_practice_ranking: int = 1

    @property
    def movies_needed(self) -> int:
        return (
            self.n_pairwise_trials * 2
            + self.n_ranking_trials * self.ranking_size
            + self.n_heldout_pairs * 2
            + self.n_practice_pairwise * 2
            + self.n_practice_ranking * self.ranking_size
        )


DEFAULT_CONFIG = StudyConfig()


@dataclass
class TrialPlan:
    """All movie ids a session will ever show, split by purpose."""

    pairwise: list          # [(left_id, right_id), ...]
    ranking: list           # [[10 movie ids], ...]
    heldout: list           # [(left_id, right_id), ...]
    practice_pairwise: list
    practice_ranking: list

    @property
    def all_movie_ids(self) -> list:
        ids = []
        for a, b in self.pairwise + self.heldout + self.practice_pairwise:
            ids += [a, b]
        for group in self.ranking + self.practice_ranking:
            ids += list(group)
        return ids


def _take_pairs(pool, count):
    """Consume 2*count ids from the front of `pool` as (left, right) pairs."""
    pairs = []
    for k in range(count):
        pairs.append((int(pool[2 * k]), int(pool[2 * k + 1])))
    return pairs, pool[2 * count:]


def _take_groups(pool, count, size):
    """Consume count*size ids from the front of `pool` as groups of `size`."""
    groups = []
    for k in range(count):
        groups.append([int(v) for v in pool[k * size:(k + 1) * size]])
    return groups, pool[count * size:]


def build_trial_plan(seed: int, n_movies: int, config: StudyConfig = DEFAULT_CONFIG) -> TrialPlan:
    """Draw one disjoint pool for the session and split it by purpose."""
    if n_movies < config.movies_needed:
        raise ValueError(
            f"corpus has {n_movies} movies but the study needs {config.movies_needed}"
        )

    rng = np.random.default_rng(seed)
    pool = list(rng.choice(n_movies, size=config.movies_needed, replace=False))

    practice_pairwise, pool = _take_pairs(pool, config.n_practice_pairwise)
    practice_ranking, pool = _take_groups(pool, config.n_practice_ranking, config.ranking_size)
    pairwise, pool = _take_pairs(pool, config.n_pairwise_trials)
    ranking, pool = _take_groups(pool, config.n_ranking_trials, config.ranking_size)
    heldout, pool = _take_pairs(pool, config.n_heldout_pairs)

    return TrialPlan(
        pairwise=pairwise,
        ranking=ranking,
        heldout=heldout,
        practice_pairwise=practice_pairwise,
        practice_ranking=practice_ranking,
    )


def new_session_seed() -> int:
    """A fresh random seed for a new participant session."""
    return int(np.random.default_rng().integers(1, 2**31 - 1))
