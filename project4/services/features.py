"""Task 1 — movie feature representation.

Each movie becomes a vector ``x in R^d`` and each participant a latent
preference vector ``w in R^d`` with utility ``U(x) = w^T x``.

Design constraint that drives every choice here: ``w`` must be estimated from
only a handful of interactions (~15-20). The representation is therefore kept
small, dense, consistently scaled, and human-meaningful — every dimension should
be something a person could plausibly have a taste about.

Feature groups (d = 31 on the cleaned corpus):

    24  genre indicators (multi-hot)      the core axes of movie taste
     2  duration, release year            standardized; "long epics" / "classics"
     4  audience-maturity bucket          one-hot: Family / Teen / Mature / Other
     1  country == USA                    Hollywood vs. international

Deliberately excluded: imdb_score, gross, num_voted_users and the facebook-like
counts. These measure a film's *fame or acclaim*, not the participant's intrinsic
taste; including them would let ``w`` absorb a popularity prior that merely looks
like a preference. Language is also dropped: 93% of the corpus is English, so the
feature is nearly constant and carries almost no signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import genre_vocabulary, get_movie_corpus

# Audience-maturity buckets. Collapsing the 16 raw content ratings into four
# ordered buckets avoids one-hot dimensions with a handful of movies each
# (e.g. only 3 TV-14 titles), which could never be estimated reliably.
RATING_BUCKETS = ["Family", "Teen", "Mature", "Other"]

_RATING_MAP = {
    # Family audiences
    "G": "Family", "PG": "Family", "TV-G": "Family", "TV-PG": "Family",
    "TV-Y": "Family", "TV-Y7": "Family", "Approved": "Family", "Passed": "Family",
    # Teen / young adult
    "PG-13": "Teen", "TV-14": "Teen", "GP": "Teen", "M": "Teen",
    # Mature / restricted
    "R": "Mature", "NC-17": "Mature", "X": "Mature", "TV-MA": "Mature",
}

NUMERIC_COLUMNS = ["duration", "title_year"]

# Standard deviations below this are treated as zero to avoid divide-by-zero.
_STD_EPSILON = 1e-8


def rating_bucket(content_rating: str) -> str:
    """Map a raw IMDB content rating to one of RATING_BUCKETS."""
    return _RATING_MAP.get(str(content_rating), "Other")


@dataclass
class MovieEncoder:
    """Fitted preprocessing state; applying it is fully reproducible."""

    genres: list          # sorted genre vocabulary
    numeric_mean: dict    # column -> corpus mean
    numeric_std: dict     # column -> corpus std (floored at _STD_EPSILON)
    feature_names: list   # length d, aligned with the columns of X

    @property
    def dim(self) -> int:
        return len(self.feature_names)


def fit_movie_encoder(df: pd.DataFrame) -> MovieEncoder:
    """Fit the preprocessing (genre vocabulary + numeric scaling) on the corpus."""
    genres = genre_vocabulary(df)

    numeric_mean, numeric_std = {}, {}
    for col in NUMERIC_COLUMNS:
        values = df[col].astype(float)
        numeric_mean[col] = float(values.mean())
        numeric_std[col] = float(max(values.std(), _STD_EPSILON))

    feature_names = (
        [f"genre:{g}" for g in genres]
        + [f"num:{c}" for c in NUMERIC_COLUMNS]
        + [f"rating:{b}" for b in RATING_BUCKETS]
        + ["country:USA"]
    )
    return MovieEncoder(
        genres=genres,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        feature_names=feature_names,
    )


def transform_movies(df: pd.DataFrame, encoder: MovieEncoder) -> np.ndarray:
    """Return the movie feature matrix X of shape (n_movies, d)."""
    n = len(df)
    blocks = []

    # ── Genres: multi-hot. Left as 0/1 rather than standardized so that a
    # coefficient of w reads directly as "how much this genre is liked".
    genre_index = {g: k for k, g in enumerate(encoder.genres)}
    genre_block = np.zeros((n, len(encoder.genres)), dtype=float)
    for row, value in enumerate(df["genres"].to_numpy()):
        for g in str(value).split("|"):
            k = genre_index.get(g)
            if k is not None:
                genre_block[row, k] = 1.0
    blocks.append(genre_block)

    # ── Numeric: standardized with the corpus statistics stored in the encoder.
    numeric_block = np.zeros((n, len(NUMERIC_COLUMNS)), dtype=float)
    for j, col in enumerate(NUMERIC_COLUMNS):
        values = df[col].astype(float).to_numpy()
        numeric_block[:, j] = (values - encoder.numeric_mean[col]) / encoder.numeric_std[col]
    blocks.append(numeric_block)

    # ── Audience maturity: one-hot over the four buckets.
    bucket_index = {b: k for k, b in enumerate(RATING_BUCKETS)}
    rating_block = np.zeros((n, len(RATING_BUCKETS)), dtype=float)
    for row, value in enumerate(df["content_rating"].to_numpy()):
        rating_block[row, bucket_index[rating_bucket(value)]] = 1.0
    blocks.append(rating_block)

    # ── Country: USA vs. rest of the world.
    usa_block = (df["country"].to_numpy() == "USA").astype(float).reshape(-1, 1)
    blocks.append(usa_block)

    return np.hstack(blocks)


@lru_cache(maxsize=1)
def get_features() -> tuple:
    """Return (X, encoder) for the cleaned corpus, cached for the process."""
    corpus = get_movie_corpus()
    encoder = fit_movie_encoder(corpus.df)
    X = transform_movies(corpus.df, encoder)
    return X, encoder


def describe_feature_groups(encoder: MovieEncoder) -> list:
    """Group summary for the interface / report."""
    return [
        {
            "name": "Genres (multi-hot)",
            "dims": len(encoder.genres),
            "detail": ", ".join(encoder.genres),
            "why": "The core axes of movie taste; each is directly interpretable "
                   "as a preference dimension.",
        },
        {
            "name": "Numeric (standardized)",
            "dims": len(NUMERIC_COLUMNS),
            "detail": "duration, title_year",
            "why": "Real taste dimensions: long epics vs. short films, "
                   "classics vs. recent releases.",
        },
        {
            "name": "Audience maturity (one-hot)",
            "dims": len(RATING_BUCKETS),
            "detail": " / ".join(RATING_BUCKETS),
            "why": "Family-friendly vs. mature is a genuine preference axis; "
                   "collapsing 16 raw ratings avoids near-empty categories.",
        },
        {
            "name": "Country",
            "dims": 1,
            "detail": "country == USA",
            "why": "Hollywood vs. international cinema; a 76/24 split carries "
                   "real signal (unlike language, which is 93% English).",
        },
    ]
