"""IMDB 5000 Movie Dataset loading and cleaning for Project 4.

The dataset contains metadata for ~5000 movies and, crucially, **no user
ratings or preference labels** — preferences must be elicited directly from a
participant. This module loads the CSV, cleans it into the subset of movies that
can be shown to a participant, and exposes lightweight display records for the
study interface.

Cleaning rules (documented in the report):
  - require the fields the feature representation depends on
    (genres, duration, title_year) — movies missing them are dropped,
  - de-duplicate by title (the raw CSV contains repeated entries),
  - strip the trailing NUL/whitespace characters present in ``movie_title``,
  - fill missing ``content_rating`` / ``country`` with "Other" rather than
    dropping the row, since those are only coarse categorical features.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from django.conf import settings

# The dataset ships with the repository (shared with the other projects).
DATA_FILE = os.path.join(settings.BASE_DIR, "data", "movie_metadata.csv")

# Columns the study/feature pipeline depends on.
REQUIRED_COLUMNS = ["genres", "duration", "title_year"]

# Columns kept for display and/or feature extraction.
KEEP_COLUMNS = [
    "movie_title", "title_year", "duration", "genres",
    "content_rating", "country", "language",
    "director_name", "actor_1_name",
]

OTHER = "Other"


@dataclass
class MovieCorpus:
    """Cleaned movie table plus convenience accessors."""

    df: pd.DataFrame          # cleaned, index reset; row position == movie_id

    @property
    def n_movies(self) -> int:
        return len(self.df)

    def display_records(self, movie_ids) -> list:
        """Display dicts for the given movie ids (row positions)."""
        return [self.display_record(int(i)) for i in movie_ids]

    def display_record(self, movie_id: int) -> dict:
        row = self.df.iloc[int(movie_id)]
        return {
            "movie_id": int(movie_id),
            "title": row["movie_title"],
            "year": int(row["title_year"]),
            "duration": int(row["duration"]),
            "genres": row["genres"].split("|"),
            "genres_display": ", ".join(row["genres"].split("|")),
            "content_rating": row["content_rating"],
            "country": row["country"],
            "director": row["director_name"],
            "actor": row["actor_1_name"],
        }


def _clean_title(value: str) -> str:
    """The raw CSV stores titles with a trailing non-breaking/NUL character."""
    return str(value).replace("\xa0", " ").strip()


def load_raw_movies() -> pd.DataFrame:
    """Read the raw IMDB 5000 CSV."""
    return pd.read_csv(DATA_FILE)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw table into the corpus shown to participants."""
    out = df.copy()

    # Only keep movies that have everything the features need.
    out = out.dropna(subset=REQUIRED_COLUMNS)

    out["movie_title"] = out["movie_title"].map(_clean_title)
    out = out[out["movie_title"].str.len() > 0]

    # Coarse categoricals: keep the row, mark the gap explicitly.
    for col in ["content_rating", "country", "language"]:
        out[col] = out[col].fillna(OTHER)
    for col in ["director_name", "actor_1_name"]:
        out[col] = out[col].fillna("Unknown")

    # The raw file repeats some movies.
    out = out.drop_duplicates(subset=["movie_title", "title_year"])

    out["title_year"] = out["title_year"].astype(int)
    out["duration"] = out["duration"].astype(int)

    out = out[KEEP_COLUMNS].reset_index(drop=True)
    return out


@lru_cache(maxsize=1)
def get_movie_corpus() -> MovieCorpus:
    """Load + clean the corpus (cached in memory for the process)."""
    return MovieCorpus(df=clean_movies(load_raw_movies()))


def genre_vocabulary(df: pd.DataFrame) -> list:
    """Sorted list of all genres appearing in the corpus."""
    seen = set()
    for value in df["genres"]:
        seen.update(value.split("|"))
    return sorted(seen)
