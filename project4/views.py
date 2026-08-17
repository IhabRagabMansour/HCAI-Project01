import numpy as np
from django.shortcuts import render

from .services.data import get_movie_corpus, genre_vocabulary
from .services.features import describe_feature_groups, get_features


def index(request):
    """Landing page: project description + (later) report download and study start."""
    corpus = get_movie_corpus()
    df = corpus.df
    genres = genre_vocabulary(df)

    context = {
        "title": "Project 4 — Preference Elicitation",
        "n_movies": corpus.n_movies,
        "n_genres": len(genres),
        "genres": genres,
        "year_min": int(df["title_year"].min()),
        "year_max": int(df["title_year"].max()),
        "sample_movies": corpus.display_records(range(6)),
    }
    return render(request, "project4/index.html", context)


# Movies used to illustrate the feature representation on the features page.
_EXAMPLE_MOVIE_IDS = [0, 1, 2]


def features(request):
    """Task 1: the movie feature representation and how it is extracted."""
    corpus = get_movie_corpus()
    X, encoder = get_features()

    examples = []
    for movie_id in _EXAMPLE_MOVIE_IDS:
        record = corpus.display_record(movie_id)
        vector = X[movie_id]
        record["nonzero"] = [
            {"name": encoder.feature_names[i], "value": float(vector[i])}
            for i in np.nonzero(vector)[0]
        ]
        examples.append(record)

    context = {
        "title": "Task 1 — Feature Representation",
        "n_movies": corpus.n_movies,
        "dim": encoder.dim,
        "groups": describe_feature_groups(encoder),
        "feature_names": encoder.feature_names,
        "examples": examples,
    }
    return render(request, "project4/features.html", context)
