from django.shortcuts import render

from .services.data import get_movie_corpus, genre_vocabulary


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
