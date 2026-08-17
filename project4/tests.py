import pandas as pd
from django.test import TestCase

from .services.data import (
    clean_movies, genre_vocabulary, get_movie_corpus, load_raw_movies,
    KEEP_COLUMNS, OTHER, REQUIRED_COLUMNS,
)


# ── Stage P4-1: IMDB data service ──────────────────────────────────────────

class CleanMoviesTest(TestCase):
    def _raw(self, **overrides):
        """A minimal raw-CSV-shaped row."""
        row = {
            "movie_title": "Test Movie\xa0",
            "title_year": 2001.0,
            "duration": 120.0,
            "genres": "Action|Drama",
            "content_rating": "PG-13",
            "country": "USA",
            "language": "English",
            "director_name": "Someone",
            "actor_1_name": "Someone Else",
        }
        row.update(overrides)
        return row

    def test_strips_title_whitespace_and_nbsp(self):
        out = clean_movies(pd.DataFrame([self._raw()]))
        self.assertEqual(out.iloc[0]["movie_title"], "Test Movie")

    def test_drops_rows_missing_required_columns(self):
        rows = [self._raw(), self._raw(movie_title="No Duration", duration=None)]
        out = clean_movies(pd.DataFrame(rows))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["movie_title"], "Test Movie")

    def test_fills_missing_categoricals_with_other(self):
        out = clean_movies(pd.DataFrame([self._raw(content_rating=None, country=None)]))
        self.assertEqual(out.iloc[0]["content_rating"], OTHER)
        self.assertEqual(out.iloc[0]["country"], OTHER)

    def test_drops_duplicate_titles(self):
        out = clean_movies(pd.DataFrame([self._raw(), self._raw()]))
        self.assertEqual(len(out), 1)

    def test_same_title_different_year_kept(self):
        rows = [self._raw(), self._raw(title_year=1985.0)]
        out = clean_movies(pd.DataFrame(rows))
        self.assertEqual(len(out), 2)

    def test_numeric_columns_are_ints(self):
        out = clean_movies(pd.DataFrame([self._raw()]))
        self.assertTrue(str(out["title_year"].dtype).startswith("int"))
        self.assertTrue(str(out["duration"].dtype).startswith("int"))

    def test_keeps_only_expected_columns(self):
        out = clean_movies(pd.DataFrame([self._raw()]))
        self.assertEqual(list(out.columns), KEEP_COLUMNS)


class MovieCorpusTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.corpus = get_movie_corpus()

    def test_corpus_is_large(self):
        # ~4800 movies survive cleaning
        self.assertGreater(self.corpus.n_movies, 4000)

    def test_no_missing_values(self):
        self.assertFalse(self.corpus.df.isna().any().any())

    def test_required_columns_present(self):
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.corpus.df.columns)

    def test_no_duplicate_title_year(self):
        dupes = self.corpus.df.duplicated(subset=["movie_title", "title_year"]).sum()
        self.assertEqual(dupes, 0)

    def test_genre_vocabulary_is_sorted_and_nonempty(self):
        genres = genre_vocabulary(self.corpus.df)
        self.assertGreater(len(genres), 10)
        self.assertEqual(genres, sorted(genres))
        self.assertIn("Drama", genres)

    def test_display_record_shape(self):
        rec = self.corpus.display_record(0)
        for key in ["movie_id", "title", "year", "duration", "genres",
                    "genres_display", "content_rating"]:
            self.assertIn(key, rec)
        self.assertIsInstance(rec["genres"], list)
        self.assertGreater(len(rec["genres"]), 0)

    def test_display_records_batch(self):
        recs = self.corpus.display_records([0, 1, 2])
        self.assertEqual(len(recs), 3)
        self.assertEqual([r["movie_id"] for r in recs], [0, 1, 2])

    def test_cached_returns_same_object(self):
        self.assertIs(get_movie_corpus(), self.corpus)

    def test_dataset_has_no_user_ratings(self):
        # Central to the project: no rating/preference columns are carried through.
        for col in self.corpus.df.columns:
            self.assertNotIn("rating", col.replace("content_rating", ""))


# ── Landing page ────────────────────────────────────────────────────────────

class LandingPageTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project4/")
        self.assertEqual(response.status_code, 200)

    def test_explains_both_designs(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "Pairwise")
        self.assertContains(response, "Ranking")

    def test_shows_dataset_info(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "IMDB 5000")
        self.assertContains(response, "Movies after cleaning")

    def test_shows_utility_model(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "Bradley")
        self.assertContains(response, "Plackett")

    def test_shows_example_movies(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "Example Movies")


class HomePageProject4LinkTest(TestCase):
    def test_home_lists_project4(self):
        response = self.client.get("/home/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Project 4")
