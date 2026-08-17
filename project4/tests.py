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


# ── Stage P4-2: feature representation (Task 1) ────────────────────────────

class RatingBucketTest(TestCase):
    def test_family_ratings(self):
        from .services.features import rating_bucket
        for r in ["G", "PG", "TV-G", "Approved", "Passed"]:
            self.assertEqual(rating_bucket(r), "Family")

    def test_teen_ratings(self):
        from .services.features import rating_bucket
        for r in ["PG-13", "TV-14", "GP", "M"]:
            self.assertEqual(rating_bucket(r), "Teen")

    def test_mature_ratings(self):
        from .services.features import rating_bucket
        for r in ["R", "NC-17", "X", "TV-MA"]:
            self.assertEqual(rating_bucket(r), "Mature")

    def test_unknown_becomes_other(self):
        from .services.features import rating_bucket
        for r in ["Not Rated", "Unrated", "Other", "Bogus", None]:
            self.assertEqual(rating_bucket(r), "Other")


class MovieEncoderTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from .services.features import get_features
        cls.X, cls.encoder = get_features()

    def test_matrix_shape_matches_corpus_and_dim(self):
        corpus = get_movie_corpus()
        self.assertEqual(self.X.shape, (corpus.n_movies, self.encoder.dim))

    def test_dimension_is_compact(self):
        # Small d is a design requirement: w is fit from ~15-20 interactions.
        self.assertLess(self.encoder.dim, 50)
        self.assertGreater(self.encoder.dim, 20)

    def test_feature_names_match_dim(self):
        self.assertEqual(len(self.encoder.feature_names), self.encoder.dim)
        self.assertEqual(len(set(self.encoder.feature_names)), self.encoder.dim)

    def test_all_values_finite(self):
        import numpy as np
        self.assertTrue(np.isfinite(self.X).all())

    def test_genre_block_is_binary(self):
        import numpy as np
        n_genres = len(self.encoder.genres)
        self.assertTrue(np.isin(self.X[:, :n_genres], [0.0, 1.0]).all())

    def test_numeric_block_is_standardized(self):
        import numpy as np
        from .services.features import NUMERIC_COLUMNS
        start = len(self.encoder.genres)
        block = self.X[:, start:start + len(NUMERIC_COLUMNS)]
        np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(block.std(axis=0), 1.0, atol=1e-3)

    def test_rating_one_hot_sums_to_one(self):
        import numpy as np
        from .services.features import NUMERIC_COLUMNS, RATING_BUCKETS
        start = len(self.encoder.genres) + len(NUMERIC_COLUMNS)
        block = self.X[:, start:start + len(RATING_BUCKETS)]
        np.testing.assert_array_equal(block.sum(axis=1), np.ones(len(self.X)))

    def test_country_column_is_binary(self):
        import numpy as np
        self.assertTrue(np.isin(self.X[:, -1], [0.0, 1.0]).all())

    def test_no_popularity_features(self):
        # Popularity/acclaim columns must not leak into the taste representation.
        joined = " ".join(self.encoder.feature_names).lower()
        for banned in ["imdb_score", "gross", "voted", "facebook", "likes"]:
            self.assertNotIn(banned, joined)

    def test_encoding_is_reproducible(self):
        import numpy as np
        from .services.features import fit_movie_encoder, transform_movies
        df = get_movie_corpus().df
        enc = fit_movie_encoder(df)
        np.testing.assert_allclose(transform_movies(df, enc), self.X)

    def test_known_movie_vector(self):
        # Avatar: Action|Adventure|Fantasy|Sci-Fi, PG-13 (Teen), USA, 178 min.
        corpus = get_movie_corpus()
        record = corpus.display_record(0)
        self.assertEqual(record["title"], "Avatar")
        names = self.encoder.feature_names
        vector = self.X[0]
        for genre in ["Action", "Adventure", "Fantasy", "Sci-Fi"]:
            self.assertEqual(vector[names.index(f"genre:{genre}")], 1.0)
        self.assertEqual(vector[names.index("genre:Comedy")], 0.0)
        self.assertEqual(vector[names.index("rating:Teen")], 1.0)
        self.assertEqual(vector[names.index("rating:Mature")], 0.0)
        self.assertEqual(vector[names.index("country:USA")], 1.0)
        # 178 min is well above the ~108 min corpus mean
        self.assertGreater(vector[names.index("num:duration")], 1.0)

    def test_cached_returns_same_object(self):
        from .services.features import get_features
        X2, enc2 = get_features()
        self.assertIs(X2, self.X)
        self.assertIs(enc2, self.encoder)


class FeaturesPageTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project4/features/")
        self.assertEqual(response.status_code, 200)

    def test_shows_dimension_and_groups(self):
        response = self.client.get("/project4/features/")
        self.assertContains(response, "Feature Groups")
        self.assertContains(response, "Genres (multi-hot)")
        self.assertContains(response, "Audience maturity")

    def test_documents_exclusions(self):
        response = self.client.get("/project4/features/")
        self.assertContains(response, "Excluded Features")
        self.assertContains(response, "imdb_score")

    def test_documents_preprocessing(self):
        response = self.client.get("/project4/features/")
        self.assertContains(response, "Preprocessing Decisions")
        self.assertContains(response, "Missing values")

    def test_shows_worked_examples(self):
        response = self.client.get("/project4/features/")
        self.assertContains(response, "Worked Examples")
        self.assertContains(response, "genre:Action")

    def test_nav_links_to_features(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "/project4/features/")
