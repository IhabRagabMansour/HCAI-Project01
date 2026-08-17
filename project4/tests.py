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


# ── Stage P4-3: preference models (Task 2) ─────────────────────────────────

class PairwiseModelTest(TestCase):
    def setUp(self):
        import numpy as np
        self.X = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        self.w = np.array([2.0, -1.0])

    def test_probabilities_are_complementary(self):
        from .services.preference_models import pairwise_probability
        p_ij = pairwise_probability(self.w, self.X[0], self.X[1])
        p_ji = pairwise_probability(self.w, self.X[1], self.X[0])
        self.assertAlmostEqual(p_ij + p_ji, 1.0, places=12)

    def test_equal_items_give_half(self):
        from .services.preference_models import pairwise_probability
        self.assertAlmostEqual(
            pairwise_probability(self.w, self.X[0], self.X[0]), 0.5, places=12
        )

    def test_higher_utility_is_preferred(self):
        from .services.preference_models import pairwise_probability
        # w favours feature 0 and dislikes feature 1, so item 0 beats item 1.
        self.assertGreater(pairwise_probability(self.w, self.X[0], self.X[1]), 0.5)

    def test_probability_in_unit_interval(self):
        from .services.preference_models import pairwise_probability
        for i in range(3):
            for j in range(3):
                p = pairwise_probability(self.w, self.X[i], self.X[j])
                self.assertGreater(p, 0.0)
                self.assertLess(p, 1.0)

    def test_log_likelihood_matches_manual(self):
        import numpy as np
        from .services.preference_models import (
            pairwise_log_likelihood, pairwise_probability,
        )
        pairs = [(0, 1), (2, 1)]
        expected = np.log(pairwise_probability(self.w, self.X[0], self.X[1])) + \
            np.log(pairwise_probability(self.w, self.X[2], self.X[1]))
        self.assertAlmostEqual(
            pairwise_log_likelihood(self.w, pairs, self.X), float(expected), places=10
        )

    def test_empty_pairs_give_zero(self):
        from .services.preference_models import pairwise_log_likelihood
        self.assertEqual(pairwise_log_likelihood(self.w, [], self.X), 0.0)

    def test_log_likelihood_is_negative_and_finite(self):
        import numpy as np
        from .services.preference_models import pairwise_log_likelihood
        ll = pairwise_log_likelihood(self.w, [(0, 1), (1, 2)], self.X)
        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0.0)


class RankingModelTest(TestCase):
    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(6, 4))
        self.w = rng.normal(size=4)

    def test_two_item_ranking_equals_bradley_terry(self):
        """THE key correctness property: Plackett-Luce reduces to Bradley-Terry."""
        import numpy as np
        from .services.preference_models import (
            pairwise_probability, ranking_probability,
        )
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                bt = pairwise_probability(self.w, self.X[i], self.X[j])
                pl = ranking_probability(self.w, [i, j], self.X)
                self.assertAlmostEqual(pl, bt, places=12)

    def test_two_item_log_likelihoods_agree(self):
        from .services.preference_models import (
            pairwise_log_likelihood, ranking_log_likelihood,
        )
        pairs = [(0, 1), (2, 3), (4, 5)]
        rankings = [[c, r] for c, r in pairs]
        self.assertAlmostEqual(
            ranking_log_likelihood(self.w, rankings, self.X),
            pairwise_log_likelihood(self.w, pairs, self.X),
            places=10,
        )

    def test_ranking_probability_in_unit_interval(self):
        from .services.preference_models import ranking_probability
        p = ranking_probability(self.w, [0, 1, 2, 3, 4, 5], self.X)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_all_permutations_sum_to_one(self):
        """A proper distribution over orderings must sum to 1."""
        import itertools
        from .services.preference_models import ranking_probability
        items = [0, 1, 2, 3]
        total = sum(
            ranking_probability(self.w, list(perm), self.X)
            for perm in itertools.permutations(items)
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_best_ordering_is_most_probable(self):
        import itertools
        import numpy as np
        from .services.preference_models import ranking_probability
        items = [0, 1, 2, 3]
        scores = self.X[items] @ self.w
        best = [items[k] for k in np.argsort(-scores)]
        best_p = ranking_probability(self.w, best, self.X)
        for perm in itertools.permutations(items):
            self.assertLessEqual(ranking_probability(self.w, list(perm), self.X), best_p + 1e-12)

    def test_log_likelihood_finite_for_valid_data(self):
        import numpy as np
        from .services.preference_models import ranking_log_likelihood
        ll = ranking_log_likelihood(self.w, [[0, 1, 2], [3, 4, 5]], self.X)
        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0.0)

    def test_single_item_ranking_is_ignored(self):
        from .services.preference_models import ranking_log_likelihood
        self.assertEqual(ranking_log_likelihood(self.w, [[2]], self.X), 0.0)


class GradientCheckTest(TestCase):
    """Analytic gradients must match numerical differentiation."""

    def _numerical_gradient(self, f, w, eps=1e-6):
        import numpy as np
        grad = np.zeros_like(w)
        for k in range(len(w)):
            step = np.zeros_like(w)
            step[k] = eps
            grad[k] = (f(w + step) - f(w - step)) / (2 * eps)
        return grad

    def test_pairwise_gradient(self):
        import numpy as np
        from .services.preference_models import _pair_difference_matrix, _pairwise_objective
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5, 3))
        pairs = [(0, 1), (2, 3), (4, 0)]
        D = _pair_difference_matrix(pairs, X)
        w = rng.normal(size=3)
        _, analytic = _pairwise_objective(w, D, 0.5)
        numeric = self._numerical_gradient(lambda v: _pairwise_objective(v, D, 0.5)[0], w)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_ranking_gradient(self):
        import numpy as np
        from .services.preference_models import _ranking_objective
        rng = np.random.default_rng(2)
        X = rng.normal(size=(6, 3))
        feats = [X[[0, 1, 2, 3]], X[[4, 5, 0]]]
        w = rng.normal(size=3)
        _, analytic = _ranking_objective(w, feats, 0.5)
        numeric = self._numerical_gradient(lambda v: _ranking_objective(v, feats, 0.5)[0], w)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)


class SyntheticRecoveryTest(TestCase):
    """Fitting simulated preferences should approximately recover the true w."""

    @staticmethod
    def _cosine(a, b):
        import numpy as np
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_pairwise_recovers_direction(self):
        import numpy as np
        from .services.preference_models import fit_pairwise_w
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 5))
        w_true = np.array([2.0, -1.5, 0.5, 0.0, 1.0])
        pairs = []
        for _ in range(600):
            i, j = rng.choice(200, size=2, replace=False)
            # Simulate a Bradley-Terry choice from the true w.
            p = 1.0 / (1.0 + np.exp(-(X[i] - X[j]) @ w_true))
            pairs.append((i, j) if rng.random() < p else (j, i))
        w_hat = fit_pairwise_w(pairs, X, regularization=0.01)
        self.assertGreater(self._cosine(w_hat, w_true), 0.9)

    def test_ranking_recovers_direction(self):
        import numpy as np
        from .services.preference_models import fit_ranking_w
        rng = np.random.default_rng(8)
        X = rng.normal(size=(200, 5))
        w_true = np.array([2.0, -1.5, 0.5, 0.0, 1.0])
        rankings = []
        for _ in range(60):
            items = list(rng.choice(200, size=10, replace=False))
            # Plackett-Luce sampling: repeatedly draw from a softmax over the rest.
            remaining, order = list(items), []
            while remaining:
                scores = X[remaining] @ w_true
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                pick = rng.choice(len(remaining), p=probs)
                order.append(remaining.pop(pick))
            rankings.append(order)
        w_hat = fit_ranking_w(rankings, X, regularization=0.01)
        self.assertGreater(self._cosine(w_hat, w_true), 0.9)

    def test_both_fitters_agree_on_two_item_data(self):
        """Same data expressed as pairs or as 2-item rankings -> same estimate."""
        import numpy as np
        from .services.preference_models import fit_pairwise_w, fit_ranking_w
        rng = np.random.default_rng(9)
        X = rng.normal(size=(40, 4))
        pairs = [(int(a), int(b)) for a, b in rng.choice(40, size=(50, 2))
                 if a != b]
        rankings = [[c, r] for c, r in pairs]
        w_pair = fit_pairwise_w(pairs, X, regularization=0.5)
        w_rank = fit_ranking_w(rankings, X, regularization=0.5)
        np.testing.assert_allclose(w_pair, w_rank, atol=1e-5)


class FitEdgeCaseTest(TestCase):
    def test_no_observations_returns_zero_vector(self):
        import numpy as np
        from .services.preference_models import fit_pairwise_w, fit_ranking_w
        X = np.eye(4)
        np.testing.assert_array_equal(fit_pairwise_w([], X), np.zeros(4))
        np.testing.assert_array_equal(fit_ranking_w([], X), np.zeros(4))

    def test_regularization_shrinks_estimate(self):
        import numpy as np
        from .services.preference_models import fit_pairwise_w
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 4))
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        weak = fit_pairwise_w(pairs, X, regularization=0.01)
        strong = fit_pairwise_w(pairs, X, regularization=100.0)
        self.assertLess(np.linalg.norm(strong), np.linalg.norm(weak))

    def test_fit_works_with_real_movie_features(self):
        import numpy as np
        from .services.features import get_features
        from .services.preference_models import fit_pairwise_w, fit_ranking_w
        X, encoder = get_features()
        w_pair = fit_pairwise_w([(0, 1), (2, 3)], X)
        w_rank = fit_ranking_w([list(range(10))], X)
        self.assertEqual(w_pair.shape, (encoder.dim,))
        self.assertEqual(w_rank.shape, (encoder.dim,))
        self.assertTrue(np.isfinite(w_pair).all())
        self.assertTrue(np.isfinite(w_rank).all())


class HeldOutEvaluationTest(TestCase):
    def test_perfect_model_scores_well(self):
        import numpy as np
        from .services.preference_models import heldout_accuracy, heldout_log_loss
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([10.0, -10.0])          # strongly prefers item 0
        pairs = [(0, 1)]
        self.assertEqual(heldout_accuracy(w, pairs, X), 1.0)
        self.assertLess(heldout_log_loss(w, pairs, X), 0.01)

    def test_wrong_model_scores_poorly(self):
        import numpy as np
        from .services.preference_models import heldout_accuracy, heldout_log_loss
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([-10.0, 10.0])          # prefers the rejected item
        pairs = [(0, 1)]
        self.assertEqual(heldout_accuracy(w, pairs, X), 0.0)
        self.assertGreater(heldout_log_loss(w, pairs, X), 1.0)

    def test_uninformed_model_is_chance(self):
        import numpy as np
        from .services.preference_models import heldout_log_loss
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.zeros(2)
        self.assertAlmostEqual(heldout_log_loss(w, [(0, 1)], X), np.log(2), places=9)


# ── Stage P4-4: sampling, state machine, models, opening flow ──────────────

class TrialPlanTest(TestCase):
    def _plan(self, seed=123):
        from .services.sampling import build_trial_plan
        return build_trial_plan(seed, n_movies=4801)

    def test_counts_match_config(self):
        from .services.sampling import DEFAULT_CONFIG
        plan = self._plan()
        self.assertEqual(len(plan.pairwise), DEFAULT_CONFIG.n_pairwise_trials)
        self.assertEqual(len(plan.ranking), DEFAULT_CONFIG.n_ranking_trials)
        self.assertEqual(len(plan.heldout), DEFAULT_CONFIG.n_heldout_pairs)

    def test_every_ranking_has_exactly_ten_movies(self):
        """Hard requirement from the project sheet."""
        plan = self._plan()
        for group in plan.ranking + plan.practice_ranking:
            self.assertEqual(len(group), 10)

    def test_all_movies_distinct_across_whole_session(self):
        """No movie is ever shown twice to the same participant."""
        ids = self._plan().all_movie_ids
        self.assertEqual(len(ids), len(set(ids)))

    def test_condition_pools_are_disjoint(self):
        """Carry-over control: pairwise and ranking share no movies."""
        plan = self._plan()
        pairwise_ids = {m for pair in plan.pairwise for m in pair}
        ranking_ids = {m for group in plan.ranking for m in group}
        self.assertEqual(pairwise_ids & ranking_ids, set())

    def test_heldout_disjoint_from_both_conditions(self):
        plan = self._plan()
        heldout_ids = {m for pair in plan.heldout for m in pair}
        pairwise_ids = {m for pair in plan.pairwise for m in pair}
        ranking_ids = {m for group in plan.ranking for m in group}
        self.assertEqual(heldout_ids & pairwise_ids, set())
        self.assertEqual(heldout_ids & ranking_ids, set())

    def test_no_pair_shows_the_same_movie_twice(self):
        plan = self._plan()
        for left, right in plan.pairwise + plan.heldout + plan.practice_pairwise:
            self.assertNotEqual(left, right)

    def test_plan_is_deterministic_for_a_seed(self):
        self.assertEqual(self._plan(7).all_movie_ids, self._plan(7).all_movie_ids)

    def test_different_seeds_give_different_plans(self):
        self.assertNotEqual(self._plan(1).all_movie_ids, self._plan(2).all_movie_ids)

    def test_movie_ids_within_corpus_range(self):
        plan = self._plan()
        for movie_id in plan.all_movie_ids:
            self.assertGreaterEqual(movie_id, 0)
            self.assertLess(movie_id, 4801)

    def test_raises_when_corpus_too_small(self):
        from .services.sampling import build_trial_plan
        with self.assertRaises(ValueError):
            build_trial_plan(1, n_movies=10)


class StudyFlowTest(TestCase):
    def test_counterbalancing_alternates(self):
        from .services.study import ORDER_AB, ORDER_BA, assign_condition_order
        self.assertEqual(assign_condition_order(0), ORDER_AB)
        self.assertEqual(assign_condition_order(1), ORDER_BA)
        self.assertEqual(assign_condition_order(2), ORDER_AB)

    def test_orders_run_conditions_in_opposite_sequence(self):
        from .services.study import (
            ORDER_AB, ORDER_BA, PAIRWISE, RANKING, conditions_for_order,
        )
        self.assertEqual(conditions_for_order(ORDER_AB), [PAIRWISE, RANKING])
        self.assertEqual(conditions_for_order(ORDER_BA), [RANKING, PAIRWISE])

    def test_both_orders_contain_both_conditions(self):
        from .services.study import CONDITION_ORDERS, PAIRWISE, RANKING, conditions_for_order
        for order in CONDITION_ORDERS:
            self.assertEqual(set(conditions_for_order(order)), {PAIRWISE, RANKING})

    def test_step_sequence_walks_to_completion(self):
        from .services.study import COMPLETE, FIRST_STEP, next_step, total_steps
        step, seen = FIRST_STEP, [FIRST_STEP]
        for _ in range(total_steps() + 5):
            step = next_step(step)
            seen.append(step)
            if step == COMPLETE:
                break
        self.assertEqual(seen[-1], COMPLETE)

    def test_complete_is_a_fixed_point(self):
        from .services.study import COMPLETE, next_step
        self.assertEqual(next_step(COMPLETE), COMPLETE)

    def test_unknown_step_raises(self):
        from .services.study import next_step
        with self.assertRaises(ValueError):
            next_step("nonsense")

    def test_condition_for_step_respects_order(self):
        from .services.study import (
            CONDITION_1, CONDITION_2, ORDER_AB, ORDER_BA, PAIRWISE, RANKING,
            condition_for_step,
        )
        self.assertEqual(condition_for_step(CONDITION_1, ORDER_AB), PAIRWISE)
        self.assertEqual(condition_for_step(CONDITION_2, ORDER_AB), RANKING)
        self.assertEqual(condition_for_step(CONDITION_1, ORDER_BA), RANKING)
        self.assertEqual(condition_for_step(CONDITION_2, ORDER_BA), PAIRWISE)

    def test_progress_is_monotonic(self):
        from .services.study import STEP_SEQUENCE, progress
        percents = [progress(s)["percent"] for s in STEP_SEQUENCE]
        self.assertEqual(percents, sorted(percents))
        self.assertEqual(percents[-1], 100)


class StudyModelTest(TestCase):
    def test_participant_codes_are_unique_and_non_identifying(self):
        from .models import StudySession
        codes = set()
        for _ in range(5):
            s = StudySession.objects.create(condition_order="AB", seed=1)
            self.assertNotIn(s.participant_code, codes)
            self.assertGreaterEqual(len(s.participant_code), 8)
            codes.add(s.participant_code)

    def test_rejected_movie_id(self):
        from .models import PairwiseTrial, StudySession
        session = StudySession.objects.create(condition_order="AB", seed=1)
        trial = PairwiseTrial.objects.create(
            session=session, task_index=0,
            left_movie_id=10, right_movie_id=20, chosen_movie_id=10,
        )
        self.assertEqual(trial.rejected_movie_id, 20)
        trial.chosen_movie_id = 20
        self.assertEqual(trial.rejected_movie_id, 10)

    def test_session_not_complete_initially(self):
        from .models import StudySession
        session = StudySession.objects.create(condition_order="AB", seed=1)
        self.assertFalse(session.is_complete)


class StudyFlowViewTest(TestCase):
    def _start(self):
        return self.client.post("/project4/study/start/")

    def test_landing_has_start_button(self):
        response = self.client.get("/project4/")
        self.assertContains(response, "/project4/study/start/")
        self.assertContains(response, "Start the study")

    def test_start_creates_session_and_redirects(self):
        from .models import StudySession
        response = self._start()
        self.assertEqual(StudySession.objects.count(), 1)
        self.assertRedirects(response, "/project4/study/", target_status_code=302)

    def test_start_requires_post(self):
        from .models import StudySession
        response = self.client.get("/project4/study/start/")
        self.assertRedirects(response, "/project4/")
        self.assertEqual(StudySession.objects.count(), 0)

    def test_fresh_session_starts_at_consent(self):
        self._start()
        response = self.client.get("/project4/study/")
        self.assertRedirects(response, "/project4/study/consent/")

    def test_study_without_session_returns_to_landing(self):
        response = self.client.get("/project4/study/")
        self.assertRedirects(response, "/project4/")

    def test_consent_page_covers_ethics_points(self):
        self._start()
        response = self.client.get("/project4/study/consent/")
        self.assertContains(response, "withdraw")
        self.assertContains(response, "random code")
        self.assertContains(response, "consent")

    def test_cannot_skip_ahead_to_a_later_step(self):
        self._start()
        # Still on consent; jumping to background must bounce back to the router.
        response = self.client.get("/project4/study/background/")
        self.assertRedirects(response, "/project4/study/", target_status_code=302)

    def test_consent_advances_and_records(self):
        from .models import StudySession
        self._start()
        self.client.post("/project4/study/consent/", {"consent": "on"})
        session = StudySession.objects.first()
        self.assertTrue(session.consented)
        self.assertEqual(session.current_step, "background")

    def test_background_stores_answers(self):
        from .models import QuestionnaireResponse, StudySession
        self._start()
        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34",
            "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat",
        })
        answer = QuestionnaireResponse.objects.get(
            kind=QuestionnaireResponse.KIND_BACKGROUND)
        self.assertEqual(answer.answers["age_range"], "25-34")
        self.assertEqual(answer.answers["movie_frequency"], "weekly")
        self.assertEqual(StudySession.objects.first().current_step, "instructions")

    def test_instructions_explain_both_interfaces(self):
        self._start()
        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34", "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat"})
        response = self.client.get("/project4/study/instructions/")
        self.assertContains(response, "two films")
        self.assertContains(response, "Ranking ten films")
        self.assertContains(response, "practice")

    def test_flow_reaches_practice_step(self):
        from .models import StudySession
        self._start()
        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34", "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat"})
        self.client.post("/project4/study/instructions/")
        self.assertEqual(StudySession.objects.first().current_step, "practice")

    def test_progress_indicator_present(self):
        self._start()
        response = self.client.get("/project4/study/consent/")
        self.assertContains(response, "p4-progress")
        self.assertContains(response, "Step 1 of")

    def test_sessions_alternate_condition_order(self):
        from .models import StudySession
        self._start()
        self.client.session.flush()
        self.client.cookies.clear()
        self._start()
        orders = list(StudySession.objects.order_by("pk").values_list(
            "condition_order", flat=True))
        self.assertEqual(orders, ["AB", "BA"])


# ── Stage P4-5: pairwise condition ─────────────────────────────────────────

class PairwiseConditionTest(TestCase):
    """Drives a real session up to and through the pairwise condition."""

    def _reach_practice(self):
        """Start a session and walk to the practice step."""
        self.client.post("/project4/study/start/")
        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34", "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat"})
        self.client.post("/project4/study/instructions/")

    def _session(self):
        from .models import StudySession
        return StudySession.objects.first()

    def _force_pairwise_first(self):
        """Pin the session to order AB so condition_1 is the pairwise one."""
        session = self._session()
        session.condition_order = "AB"
        session.save(update_fields=["condition_order"])
        return session

    def _plan(self):
        from .services.sampling import DEFAULT_CONFIG, build_trial_plan
        from .services.data import get_movie_corpus
        return build_trial_plan(
            self._session().seed, get_movie_corpus().n_movies, DEFAULT_CONFIG)

    def _complete_practice(self):
        """Finish both practice blocks, in whichever order this session uses."""
        from .services.study import conditions_for_order
        plan = self._plan()
        posts = {
            "pairwise": {"chosen_movie_id": str(plan.practice_pairwise[0][0])},
            "ranking": {"ranked_movie_ids":
                        ",".join(str(m) for m in plan.practice_ranking[0])},
        }
        for which in conditions_for_order(self._session().condition_order):
            self.client.post("/project4/study/practice/", posts[which])
        # Follow the redirect chain: the router re-enters the practice view,
        # which sees both blocks are finished and advances the step.
        self.client.get("/project4/study/", follow=True)

    # ── practice ────────────────────────────────────────────────────────────

    def test_practice_shows_two_movies(self):
        self._reach_practice()
        response = self.client.get("/project4/study/practice/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode().count("p4-movie-card"), 2)

    def test_practice_is_labelled_as_practice(self):
        self._reach_practice()
        response = self.client.get("/project4/study/practice/")
        self.assertContains(response, "does not count")

    def test_practice_choice_is_stored_in_practice_block(self):
        from .models import BLOCK_PRACTICE, PairwiseTrial
        self._reach_practice()
        left, right = self._plan().practice_pairwise[0]
        self.client.post("/project4/study/practice/", {
            "chosen_movie_id": str(left), "response_time_ms": "1500"})
        trial = PairwiseTrial.objects.get(block=BLOCK_PRACTICE)
        self.assertEqual(trial.chosen_movie_id, left)
        self.assertEqual(trial.rejected_movie_id, right)
        self.assertEqual(trial.response_time_ms, 1500)

    def test_practice_advances_to_first_condition(self):
        self._reach_practice()
        self._complete_practice()
        self.assertEqual(self._session().current_step, "condition_1")

    def test_practice_does_not_advance_until_both_interfaces_are_practised(self):
        """One practice block done is not enough — the other still has to run."""
        self._reach_practice()
        self._force_pairwise_first()
        left, _ = self._plan().practice_pairwise[0]
        self.client.post("/project4/study/practice/", {"chosen_movie_id": str(left)})
        self.client.get("/project4/study/", follow=True)
        self.assertEqual(self._session().current_step, "practice")

    # ── main pairwise condition ─────────────────────────────────────────────

    def _enter_pairwise_condition(self):
        self._reach_practice()
        self._force_pairwise_first()
        self._complete_practice()

    def test_condition_shows_pairwise_trial(self):
        self._enter_pairwise_condition()
        response = self.client.get("/project4/study/condition/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Which would you rather watch?")
        self.assertContains(response, "Task 1 of")

    def test_trial_uses_the_planned_pair(self):
        self._enter_pairwise_condition()
        left, right = self._plan().pairwise[0]
        from .services.data import get_movie_corpus
        corpus = get_movie_corpus()
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, f'value="{left}"')
        self.assertContains(response, f'value="{right}"')

    def test_choice_is_recorded_and_trial_advances(self):
        from .models import BLOCK_MAIN, PairwiseTrial
        self._enter_pairwise_condition()
        left, _ = self._plan().pairwise[0]
        self.client.post("/project4/study/condition/", {"chosen_movie_id": str(left)})
        self.assertEqual(PairwiseTrial.objects.filter(block=BLOCK_MAIN).count(), 1)
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, "Task 2 of")

    def test_invalid_choice_is_rejected(self):
        """A movie that is not in the displayed pair must not be recorded."""
        from .models import BLOCK_MAIN, PairwiseTrial
        self._enter_pairwise_condition()
        self.client.post("/project4/study/condition/", {"chosen_movie_id": "999999"})
        self.assertEqual(PairwiseTrial.objects.filter(block=BLOCK_MAIN).count(), 0)

    def test_missing_choice_is_rejected(self):
        from .models import BLOCK_MAIN, PairwiseTrial
        self._enter_pairwise_condition()
        self.client.post("/project4/study/condition/", {})
        self.assertEqual(PairwiseTrial.objects.filter(block=BLOCK_MAIN).count(), 0)

    def test_malformed_response_time_is_ignored_but_choice_kept(self):
        from .models import BLOCK_MAIN, PairwiseTrial
        self._enter_pairwise_condition()
        left, _ = self._plan().pairwise[0]
        self.client.post("/project4/study/condition/", {
            "chosen_movie_id": str(left), "response_time_ms": "not-a-number"})
        trial = PairwiseTrial.objects.get(block=BLOCK_MAIN)
        self.assertEqual(trial.chosen_movie_id, left)
        self.assertIsNone(trial.response_time_ms)

    def test_completing_all_trials_advances_the_step(self):
        from .models import BLOCK_MAIN, PairwiseTrial
        from .services.sampling import DEFAULT_CONFIG
        self._enter_pairwise_condition()
        plan = self._plan()
        for left, _ in plan.pairwise:
            self.client.post("/project4/study/condition/", {"chosen_movie_id": str(left)})
        self.assertEqual(
            PairwiseTrial.objects.filter(block=BLOCK_MAIN).count(),
            DEFAULT_CONFIG.n_pairwise_trials,
        )
        self.client.get("/project4/study/condition/", follow=True)
        self.assertEqual(self._session().current_step, "questionnaire_1")

    def test_reload_does_not_duplicate_or_skip(self):
        """Refreshing the trial page must show the same trial again."""
        self._enter_pairwise_condition()
        first = self.client.get("/project4/study/condition/").content.decode()
        second = self.client.get("/project4/study/condition/").content.decode()
        self.assertIn("Task 1 of", first)
        self.assertIn("Task 1 of", second)

    def test_pairwise_trials_use_disjoint_movies_from_practice(self):
        plan_pairs = None
        self._enter_pairwise_condition()
        plan = self._plan()
        practice_ids = {m for pair in plan.practice_pairwise for m in pair}
        main_ids = {m for pair in plan.pairwise for m in pair}
        self.assertEqual(practice_ids & main_ids, set())

    def test_condition_step_guard(self):
        """Hitting the condition URL outside a condition step redirects away."""
        self._reach_practice()      # still on the practice step
        response = self.client.get("/project4/study/condition/")
        self.assertRedirects(response, "/project4/study/", target_status_code=302)

    def test_timer_script_loaded(self):
        self._enter_pairwise_condition()
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, "trial_timer.js")


# ── Stage P4-6: ranking condition ──────────────────────────────────────────

class RankingOrderHelperTest(TestCase):
    """The two pure helpers behind the ranking engine, tested in isolation."""

    def setUp(self):
        from django.test import RequestFactory
        self.factory = RequestFactory()

    def _parse(self, posted, shown):
        from .views import _parse_submitted_order
        request = self.factory.post("/", {"ranked_movie_ids": posted})
        return _parse_submitted_order(request, shown)

    def test_accepts_a_permutation_of_the_shown_movies(self):
        self.assertEqual(self._parse("3,1,2", [1, 2, 3]), [3, 1, 2])

    def test_accepts_the_unchanged_order(self):
        self.assertEqual(self._parse("1,2,3", [1, 2, 3]), [1, 2, 3])

    def test_rejects_a_duplicate(self):
        self.assertIsNone(self._parse("1,1,2", [1, 2, 3]))

    def test_rejects_a_missing_movie(self):
        self.assertIsNone(self._parse("1,2", [1, 2, 3]))

    def test_rejects_an_extra_movie(self):
        self.assertIsNone(self._parse("1,2,3,4", [1, 2, 3]))

    def test_rejects_a_movie_that_was_never_shown(self):
        self.assertIsNone(self._parse("1,2,99", [1, 2, 3]))

    def test_rejects_non_numeric_input(self):
        self.assertIsNone(self._parse("1,two,3", [1, 2, 3]))

    def test_rejects_empty_input(self):
        self.assertIsNone(self._parse("", [1, 2, 3]))

    # ── the no-JavaScript move fallback ─────────────────────────────────────

    def test_move_up_swaps_with_the_row_above(self):
        from .views import _apply_move
        self.assertEqual(_apply_move([1, 2, 3], "up:2"), [2, 1, 3])

    def test_move_down_swaps_with_the_row_below(self):
        from .views import _apply_move
        self.assertEqual(_apply_move([1, 2, 3], "down:2"), [1, 3, 2])

    def test_move_off_either_end_is_ignored(self):
        from .views import _apply_move
        self.assertEqual(_apply_move([1, 2, 3], "up:1"), [1, 2, 3])
        self.assertEqual(_apply_move([1, 2, 3], "down:3"), [1, 2, 3])

    def test_malformed_move_is_ignored(self):
        from .views import _apply_move
        for move in ["up:x", "sideways:2", "up:99", "garbage", ""]:
            self.assertEqual(_apply_move([1, 2, 3], move), [1, 2, 3], move)

    def test_move_never_drops_or_duplicates_a_movie(self):
        from .views import _apply_move
        order = [5, 6, 7, 8]
        for move in ["up:7", "down:5", "up:8", "down:6"]:
            order = _apply_move(order, move)
            self.assertEqual(sorted(order), [5, 6, 7, 8])


class RankingConditionTest(TestCase):
    """Drives a real session through the ranking practice and condition."""

    def _reach_practice(self, order="BA"):
        self.client.post("/project4/study/start/")
        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34", "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat"})
        self.client.post("/project4/study/instructions/")
        session = self._session()
        session.condition_order = order       # BA runs ranking first
        session.save(update_fields=["condition_order"])

    def _session(self):
        from .models import StudySession
        return StudySession.objects.first()

    def _plan(self):
        from .services.sampling import DEFAULT_CONFIG, build_trial_plan
        from .services.data import get_movie_corpus
        return build_trial_plan(
            self._session().seed, get_movie_corpus().n_movies, DEFAULT_CONFIG)

    def _enter_ranking_condition(self):
        """Order BA: practice ranking, then practice pairwise, then condition 1."""
        self._reach_practice(order="BA")
        plan = self._plan()
        self.client.post("/project4/study/practice/", {
            "ranked_movie_ids": ",".join(str(m) for m in plan.practice_ranking[0])})
        self.client.post("/project4/study/practice/", {
            "chosen_movie_id": str(plan.practice_pairwise[0][0])})
        self.client.get("/project4/study/", follow=True)

    # ── practice ────────────────────────────────────────────────────────────

    def test_ranking_practice_shows_ten_movies(self):
        self._reach_practice(order="BA")
        response = self.client.get("/project4/study/practice/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode().count("p4-rank-item"), 10)

    def test_ranking_practice_is_labelled_as_practice(self):
        self._reach_practice(order="BA")
        response = self.client.get("/project4/study/practice/")
        self.assertContains(response, "does not count")

    def test_ranking_practice_is_stored_in_the_practice_block(self):
        from .models import BLOCK_PRACTICE, RankingTrial
        self._reach_practice(order="BA")
        group = self._plan().practice_ranking[0]
        self.client.post("/project4/study/practice/", {
            "ranked_movie_ids": ",".join(str(m) for m in reversed(group)),
            "response_time_ms": "9000"})
        trial = RankingTrial.objects.get(block=BLOCK_PRACTICE)
        self.assertEqual(trial.movie_ids, list(group))
        self.assertEqual(trial.ranked_movie_ids, list(reversed(group)))
        self.assertEqual(trial.response_time_ms, 9000)

    # ── main ranking condition ──────────────────────────────────────────────

    def test_condition_shows_a_ranking_trial(self):
        self._enter_ranking_condition()
        response = self.client.get("/project4/study/condition/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Put these films in order")
        self.assertContains(response, "Ranking 1 of 3")

    def test_condition_shows_the_planned_group(self):
        self._enter_ranking_condition()
        response = self.client.get("/project4/study/condition/")
        body = response.content.decode()
        for movie_id in self._plan().ranking[0]:
            self.assertIn(f'data-movie-id="{movie_id}"', body)

    def test_submitted_ranking_is_recorded_and_the_trial_advances(self):
        from .models import BLOCK_MAIN, RankingTrial
        self._enter_ranking_condition()
        group = self._plan().ranking[0]
        submitted = list(reversed(group))
        self.client.post("/project4/study/condition/", {
            "ranked_movie_ids": ",".join(str(m) for m in submitted),
            "response_time_ms": "12000"})

        trial = RankingTrial.objects.get(block=BLOCK_MAIN, task_index=0)
        self.assertEqual(trial.ranked_movie_ids, submitted)
        self.assertEqual(trial.response_time_ms, 12000)
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, "Ranking 2 of 3")

    def test_a_tampered_order_is_not_stored(self):
        from .models import RankingTrial
        self._enter_ranking_condition()
        group = self._plan().ranking[0]
        self.client.post("/project4/study/condition/", {
            "ranked_movie_ids": ",".join(str(m) for m in group[:5])})
        self.assertEqual(RankingTrial.objects.filter(block="main").count(), 0)
        # ...and the participant is shown the very same task again.
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, "Ranking 1 of 3")

    def test_a_foreign_movie_id_is_not_stored(self):
        from .models import RankingTrial
        self._enter_ranking_condition()
        group = list(self._plan().ranking[0])
        group[0] = 99999                     # not one of the films shown
        self.client.post("/project4/study/condition/", {
            "ranked_movie_ids": ",".join(str(m) for m in group)})
        self.assertEqual(RankingTrial.objects.filter(block="main").count(), 0)

    def test_move_button_reorders_without_recording_a_trial(self):
        """The no-JavaScript fallback: one press swaps two rows, stores nothing."""
        from .models import RankingTrial
        self._enter_ranking_condition()
        group = self._plan().ranking[0]
        response = self.client.post("/project4/study/condition/", {
            "ranked_movie_ids": ",".join(str(m) for m in group),
            "move": f"up:{group[1]}"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RankingTrial.objects.filter(block="main").count(), 0)
        expected = [group[1], group[0]] + list(group[2:])
        self.assertContains(
            response, ",".join(str(m) for m in expected))

    def test_reload_shows_the_same_ranking_trial(self):
        self._enter_ranking_condition()
        first = self.client.get("/project4/study/condition/").content.decode()
        second = self.client.get("/project4/study/condition/").content.decode()
        self.assertIn("Ranking 1 of 3", first)
        self.assertIn("Ranking 1 of 3", second)

    def test_completing_all_rankings_advances_the_step(self):
        self._enter_ranking_condition()
        plan = self._plan()
        for group in plan.ranking:
            self.client.post("/project4/study/condition/", {
                "ranked_movie_ids": ",".join(str(m) for m in group)})
        self.client.get("/project4/study/", follow=True)
        self.assertEqual(self._session().current_step, "questionnaire_1")

    def test_ranking_and_pairwise_movies_are_disjoint(self):
        """No film appears in both conditions, so neither can prime the other."""
        self._enter_ranking_condition()
        plan = self._plan()
        ranking_ids = {m for group in plan.ranking for m in group}
        pairwise_ids = {m for pair in plan.pairwise for m in pair}
        self.assertEqual(ranking_ids & pairwise_ids, set())

    def test_ranking_scripts_loaded(self):
        self._enter_ranking_condition()
        response = self.client.get("/project4/study/condition/")
        self.assertContains(response, "ranking.js")
        self.assertContains(response, "trial_timer.js")

    def test_move_buttons_are_labelled_for_screen_readers(self):
        self._enter_ranking_condition()
        response = self.client.get("/project4/study/condition/")
        body = response.content.decode()
        self.assertEqual(body.count('aria-label="Move '), 20)   # up+down per film
        self.assertIn('aria-live="polite"', body)


# ── Stage P4-7: questionnaires, evaluation and debrief ─────────────────────

class AnalysisServiceTest(TestCase):
    """Fitting a session's two models and reading the result back."""

    def _session(self, step="heldout"):
        from .models import StudySession
        return StudySession.objects.create(
            condition_order="AB", seed=7, current_step=step)

    def _add_pairwise(self, session, block, pairs):
        from .models import PairwiseTrial
        for index, (chosen, rejected) in enumerate(pairs):
            PairwiseTrial.objects.create(
                session=session, block=block, task_index=index,
                left_movie_id=chosen, right_movie_id=rejected,
                chosen_movie_id=chosen)

    def _add_ranking(self, session, block, groups):
        from .models import RankingTrial
        for index, group in enumerate(groups):
            RankingTrial.objects.create(
                session=session, block=block, task_index=index,
                movie_ids=sorted(group), ranked_movie_ids=list(group))

    # ── reading the elicited data ───────────────────────────────────────────

    def test_pairwise_pairs_are_chosen_then_rejected(self):
        """The model's convention is (chosen, rejected), so the order matters."""
        from .services.analysis import pairwise_pairs
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20), (30, 40)])
        self.assertEqual(pairwise_pairs(session), [(10, 20), (30, 40)])

    def test_rankings_are_read_best_first(self):
        from .services.analysis import rankings
        session = self._session()
        self._add_ranking(session, "main", [[3, 1, 2]])
        self.assertEqual(rankings(session), [[3, 1, 2]])

    # ── fitting ─────────────────────────────────────────────────────────────

    def test_fit_creates_one_row_per_condition(self):
        from .services.analysis import fit_session_models
        from .services.features import get_features
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20), (30, 40)])
        self._add_ranking(session, "main", [[1, 2, 3]])

        fits = fit_session_models(session)
        _, encoder = get_features()
        self.assertEqual(set(fits), {"pairwise", "ranking"})
        for condition, fit in fits.items():
            self.assertEqual(len(fit.weights), encoder.dim, condition)

    def test_fit_records_how_much_data_each_condition_gave(self):
        from .services.analysis import fit_session_models
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20), (30, 40)])
        self._add_ranking(session, "main", [[1, 2, 3]])
        fits = fit_session_models(session)
        self.assertEqual(fits["pairwise"].n_observations, 2)
        self.assertEqual(fits["ranking"].n_observations, 1)

    def test_fitting_twice_replaces_rather_than_duplicates(self):
        from .models import PreferenceModelFit
        from .services.analysis import fit_session_models
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20)])
        fit_session_models(session)
        fit_session_models(session)
        self.assertEqual(PreferenceModelFit.objects.filter(session=session).count(), 2)

    def test_both_models_are_scored_on_the_same_heldout_pairs(self):
        from .services.analysis import fit_session_models
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20), (30, 40)])
        self._add_ranking(session, "main", [[1, 2, 3]])
        self._add_pairwise(session, "heldout", [(50, 60), (70, 80)])

        fits = fit_session_models(session)
        for condition, fit in fits.items():
            self.assertIsNotNone(fit.heldout_accuracy, condition)
            self.assertTrue(0.0 <= fit.heldout_accuracy <= 1.0, condition)
            self.assertGreater(fit.heldout_log_loss, 0.0, condition)

    def test_no_heldout_data_means_no_score_rather_than_a_fake_one(self):
        from .services.analysis import fit_session_models
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20)])
        fits = fit_session_models(session)
        self.assertIsNone(fits["pairwise"].heldout_accuracy)
        self.assertIsNone(fits["pairwise"].heldout_log_loss)

    def test_a_condition_with_no_data_still_fits_a_neutral_model(self):
        """A participant who gave nothing gets w = 0, not a crash."""
        from .services.analysis import fit_session_models
        session = self._session()
        self._add_pairwise(session, "main", [(10, 20)])
        fits = fit_session_models(session)
        self.assertEqual(fits["ranking"].n_observations, 0)
        self.assertEqual(set(fits["ranking"].weights), {0.0})

    # ── interpreting w ──────────────────────────────────────────────────────

    def test_feature_names_are_humanized(self):
        from .services.analysis import humanize_feature
        self.assertEqual(humanize_feature("genre:Sci-Fi"), "Sci-Fi films")
        self.assertEqual(humanize_feature("num:duration"), "longer films")
        self.assertEqual(humanize_feature("num:title_year"), "more recent films")
        self.assertEqual(humanize_feature("country:USA"), "films made in the USA")
        self.assertEqual(humanize_feature("rating:Family"), "family-friendly films")

    def test_top_features_ranks_by_signed_weight(self):
        import numpy as np
        from .services.analysis import top_features
        from .services.features import get_features
        _, encoder = get_features()
        w = np.zeros(encoder.dim)
        w[0], w[1] = 2.0, 1.0
        w[2], w[3] = -3.0, -0.5

        top = top_features(w, encoder, count=2)
        self.assertEqual([f["name"] for f in top["liked"]],
                         [encoder.feature_names[0], encoder.feature_names[1]])
        self.assertEqual([f["name"] for f in top["disliked"]],
                         [encoder.feature_names[2], encoder.feature_names[3]])

    def test_top_features_drops_weights_that_say_nothing(self):
        """A model that never moved a feature should not claim an opinion on it."""
        import numpy as np
        from .services.analysis import top_features
        from .services.features import get_features
        _, encoder = get_features()
        w = np.zeros(encoder.dim)
        w[0] = 1.5
        top = top_features(w, encoder, count=5)
        self.assertEqual(len(top["liked"]), 1)
        self.assertEqual(top["disliked"], [])

    # ── the debrief summary ─────────────────────────────────────────────────

    def _summary_with(self, pairwise, ranking):
        """`pairwise`/`ranking` are (accuracy, log_loss) for that condition."""
        from .models import PreferenceModelFit
        from .services.analysis import session_summary
        from .services.features import get_features
        _, encoder = get_features()
        session = self._session(step="complete")
        for condition, (accuracy, log_loss) in (("pairwise", pairwise),
                                                ("ranking", ranking)):
            PreferenceModelFit.objects.create(
                session=session, condition=condition,
                weights=[0.0] * encoder.dim, n_observations=5,
                heldout_accuracy=accuracy, heldout_log_loss=log_loss)
        return session_summary(session)

    def test_summary_names_the_better_method(self):
        summary = self._summary_with((0.9, 0.5), (0.6, 0.4))
        self.assertEqual(summary["best"]["condition"], "pairwise")
        self.assertFalse(summary["tied"])
        # Accuracy decided it, so log loss never came into play.
        self.assertFalse(summary["decided_on_log_loss"])

    def test_equal_accuracy_is_broken_on_log_loss(self):
        """Ten pairs give only eleven possible scores, so ties are common."""
        summary = self._summary_with((0.7, 0.705), (0.7, 0.569))
        self.assertEqual(summary["best"]["condition"], "ranking")
        self.assertTrue(summary["decided_on_log_loss"])
        self.assertFalse(summary["tied"])

    def test_summary_reports_a_tie_rather_than_picking_one(self):
        summary = self._summary_with((0.7, 0.5), (0.7, 0.5))
        self.assertTrue(summary["tied"])
        self.assertIsNone(summary["best"])
        self.assertFalse(summary["decided_on_log_loss"])

    def test_summary_converts_accuracy_to_a_percentage(self):
        summary = self._summary_with((0.9, 0.5), (0.6, 0.4))
        percents = {m["condition"]: m["accuracy_percent"] for m in summary["methods"]}
        self.assertEqual(percents, {"pairwise": 90, "ranking": 60})


class ClosingStepViewTest(TestCase):
    """The questionnaire, break, held-out and debrief pages, in isolation."""

    QUESTIONNAIRE_ANSWERS = {
        "ease": "6", "effort": "3", "expressive": "5",
        "confidence": "5", "willing": "4", "comment": "the ranking took longer",
    }

    FINAL_ANSWERS = {
        "preferred": "pairwise", "expressive": "ranking",
        "effort": "pairwise", "comment": "",
    }

    def _session(self):
        from .models import StudySession
        return StudySession.objects.first()

    def _start(self, order="AB"):
        self.client.post("/project4/study/start/")
        session = self._session()
        session.condition_order = order
        session.save(update_fields=["condition_order"])
        return session

    def _jump(self, step):
        session = self._session()
        session.current_step = step
        session.save(update_fields=["current_step"])
        return session

    def _plan(self):
        from .services.sampling import DEFAULT_CONFIG, build_trial_plan
        from .services.data import get_movie_corpus
        return build_trial_plan(
            self._session().seed, get_movie_corpus().n_movies, DEFAULT_CONFIG)

    # ── per-condition questionnaire ─────────────────────────────────────────

    def test_first_questionnaire_asks_about_the_first_condition(self):
        self._start(order="AB")
        self._jump("questionnaire_1")
        response = self.client.get("/project4/study/questionnaire/")
        self.assertContains(response, "choosing between two films")

    def test_second_questionnaire_asks_about_the_second_condition(self):
        self._start(order="AB")
        self._jump("questionnaire_2")
        response = self.client.get("/project4/study/questionnaire/")
        self.assertContains(response, "ranking ten films")

    def test_questionnaire_follows_the_counterbalancing(self):
        """Order BA did ranking first, so its first questionnaire must ask that."""
        self._start(order="BA")
        self._jump("questionnaire_1")
        response = self.client.get("/project4/study/questionnaire/")
        self.assertContains(response, "ranking ten films")

    def test_questionnaire_offers_a_full_likert_scale(self):
        self._start()
        self._jump("questionnaire_1")
        response = self.client.get("/project4/study/questionnaire/")
        self.assertEqual(response.content.decode().count('type="radio"'), 35)

    def test_questionnaire_answers_are_stored_against_the_condition(self):
        from .models import QuestionnaireResponse
        self._start(order="AB")
        self._jump("questionnaire_1")
        self.client.post("/project4/study/questionnaire/", self.QUESTIONNAIRE_ANSWERS)

        answer = QuestionnaireResponse.objects.get(kind="condition")
        self.assertEqual(answer.condition, "pairwise")
        self.assertEqual(answer.answers["ease"], "6")
        self.assertEqual(answer.answers["comment"], "the ranking took longer")

    def test_questionnaire_advances_the_step(self):
        self._start()
        self._jump("questionnaire_1")
        self.client.post("/project4/study/questionnaire/", self.QUESTIONNAIRE_ANSWERS)
        self.assertEqual(self._session().current_step, "break")

    def test_questionnaire_url_is_guarded(self):
        self._start()
        self._jump("break")
        response = self.client.get("/project4/study/questionnaire/")
        self.assertRedirects(response, "/project4/study/", target_status_code=302)

    # ── break ───────────────────────────────────────────────────────────────

    def test_break_previews_the_interface_that_comes_next(self):
        self._start(order="AB")          # pairwise done, ranking next
        self._jump("break")
        response = self.client.get("/project4/study/break/")
        self.assertContains(response, "ten-film ranking")

    def test_break_preview_follows_the_counterbalancing(self):
        self._start(order="BA")          # ranking done, pairwise next
        self._jump("break")
        response = self.client.get("/project4/study/break/")
        self.assertContains(response, "two-film choice")

    def test_break_advances_to_the_second_condition(self):
        self._start()
        self._jump("break")
        self.client.post("/project4/study/break/")
        self.assertEqual(self._session().current_step, "condition_2")

    # ── held-out preference check ───────────────────────────────────────────

    def test_heldout_explains_why_it_is_being_asked(self):
        self._start()
        self._jump("heldout")
        response = self.client.get("/project4/study/check/")
        self.assertContains(response, "preference check")
        self.assertContains(response, "Task 1 of 10")

    def test_heldout_uses_the_planned_pairs(self):
        self._start()
        self._jump("heldout")
        left, right = self._plan().heldout[0]
        response = self.client.get("/project4/study/check/")
        body = response.content.decode()
        self.assertIn(f'value="{left}"', body)
        self.assertIn(f'value="{right}"', body)

    def test_heldout_choices_are_stored_in_their_own_block(self):
        from .models import PairwiseTrial
        self._start()
        self._jump("heldout")
        left, _ = self._plan().heldout[0]
        self.client.post("/project4/study/check/", {"chosen_movie_id": str(left)})
        self.assertEqual(PairwiseTrial.objects.filter(block="heldout").count(), 1)

    def test_heldout_movies_were_never_shown_in_either_condition(self):
        """The evaluation must test generalization, not recall."""
        self._start()
        plan = self._plan()
        heldout_ids = {m for pair in plan.heldout for m in pair}
        shown_ids = {m for pair in plan.pairwise for m in pair}
        shown_ids |= {m for group in plan.ranking for m in group}
        shown_ids |= {m for pair in plan.practice_pairwise for m in pair}
        shown_ids |= {m for group in plan.practice_ranking for m in group}
        self.assertEqual(heldout_ids & shown_ids, set())

    def test_completing_the_check_fits_both_models(self):
        from .models import PreferenceModelFit
        self._start()
        self._jump("heldout")
        for left, _ in self._plan().heldout:
            self.client.post("/project4/study/check/", {"chosen_movie_id": str(left)})
        self.client.get("/project4/study/", follow=True)

        self.assertEqual(self._session().current_step, "final")
        self.assertEqual(
            set(PreferenceModelFit.objects.values_list("condition", flat=True)),
            {"pairwise", "ranking"})

    # ── final comparison ────────────────────────────────────────────────────

    def test_final_names_the_interfaces_not_their_order(self):
        """Which came first differs per participant, so the labels must not say."""
        self._start()
        self._jump("final")
        response = self.client.get("/project4/study/final/")
        self.assertContains(response, "Choosing between two films")
        self.assertContains(response, "Ranking ten films")
        self.assertNotContains(response, "the first interface")

    def test_final_answers_are_stored_and_the_session_is_marked_complete(self):
        from .models import QuestionnaireResponse
        self._start()
        self._jump("final")
        self.client.post("/project4/study/final/", self.FINAL_ANSWERS)

        answer = QuestionnaireResponse.objects.get(kind="final")
        self.assertEqual(answer.answers["preferred"], "pairwise")
        session = self._session()
        self.assertEqual(session.current_step, "complete")
        self.assertIsNotNone(session.completed_at)
        self.assertTrue(session.is_complete)

    # ── debrief ─────────────────────────────────────────────────────────────

    def test_debrief_is_not_reachable_before_the_end(self):
        self._start()
        self._jump("final")
        response = self.client.get("/project4/study/complete/")
        self.assertRedirects(response, "/project4/study/", target_status_code=302)

    def test_debrief_shows_the_participant_code_and_both_methods(self):
        self._start()
        self._jump("complete")
        response = self.client.get("/project4/study/complete/")
        self.assertContains(response, self._session().participant_code)
        self.assertContains(response, "Choosing between two films")
        self.assertContains(response, "Ranking ten films")

    def test_debrief_explains_how_to_withdraw(self):
        self._start()
        self._jump("complete")
        response = self.client.get("/project4/study/complete/")
        self.assertContains(response, "removed")


class FullStudyWalkthroughTest(TestCase):
    """One participant, start to finish, through all twelve steps."""

    def _session(self):
        from .models import StudySession
        return StudySession.objects.first()

    def _plan(self):
        from .services.sampling import DEFAULT_CONFIG, build_trial_plan
        from .services.data import get_movie_corpus
        return build_trial_plan(
            self._session().seed, get_movie_corpus().n_movies, DEFAULT_CONFIG)

    def _pairwise_block(self, url, pairs):
        for chosen, _ in pairs:
            self.client.post(url, {"chosen_movie_id": str(chosen),
                                   "response_time_ms": "1400"})

    def _ranking_block(self, url, groups):
        for group in groups:
            self.client.post(url, {
                "ranked_movie_ids": ",".join(str(m) for m in group),
                "response_time_ms": "11000"})

    def _run(self, which, plan, url, practice=False):
        if which == "pairwise":
            self._pairwise_block(
                url, plan.practice_pairwise if practice else plan.pairwise)
        else:
            self._ranking_block(
                url, plan.practice_ranking if practice else plan.ranking)

    def _walk(self, order="AB"):
        from .services.study import conditions_for_order
        self.client.post("/project4/study/start/")
        session = self._session()
        session.condition_order = order
        session.save(update_fields=["condition_order"])

        self.client.post("/project4/study/consent/", {"consent": "on"})
        self.client.post("/project4/study/background/", {
            "age_range": "25-34", "movie_frequency": "weekly",
            "recommender_familiarity": "somewhat"})
        self.client.post("/project4/study/instructions/")

        plan = self._plan()
        conditions = conditions_for_order(order)

        for which in conditions:
            self._run(which, plan, "/project4/study/practice/", practice=True)
        self.client.get("/project4/study/", follow=True)

        answers = {"ease": "5", "effort": "4", "expressive": "5",
                   "confidence": "5", "willing": "5", "comment": ""}
        for index, which in enumerate(conditions):
            self._run(which, plan, "/project4/study/condition/")
            self.client.get("/project4/study/", follow=True)
            self.client.post("/project4/study/questionnaire/", answers)
            if index == 0:
                self.client.post("/project4/study/break/")

        self._pairwise_block("/project4/study/check/", plan.heldout)
        self.client.get("/project4/study/", follow=True)
        self.client.post("/project4/study/final/", {
            "preferred": "ranking", "expressive": "ranking",
            "effort": "pairwise", "comment": "interesting study"})
        return plan

    def test_a_full_run_reaches_the_debrief(self):
        self._walk()
        session = self._session()
        self.assertEqual(session.current_step, "complete")
        self.assertTrue(session.is_complete)
        response = self.client.get("/project4/study/complete/")
        self.assertEqual(response.status_code, 200)

    def test_a_full_run_collects_the_planned_amount_of_data(self):
        from .models import PairwiseTrial, QuestionnaireResponse, RankingTrial
        from .services.sampling import DEFAULT_CONFIG as cfg
        self._walk()
        session = self._session()

        self.assertEqual(
            PairwiseTrial.objects.filter(session=session, block="main").count(),
            cfg.n_pairwise_trials)
        self.assertEqual(
            RankingTrial.objects.filter(session=session, block="main").count(),
            cfg.n_ranking_trials)
        self.assertEqual(
            PairwiseTrial.objects.filter(session=session, block="heldout").count(),
            cfg.n_heldout_pairs)
        self.assertEqual(
            QuestionnaireResponse.objects.filter(session=session).count(), 4)

    def test_a_full_run_in_the_reverse_order_works_too(self):
        """Counterbalancing must not be a path only one group can walk."""
        self._walk(order="BA")
        self.assertEqual(self._session().current_step, "complete")

    def test_the_debrief_reports_a_real_score_for_both_methods(self):
        self._walk()
        response = self.client.get("/project4/study/complete/")
        summary = response.context["summary"]
        self.assertEqual(len(summary["methods"]), 2)
        for method in summary["methods"]:
            self.assertIsNotNone(method["accuracy"], method["condition"])
            self.assertTrue(0.0 <= method["accuracy"] <= 1.0)
        self.assertEqual(summary["n_heldout"], 10)

    def test_every_ranking_stored_is_a_permutation_of_what_was_shown(self):
        from .models import RankingTrial
        self._walk()
        for trial in RankingTrial.objects.all():
            self.assertEqual(sorted(trial.movie_ids), sorted(trial.ranked_movie_ids))
            self.assertEqual(len(trial.ranked_movie_ids), 10)

    def test_no_movie_is_ever_shown_to_a_participant_twice(self):
        """Every trial the participant saw, across every block, is distinct."""
        plan = self._walk()
        ids = plan.all_movie_ids
        self.assertEqual(len(ids), len(set(ids)))
