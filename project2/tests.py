from django.test import TestCase

from .services.data import (
    get_penguin_data, load_clean_penguins, compute_mad,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
    INPUT_FEATURES, TARGET, SPECIES_ORDER, MAD_EPSILON,
)


# ── Stage P2-1: data service ───────────────────────────────────────────────

class LoadCleanPenguinsTest(TestCase):
    def test_no_missing_values_after_clean(self):
        df = load_clean_penguins()
        self.assertFalse(df.isna().any().any())

    def test_has_expected_columns(self):
        df = load_clean_penguins()
        for col in INPUT_FEATURES + [TARGET]:
            self.assertIn(col, df.columns)

    def test_year_is_integer(self):
        df = load_clean_penguins()
        self.assertTrue(str(df["year"].dtype).startswith("int"))

    def test_three_species(self):
        df = load_clean_penguins()
        self.assertEqual(sorted(df[TARGET].unique()), sorted(SPECIES_ORDER))


class ComputeMadTest(TestCase):
    def test_mad_for_every_numeric_feature(self):
        df = load_clean_penguins()
        mad = compute_mad(df)
        for col in NUMERIC_FEATURES:
            self.assertIn(col, mad)

    def test_mad_floored_at_epsilon(self):
        import pandas as pd
        # A constant column has MAD 0 → must be floored to epsilon
        df = pd.DataFrame({c: [5.0] * 10 for c in NUMERIC_FEATURES})
        mad = compute_mad(df)
        for col in NUMERIC_FEATURES:
            self.assertGreaterEqual(mad[col], MAD_EPSILON)
            self.assertEqual(mad[col], MAD_EPSILON)

    def test_mad_positive_on_real_data(self):
        df = load_clean_penguins()
        mad = compute_mad(df)
        for col in NUMERIC_FEATURES:
            self.assertGreater(mad[col], 0)


class GetPenguinDataTest(TestCase):
    def test_split_sizes_sum_to_total(self):
        data = get_penguin_data(seed=42)
        self.assertEqual(len(data.X_train) + len(data.X_test), len(data.X_all))

    def test_test_size_is_roughly_20_percent(self):
        data = get_penguin_data(seed=42)
        ratio = len(data.X_test) / len(data.X_all)
        self.assertAlmostEqual(ratio, 0.2, delta=0.02)

    def test_target_not_in_features(self):
        data = get_penguin_data(seed=42)
        self.assertNotIn(TARGET, data.X_train.columns)

    def test_features_are_exactly_input_features(self):
        data = get_penguin_data(seed=42)
        self.assertEqual(list(data.X_train.columns), INPUT_FEATURES)

    def test_stratified_split_preserves_all_classes(self):
        data = get_penguin_data(seed=42)
        self.assertEqual(set(data.y_train.unique()), set(SPECIES_ORDER))
        self.assertEqual(set(data.y_test.unique()), set(SPECIES_ORDER))

    def test_categories_collected(self):
        data = get_penguin_data(seed=42)
        for col in CATEGORICAL_FEATURES:
            self.assertIn(col, data.categories)
            self.assertGreater(len(data.categories[col]), 0)

    def test_numeric_ranges_present(self):
        data = get_penguin_data(seed=42)
        for col in NUMERIC_FEATURES:
            lo, hi = data.numeric_ranges[col]
            self.assertLess(lo, hi)

    def test_observed_years_are_ints(self):
        data = get_penguin_data(seed=42)
        self.assertTrue(all(isinstance(y, int) for y in data.observed_years))

    def test_reproducible_with_same_seed(self):
        d1 = get_penguin_data(seed=7)
        d2 = get_penguin_data(seed=7)
        # Cached → identical object, but assert index alignment regardless
        self.assertTrue(d1.X_train.equals(d2.X_train))

    def test_biometric_excludes_year(self):
        self.assertNotIn("year", BIOMETRIC_FEATURES)
        self.assertEqual(len(BIOMETRIC_FEATURES), 4)


# ── View smoke test ─────────────────────────────────────────────────────────

class Project2IndexViewTest(TestCase):
    def test_index_returns_200(self):
        response = self.client.get("/project2/")
        self.assertEqual(response.status_code, 200)

    def test_index_shows_dataset_summary(self):
        response = self.client.get("/project2/")
        self.assertContains(response, "Palmer Penguins")
        self.assertContains(response, "species")

    def test_index_lists_biometric_features(self):
        response = self.client.get("/project2/")
        for f in BIOMETRIC_FEATURES:
            self.assertContains(response, f)


class HomePageProject2LinkTest(TestCase):
    def test_home_lists_project2(self):
        response = self.client.get("/home/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Project 2")
