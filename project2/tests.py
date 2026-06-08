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


# ── Stage P2-2: tree grid, complexity, tree viz, Task 1 ────────────────────

class PreprocessorTest(TestCase):
    def _num_pipe(self, pre):
        # ColumnTransformer.transformers entries are (name, transformer, columns)
        for name, transformer, _cols in pre.transformers:
            if name == "num":
                return transformer
        raise AssertionError("no 'num' transformer")

    def test_unscaled_has_no_scaler(self):
        from .services.pipeline import build_preprocessor
        pre = build_preprocessor(scale=False)
        step_names = [name for name, _ in self._num_pipe(pre).steps]
        self.assertNotIn("scaler", step_names)

    def test_scaled_has_scaler(self):
        from .services.pipeline import build_preprocessor
        pre = build_preprocessor(scale=True)
        step_names = [name for name, _ in self._num_pipe(pre).steps]
        self.assertIn("scaler", step_names)

    def test_feature_names_include_onehot_expansion(self):
        from .services.grids import unconstrained_tree_entry
        from .services.pipeline import feature_names_out
        entry = unconstrained_tree_entry(seed=42)
        names = feature_names_out(entry.pipeline)
        # numeric kept as-is, categoricals expanded
        self.assertIn("flipper_length_mm", names)
        self.assertTrue(any(n.startswith("island_") for n in names))
        self.assertTrue(any(n.startswith("sex_") for n in names))


class TreeGridTest(TestCase):
    def test_grid_has_one_entry_per_param(self):
        from .services.grids import train_tree_grid, TREE_MAX_LEAF_NODES_GRID
        grid = train_tree_grid(seed=42)
        self.assertEqual(len(grid), len(TREE_MAX_LEAF_NODES_GRID))

    def test_entries_have_accuracy_and_complexity(self):
        from .services.grids import train_tree_grid
        for entry in train_tree_grid(seed=42):
            self.assertGreaterEqual(entry.test_accuracy, 0.0)
            self.assertLessEqual(entry.test_accuracy, 1.0)
            self.assertGreaterEqual(entry.complexity, 1)
            self.assertEqual(entry.model_class, "tree")
            self.assertEqual(entry.param_name, "max_leaf_nodes")

    def test_complexity_matches_get_n_leaves(self):
        from .services.grids import train_tree_grid
        from .services.complexity import tree_n_leaves
        for entry in train_tree_grid(seed=42):
            self.assertEqual(entry.complexity, tree_n_leaves(entry.pipeline))

    def test_max_leaf_nodes_caps_leaf_count(self):
        from .services.grids import train_tree_grid
        for entry in train_tree_grid(seed=42):
            if entry.param_value is not None:
                self.assertLessEqual(entry.complexity, entry.param_value)

    def test_higher_max_leaf_nodes_allows_more_leaves(self):
        from .services.grids import train_tree_grid
        grid = train_tree_grid(seed=42)
        two_leaf = next(e for e in grid if e.param_value == 2)
        unconstrained = next(e for e in grid if e.param_value is None)
        self.assertLessEqual(two_leaf.complexity, unconstrained.complexity)
        self.assertEqual(two_leaf.complexity, 2)

    def test_penguins_tree_is_accurate(self):
        # Penguins is an easy dataset; the full tree should classify well
        from .services.grids import unconstrained_tree_entry
        entry = unconstrained_tree_entry(seed=42)
        self.assertGreater(entry.test_accuracy, 0.9)


class TreeVizTest(TestCase):
    def test_png_is_nonempty_base64(self):
        import base64
        from .services.grids import unconstrained_tree_entry
        from .services.treeviz import tree_to_png_base64
        entry = unconstrained_tree_entry(seed=42)
        b64 = tree_to_png_base64(entry.pipeline)
        self.assertGreater(len(b64), 100)
        # Must decode without error and start with the PNG magic header
        raw = base64.b64decode(b64)
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")

    def test_text_mentions_a_feature(self):
        from .services.grids import unconstrained_tree_entry
        from .services.treeviz import tree_to_text
        entry = unconstrained_tree_entry(seed=42)
        text = tree_to_text(entry.pipeline)
        # At least one biometric feature should appear in the split text
        self.assertTrue(
            any(f in text for f in
                ["flipper_length_mm", "bill_length_mm", "bill_depth_mm", "body_mass_g"])
        )


class DecisionTreeViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project2/decision-tree/")
        self.assertEqual(response.status_code, 200)

    def test_shows_accuracy_and_leaves(self):
        response = self.client.get("/project2/decision-tree/")
        self.assertContains(response, "Test accuracy")
        self.assertContains(response, "Number of leaves")

    def test_embeds_png(self):
        response = self.client.get("/project2/decision-tree/")
        self.assertContains(response, "data:image/png;base64,")

    def test_nav_links_to_decision_tree(self):
        response = self.client.get("/project2/")
        self.assertContains(response, "/project2/decision-tree/")
