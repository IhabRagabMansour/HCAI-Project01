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


# ── Stage P2-3: lambda selection + dashboard (Task 2) ──────────────────────

class SelectionScoreTest(TestCase):
    def test_score_is_error_plus_penalty(self):
        from .services.selection import selection_score
        # (1 - 0.9) + 0.01 * 5 = 0.1 + 0.05 = 0.15
        self.assertAlmostEqual(selection_score(0.9, 5, 0.01), 0.15)

    def test_lambda_zero_is_pure_error(self):
        from .services.selection import selection_score
        self.assertAlmostEqual(selection_score(0.95, 20, 0.0), 0.05)


class SelectBestTest(TestCase):
    def _grid(self):
        # (model_class, param_name, param_value, pipeline, test_acc, complexity)
        from .services.grids import ModelEntry
        return (
            ModelEntry("tree", "max_leaf_nodes", 2, None, 0.94, 2),
            ModelEntry("tree", "max_leaf_nodes", 5, None, 0.97, 5),
            ModelEntry("tree", "max_leaf_nodes", None, None, 1.00, 17),
        )

    def test_lambda_zero_picks_most_accurate(self):
        from .services.selection import select_best
        best = select_best(self._grid(), lam=0.0)
        self.assertEqual(best.test_accuracy, 1.00)

    def test_large_lambda_picks_simplest(self):
        from .services.selection import select_best
        best = select_best(self._grid(), lam=0.05)
        self.assertEqual(best.complexity, 2)

    def test_intermediate_lambda_picks_middle(self):
        from .services.selection import select_best
        # At lam=0.005: scores = 0.06+0.010=0.070 ; 0.03+0.025=0.055 ; 0.0+0.085=0.085
        # The middle model (Omega=5) is the unique minimizer.
        best = select_best(self._grid(), lam=0.005)
        self.assertEqual(best.complexity, 5)

    def test_tie_breaks_toward_simpler(self):
        from .services.grids import ModelEntry
        from .services.selection import select_best
        grid = (
            ModelEntry("tree", "max_leaf_nodes", 3, None, 1.0, 3),
            ModelEntry("tree", "max_leaf_nodes", 10, None, 1.0, 10),
        )
        best = select_best(grid, lam=0.0)  # both error 0 -> simpler wins
        self.assertEqual(best.complexity, 3)


class GetSelectedModelTest(TestCase):
    def test_lambda_zero_tree_is_accurate(self):
        from .services.selection import get_selected_model
        sel = get_selected_model("tree", lam=0.0, seed=42)
        self.assertGreater(sel.test_accuracy, 0.9)

    def test_high_lambda_tree_is_simplest_useful(self):
        # The 2-leaf tree cannot separate 3 species (only ~76% acc), so even at
        # high lambda the 3-leaf tree (the 3-class floor) is the minimizer.
        from .services.selection import get_selected_model
        sel = get_selected_model("tree", lam=0.05, seed=42)
        self.assertEqual(sel.complexity, 3)

    def test_increasing_lambda_is_non_increasing_in_complexity(self):
        from .services.selection import get_selected_model
        complexities = [
            get_selected_model("tree", lam=lam, seed=42).complexity
            for lam in [0.0, 0.005, 0.01, 0.02, 0.05]
        ]
        # Higher lambda should never select a more complex model
        for a, b in zip(complexities, complexities[1:]):
            self.assertGreaterEqual(a, b)

    def test_works_for_logreg(self):
        from .services.selection import get_selected_model
        sel = get_selected_model("logreg", lam=0.0, seed=42)
        self.assertEqual(sel.model_class, "logreg")
        self.assertGreater(sel.test_accuracy, 0.9)

    def test_clamp_lambda_out_of_range(self):
        from .services.selection import clamp_lambda, LAMBDA_MAX
        self.assertEqual(clamp_lambda("999"), LAMBDA_MAX)
        self.assertEqual(clamp_lambda("-1"), 0.0)
        self.assertEqual(clamp_lambda("abc"), 0.0)

    def test_normalize_model_class(self):
        from .services.selection import normalize_model_class
        self.assertEqual(normalize_model_class("logreg"), "logreg")
        self.assertEqual(normalize_model_class("bogus"), "tree")
        self.assertEqual(normalize_model_class(None), "tree")


class DashboardViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project2/dashboard/")
        self.assertEqual(response.status_code, 200)

    def test_has_controls(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, 'name="model"')
        self.assertContains(response, 'name="lambda"')
        self.assertContains(response, 'id="frontier-chart"')

    def test_default_is_tree_with_plot(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "data:image/png;base64,")

    def test_grid_table_present(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "Selection score")

    def test_high_lambda_selects_simplest_tree(self):
        response = self.client.get("/project2/dashboard/?model=tree&lambda=0.05")
        self.assertContains(response, "p2-row-selected")
        # The 3-leaf tree is the simplest useful model for 3 classes
        self.assertContains(response, "<strong>3</strong>")

    def test_logreg_mode_defers_coef_table(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertContains(response, "Task 3")
        # No tree PNG in logreg mode
        self.assertNotContains(response, "data:image/png;base64,")

    def test_frontier_payload_embedded(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, 'id="frontier-data"')
        self.assertContains(response, "frontier_chart.js")

    def test_invalid_params_fall_back_gracefully(self):
        response = self.client.get("/project2/dashboard/?model=bogus&lambda=nan")
        self.assertEqual(response.status_code, 200)

    def test_nav_has_dashboard_link(self):
        response = self.client.get("/project2/")
        self.assertContains(response, "/project2/dashboard/")
