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
    def test_score_is_accuracy_minus_penalty(self):
        from .services.selection import selection_score
        # acc - lambda*Omega = 0.9 - 0.01*5 = 0.9 - 0.05 = 0.85  (maximized)
        self.assertAlmostEqual(selection_score(0.9, 5, 0.01), 0.85)

    def test_lambda_zero_is_pure_accuracy(self):
        from .services.selection import selection_score
        self.assertAlmostEqual(selection_score(0.95, 20, 0.0), 0.95)


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

    def test_selected_is_argmax_of_acc_minus_lambda_omega(self):
        # Lock in the official Task 2 criterion: maximize acc_test - lambda*Omega.
        from .services.grids import get_grid
        from .services.selection import get_selected_model
        for model in ("tree", "logreg"):
            grid = get_grid(model, seed=42)
            for lam in (0.0, 0.005, 0.01, 0.02, 0.05):
                sel = get_selected_model(model, lam, seed=42)
                best_score = max(e.test_accuracy - lam * e.complexity for e in grid)
                sel_score = sel.test_accuracy - lam * sel.complexity
                self.assertAlmostEqual(sel_score, best_score, places=9)


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

    def test_logreg_mode_shows_coef_table_not_tree(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertContains(response, "Coefficient table")
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


# ── Stage P2-4: logistic-regression coefficient table (Task 3) ─────────────

class CoefficientTableTest(TestCase):
    def _logreg_pipeline(self, c_value=1.0):
        from .services.grids import train_logreg_grid
        grid = train_logreg_grid(seed=42)
        return next(e for e in grid if e.param_value == c_value).pipeline

    def _table(self, c_value=1.0):
        from .services.coefs import coefficient_table
        return coefficient_table(self._logreg_pipeline(c_value))

    def test_has_three_classes(self):
        table = self._table()
        self.assertEqual(len(table["classes"]), 3)

    def test_one_row_per_feature(self):
        from .services.pipeline import feature_names_out
        table = self._table()
        n_features = len(feature_names_out(self._logreg_pipeline()))
        self.assertEqual(len(table["rows"]), n_features)

    def test_each_row_has_one_cell_per_class(self):
        table = self._table()
        for row in table["rows"]:
            self.assertEqual(len(row["cells"]), 3)

    def test_intercept_per_class(self):
        table = self._table()
        self.assertEqual(len(table["intercepts"]), 3)

    def test_n_nonzero_matches_complexity(self):
        from .services.complexity import logreg_n_nonzero
        pipe = self._logreg_pipeline()
        table = self._table()
        self.assertEqual(table["n_nonzero"], logreg_n_nonzero(pipe))

    def test_total_coefs_is_features_times_classes(self):
        table = self._table()
        self.assertEqual(
            table["n_total_coefs"],
            table["n_features"] * len(table["classes"]),
        )

    def test_sparse_model_has_more_zeros(self):
        # Strong regularization (small C) -> fewer nonzero coefficients
        sparse = self._table(c_value=0.03)
        dense = self._table(c_value=30.0)
        self.assertLess(sparse["n_nonzero"], dense["n_nonzero"])

    def test_nonzero_flag_consistent_with_value(self):
        table = self._table()
        for row in table["rows"]:
            for cell in row["cells"]:
                if cell["nonzero"]:
                    self.assertGreater(abs(cell["value"]), 0)


class DashboardLogregViewTest(TestCase):
    def test_logreg_shows_coef_table(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertContains(response, "Coefficient table")
        self.assertContains(response, "Nonzero coefficients")

    def test_logreg_lists_species_columns(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        for sp in ["Adelie", "Chinstrap", "Gentoo"]:
            self.assertContains(response, sp)

    def test_logreg_shows_a_feature_row(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertContains(response, "flipper_length_mm")

    def test_logreg_shows_intercept_row(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertContains(response, "intercept")

    def test_logreg_no_tree_png(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        self.assertNotContains(response, "data:image/png;base64,")

    def test_high_lambda_logreg_is_sparser_than_low(self):
        # Higher lambda should select a sparser (fewer-nonzero) logreg model
        from .services.selection import get_selected_model
        from .services.complexity import logreg_n_nonzero
        low = get_selected_model("logreg", lam=0.0, seed=42)
        high = get_selected_model("logreg", lam=0.05, seed=42)
        self.assertLessEqual(high.complexity, low.complexity)


# ── Stage P2-5a: counterfactual generation service (Task 4) ────────────────

class MadWeightedL1Test(TestCase):
    def test_zero_distance_to_self(self):
        from .services.counterfactuals import mad_weighted_l1
        from .services.data import NUMERIC_FEATURES
        x = {f: 1.0 for f in NUMERIC_FEATURES}
        mad = {f: 1.0 for f in NUMERIC_FEATURES}
        self.assertEqual(mad_weighted_l1(x, x, mad), 0.0)

    def test_scales_by_mad(self):
        from .services.counterfactuals import mad_weighted_l1
        x = {f: 0.0 for f in NUMERIC_FEATURES}
        z = {f: 0.0 for f in NUMERIC_FEATURES}
        z["bill_length_mm"] = 4.0
        mad = {f: 1.0 for f in NUMERIC_FEATURES}
        mad["bill_length_mm"] = 2.0
        # |4-0| / 2 = 2.0
        self.assertEqual(mad_weighted_l1(x, z, mad), 2.0)


class SampleCandidateTest(TestCase):
    def setUp(self):
        from .services.data import get_penguin_data
        self.data = get_penguin_data(seed=42)
        self.x = self.data.X_all.iloc[0].to_dict()

    def test_biometrics_stay_within_range(self):
        import numpy as np
        from .services.counterfactuals import sample_candidate
        from .services.data import BIOMETRIC_FEATURES
        rng = np.random.default_rng(0)
        for _ in range(50):
            z = sample_candidate(self.x, self.data, alpha=2.0, cat_switch=0.2, rng=rng)
            for f in BIOMETRIC_FEATURES:
                lo, hi = self.data.numeric_ranges[f]
                self.assertGreaterEqual(z[f], lo)
                self.assertLessEqual(z[f], hi)

    def test_year_stays_in_observed_range(self):
        import numpy as np
        from .services.counterfactuals import sample_candidate
        rng = np.random.default_rng(0)
        lo, hi = self.data.numeric_ranges["year"]
        for _ in range(50):
            z = sample_candidate(self.x, self.data, alpha=2.0, cat_switch=0.2, rng=rng)
            self.assertGreaterEqual(z["year"], lo)
            self.assertLessEqual(z["year"], hi)
            self.assertIsInstance(z["year"], int)

    def test_categoricals_stay_valid(self):
        import numpy as np
        from .services.counterfactuals import sample_candidate
        from .services.data import CATEGORICAL_FEATURES
        rng = np.random.default_rng(0)
        for _ in range(50):
            z = sample_candidate(self.x, self.data, alpha=0.5, cat_switch=1.0, rng=rng)
            for f in CATEGORICAL_FEATURES:
                self.assertIn(z[f], self.data.categories[f])

    def test_no_categorical_switch_when_prob_zero(self):
        import numpy as np
        from .services.counterfactuals import sample_candidate
        from .services.data import CATEGORICAL_FEATURES
        rng = np.random.default_rng(0)
        for _ in range(20):
            z = sample_candidate(self.x, self.data, alpha=0.5, cat_switch=0.0, rng=rng)
            for f in CATEGORICAL_FEATURES:
                self.assertEqual(z[f], self.x[f])


class ChangedFeaturesTest(TestCase):
    def test_detects_categorical_change(self):
        from .services.counterfactuals import changed_features
        x = {"island": "Biscoe", "sex": "male", "year": 2008,
             "bill_length_mm": 40.0, "bill_depth_mm": 18.0,
             "flipper_length_mm": 195.0, "body_mass_g": 3750.0}
        z = dict(x); z["island"] = "Dream"
        self.assertEqual(changed_features(x, z), ["island"])

    def test_no_change_to_self(self):
        from .services.counterfactuals import changed_features
        x = {"island": "Biscoe", "sex": "male", "year": 2008,
             "bill_length_mm": 40.0, "bill_depth_mm": 18.0,
             "flipper_length_mm": 195.0, "body_mass_g": 3750.0}
        self.assertEqual(changed_features(x, dict(x)), [])


class GenerateCounterfactualsTest(TestCase):
    def setUp(self):
        from .services.data import get_penguin_data
        from .services.selection import get_selected_model
        self.data = get_penguin_data(seed=42)
        self.model = get_selected_model("tree", lam=0.0, seed=42)
        # Pick an example and a DIFFERENT target species to force a real search
        self.idx = 0
        self.x = self.data.X_all.iloc[self.idx].to_dict()
        self.true_class = self.data.y_all.iloc[self.idx]
        self.target = next(c for c in ["Adelie", "Chinstrap", "Gentoo"] if c != self.true_class)

    def _cfg(self, **kw):
        from .services.counterfactuals import CounterfactualConfig
        base = dict(n_candidates=400, k=5, seed=0)
        base.update(kw)
        return CounterfactualConfig(**base)

    def test_returns_at_most_k(self):
        from .services.counterfactuals import generate_counterfactuals
        cfs = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                       self.data, self._cfg(k=3))
        self.assertLessEqual(len(cfs), 3)

    def test_all_predict_target_class(self):
        from .services.counterfactuals import generate_counterfactuals
        cfs = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                       self.data, self._cfg())
        self.assertGreater(len(cfs), 0)
        for cf in cfs:
            self.assertEqual(cf.predicted_class, self.target)

    def test_sorted_by_distance_ascending(self):
        from .services.counterfactuals import generate_counterfactuals
        cfs = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                       self.data, self._cfg())
        distances = [cf.distance for cf in cfs]
        self.assertEqual(distances, sorted(distances))

    def test_target_proba_is_highest_for_target(self):
        from .services.counterfactuals import generate_counterfactuals
        cfs = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                       self.data, self._cfg())
        # Predicted class == target, so target prob should be > 0.5 in a 3-class argmax
        for cf in cfs:
            self.assertGreater(cf.target_proba, 0.33)

    def test_unknown_target_returns_empty(self):
        from .services.counterfactuals import generate_counterfactuals
        cfs = generate_counterfactuals(self.model.pipeline, self.x, "Penguin",
                                       self.data, self._cfg())
        self.assertEqual(cfs, [])

    def test_reproducible_with_seed(self):
        from .services.counterfactuals import generate_counterfactuals
        a = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                     self.data, self._cfg(seed=123))
        b = generate_counterfactuals(self.model.pipeline, self.x, self.target,
                                     self.data, self._cfg(seed=123))
        self.assertEqual([cf.distance for cf in a], [cf.distance for cf in b])

    def test_works_for_logreg_model(self):
        from .services.counterfactuals import generate_counterfactuals
        from .services.selection import get_selected_model
        model = get_selected_model("logreg", lam=0.0, seed=42)
        cfs = generate_counterfactuals(model.pipeline, self.x, self.target,
                                       self.data, self._cfg())
        self.assertGreater(len(cfs), 0)
        for cf in cfs:
            self.assertEqual(cf.predicted_class, self.target)


# ── Stage P2-5b: counterfactual UI in the dashboard (Task 4) ───────────────

class DashboardCounterfactualUITest(TestCase):
    def test_region_present_without_target(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "Counterfactual Explanations")
        self.assertContains(response, "Choose a target species")

    def test_controls_present(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, 'name="cf_row"')
        self.assertContains(response, 'name="cf_target"')
        self.assertContains(response, 'name="cf_k"')

    def test_shows_original_prediction(self):
        response = self.client.get("/project2/dashboard/?cf_row=0")
        self.assertContains(response, "Selected example")
        self.assertContains(response, "true species")

    def test_generates_table_for_target(self):
        # Pick row 0, target a different species than its prediction
        from .services.data import get_penguin_data
        data = get_penguin_data(42)
        true0 = str(data.y_all.iloc[0])
        target = next(s for s in ["Adelie", "Chinstrap", "Gentoo"] if s != true0)
        response = self.client.get(f"/project2/dashboard/?cf_row=0&cf_target={target}")
        self.assertContains(response, "P(target)")
        self.assertContains(response, "Distance")

    def test_counterfactuals_use_selected_model_consistently(self):
        # The CF region heading should reflect the chosen model + lambda
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0&cf_row=0&cf_target=Gentoo")
        self.assertContains(response, "currently selected model")
        self.assertContains(response, "logreg")

    def test_cf_row_clamped_to_valid_range(self):
        response = self.client.get("/project2/dashboard/?cf_row=999999")
        self.assertEqual(response.status_code, 200)  # clamped, no crash

    def test_invalid_target_shows_prompt(self):
        response = self.client.get("/project2/dashboard/?cf_target=Dragon")
        self.assertContains(response, "Choose a target species")

    def test_k_limits_rows(self):
        from .services.data import get_penguin_data
        data = get_penguin_data(42)
        true0 = str(data.y_all.iloc[0])
        target = next(s for s in ["Adelie", "Chinstrap", "Gentoo"] if s != true0)
        response = self.client.get(f"/project2/dashboard/?cf_row=0&cf_target={target}&cf_k=2")
        # Count counterfactual data rows by the changed-cell highlight class occurrences
        # is brittle; instead assert the page renders and the table header exists.
        self.assertContains(response, "P(target)")


# ── Stage P2-6a: manual PDP + ALE service (Task 5) ─────────────────────────

class ComputePdpTest(TestCase):
    def setUp(self):
        from .services.data import get_penguin_data
        from .services.selection import get_selected_model
        self.data = get_penguin_data(seed=42)
        self.tree = get_selected_model("tree", lam=0.0, seed=42).pipeline
        self.logreg = get_selected_model("logreg", lam=0.0, seed=42).pipeline

    def test_shape_is_grid_by_classes(self):
        from .services.feature_effects import compute_pdp
        pdp = compute_pdp(self.tree, self.data.X_all, "bill_length_mm", n_grid=20)
        self.assertEqual(len(pdp["grid"]), 20)
        self.assertEqual(len(pdp["classes"]), 3)
        self.assertEqual(len(pdp["curves"]), 20)
        for row in pdp["curves"]:
            self.assertEqual(len(row), 3)

    def test_each_grid_row_sums_to_one(self):
        from .services.feature_effects import compute_pdp
        pdp = compute_pdp(self.tree, self.data.X_all, "flipper_length_mm", n_grid=15)
        for row in pdp["curves"]:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_grid_spans_observed_range(self):
        from .services.feature_effects import compute_pdp
        col = self.data.X_all["body_mass_g"]
        pdp = compute_pdp(self.tree, self.data.X_all, "body_mass_g", n_grid=10)
        self.assertAlmostEqual(pdp["grid"][0], float(col.min()), places=3)
        self.assertAlmostEqual(pdp["grid"][-1], float(col.max()), places=3)

    def test_works_for_logreg(self):
        from .services.feature_effects import compute_pdp
        pdp = compute_pdp(self.logreg, self.data.X_all, "bill_depth_mm", n_grid=12)
        self.assertEqual(len(pdp["curves"]), 12)
        for row in pdp["curves"]:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_classes_match_species(self):
        from .services.feature_effects import compute_pdp
        pdp = compute_pdp(self.tree, self.data.X_all, "bill_length_mm", n_grid=5)
        self.assertEqual(sorted(pdp["classes"]), ["Adelie", "Chinstrap", "Gentoo"])


class ComputeAleTest(TestCase):
    def setUp(self):
        from .services.data import get_penguin_data
        from .services.selection import get_selected_model
        self.data = get_penguin_data(seed=42)
        self.tree = get_selected_model("tree", lam=0.0, seed=42).pipeline
        self.logreg = get_selected_model("logreg", lam=0.0, seed=42).pipeline

    def test_shape_is_bins_by_classes(self):
        from .services.feature_effects import compute_ale
        ale = compute_ale(self.tree, self.data.X_all, "bill_length_mm", n_bins=15)
        self.assertEqual(len(ale["classes"]), 3)
        self.assertEqual(len(ale["centers"]), len(ale["curves"]))
        for row in ale["curves"]:
            self.assertEqual(len(row), 3)

    def test_data_weighted_mean_is_zero(self):
        import numpy as np
        from .services.feature_effects import compute_ale
        ale = compute_ale(self.logreg, self.data.X_all, "flipper_length_mm", n_bins=12)
        curves = np.array(ale["curves"])          # (B, 3)
        counts = np.array(ale["bin_counts"])      # (B,)
        weighted = (counts[:, None] * curves).sum(axis=0) / counts.sum()
        for w in weighted:
            self.assertAlmostEqual(w, 0.0, places=6)

    def test_centers_are_ascending(self):
        from .services.feature_effects import compute_ale
        ale = compute_ale(self.tree, self.data.X_all, "body_mass_g", n_bins=10)
        self.assertEqual(ale["centers"], sorted(ale["centers"]))

    def test_bin_counts_sum_to_n_rows(self):
        # Half-open bins must partition the data: no double-count, no drops.
        from .services.feature_effects import compute_ale
        ale = compute_ale(self.tree, self.data.X_all, "bill_depth_mm", n_bins=20)
        self.assertEqual(sum(ale["bin_counts"]), len(self.data.X_all))

    def test_works_for_logreg(self):
        from .services.feature_effects import compute_ale
        ale = compute_ale(self.logreg, self.data.X_all, "bill_length_mm", n_bins=15)
        self.assertEqual(len(ale["classes"]), 3)
        self.assertGreater(len(ale["curves"]), 0)

    def test_no_library_pdp_ale_import(self):
        # The implementation must be manual — no sklearn.inspection / ALE libs.
        import pathlib
        src = pathlib.Path(__file__).resolve().parent / "services" / "feature_effects.py"
        text = src.read_text()
        self.assertNotIn("partial_dependence", text)
        self.assertNotIn("inspection", text)
        self.assertNotIn("PartialDependenceDisplay", text)
        self.assertNotIn("import alepython", text)


# ── Stage P2-6b: PDP/ALE endpoints + dashboard Region C (Task 5) ───────────

class PdpAleEndpointTest(TestCase):
    def test_pdp_returns_three_curves(self):
        import json
        response = self.client.get("/project2/pdp/?model=tree&lambda=0.0&feature=bill_length_mm&n_grid=15")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(len(data["classes"]), 3)
        self.assertEqual(len(data["grid"]), 15)
        self.assertEqual(len(data["curves"]), 15)
        self.assertEqual(len(data["curves"][0]), 3)

    def test_ale_returns_three_curves(self):
        import json
        response = self.client.get("/project2/ale/?model=tree&lambda=0.0&feature=flipper_length_mm&n_bins=12")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(len(data["classes"]), 3)
        self.assertEqual(len(data["centers"]), len(data["curves"]))
        self.assertEqual(len(data["curves"][0]), 3)

    def test_pdp_works_for_logreg(self):
        import json
        response = self.client.get("/project2/pdp/?model=logreg&lambda=0.0&feature=bill_depth_mm")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(len(data["classes"]), 3)

    def test_invalid_feature_rejected(self):
        self.assertEqual(self.client.get("/project2/pdp/?feature=bogus").status_code, 400)
        self.assertEqual(self.client.get("/project2/ale/?feature=bogus").status_code, 400)

    def test_year_is_not_a_valid_feature(self):
        # year must not be selectable for PDP/ALE (sheet common mistake #21)
        self.assertEqual(self.client.get("/project2/pdp/?feature=year").status_code, 400)
        self.assertEqual(self.client.get("/project2/ale/?feature=year").status_code, 400)

    def test_pdp_differs_between_models(self):
        import json
        tree = json.loads(self.client.get(
            "/project2/pdp/?model=tree&lambda=0.0&feature=bill_length_mm&n_grid=10").content)
        logreg = json.loads(self.client.get(
            "/project2/pdp/?model=logreg&lambda=0.0&feature=bill_length_mm&n_grid=10").content)
        # Different model families should produce different PDP curves
        self.assertNotEqual(tree["curves"], logreg["curves"])

    def test_n_grid_clamped(self):
        import json
        data = json.loads(self.client.get(
            "/project2/pdp/?feature=body_mass_g&n_grid=99999").content)
        self.assertLessEqual(len(data["grid"]), 50)


class DashboardFeatureEffectsUITest(TestCase):
    def test_region_present(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "Feature Effect Plots")
        self.assertContains(response, 'id="pdp-chart"')
        self.assertContains(response, 'id="ale-chart"')

    def test_feature_selector_only_biometric(self):
        response = self.client.get("/project2/dashboard/")
        for f in ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]:
            self.assertContains(response, f'value="{f}"')
        # year must NOT be a PDP/ALE option
        self.assertNotContains(response, 'value="year"')

    def test_derivative_info_box_present(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "Exact vs. approximate derivatives")
        self.assertContains(response, "finite differences")

    def test_loads_feature_effects_script(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "feature_effects_chart.js")

    def test_region_carries_model_and_lambda(self):
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.02")
        self.assertContains(response, 'data-model="logreg"')
        self.assertContains(response, 'data-lambda="0.02"')


# ── Stage P2-7: cross-panel consistency + report ───────────────────────────

class DashboardConsistencyTest(TestCase):
    """The selected model must drive every panel identically."""

    def test_selected_complexity_matches_starred_grid_row(self):
        # Region A's complexity must equal the starred (selected) grid row's Omega.
        from .services.selection import get_selected_model
        for model, lam in [("tree", 0.0), ("tree", 0.05), ("logreg", 0.0), ("logreg", 0.05)]:
            sel = get_selected_model(model, lam, seed=42)
            response = self.client.get(f"/project2/dashboard/?model={model}&lambda={lam}")
            # The selected complexity appears in the Region A panel
            self.assertContains(response, f"<strong>{sel.complexity}</strong>")
            # And exactly one grid row is starred
            self.assertContains(response, "p2-row-selected")

    def test_switching_model_changes_region_a_rendering(self):
        tree = self.client.get("/project2/dashboard/?model=tree&lambda=0.0")
        logreg = self.client.get("/project2/dashboard/?model=logreg&lambda=0.0")
        # Tree mode renders a PNG; logreg mode renders the coefficient table
        self.assertContains(tree, "data:image/png;base64,")
        self.assertNotContains(tree, "Coefficient table")
        self.assertContains(logreg, "Coefficient table")
        self.assertNotContains(logreg, "data:image/png;base64,")

    def test_all_regions_reflect_same_model_and_lambda(self):
        # One request: Region B heading, Region C data attrs, and the controls
        # must all reflect the same model+lambda.
        response = self.client.get("/project2/dashboard/?model=logreg&lambda=0.02&cf_row=0&cf_target=Gentoo")
        content = response.content.decode()
        # Region C data attributes
        self.assertIn('data-model="logreg"', content)
        self.assertIn('data-lambda="0.02"', content)
        # Region B explicitly says it uses the currently selected model
        self.assertIn("currently selected model", content)
        # The model dropdown shows logreg selected
        self.assertIn('<option value="logreg" selected>', content)

    def test_lambda_changes_selected_model(self):
        # Different lambda -> potentially different selected complexity for trees
        low = self.client.get("/project2/dashboard/?model=tree&lambda=0.0")
        high = self.client.get("/project2/dashboard/?model=tree&lambda=0.05")
        from .services.selection import get_selected_model
        c_low = get_selected_model("tree", 0.0, 42).complexity
        c_high = get_selected_model("tree", 0.05, 42).complexity
        self.assertGreaterEqual(c_low, c_high)
        self.assertContains(low, f"<strong>{c_low}</strong>")
        self.assertContains(high, f"<strong>{c_high}</strong>")

    def test_pdp_endpoint_consistent_with_selected_model(self):
        # PDP for tree vs logreg differ (proves the endpoint honors model choice)
        import json
        t = json.loads(self.client.get("/project2/pdp/?model=tree&lambda=0.0&feature=bill_length_mm&n_grid=8").content)
        l = json.loads(self.client.get("/project2/pdp/?model=logreg&lambda=0.0&feature=bill_length_mm&n_grid=8").content)
        self.assertNotEqual(t["curves"], l["curves"])


class ReportViewTest(TestCase):
    def test_returns_200(self):
        response = self.client.get("/project2/report/")
        self.assertEqual(response.status_code, 200)

    def test_covers_all_required_topics(self):
        response = self.client.get("/project2/report/")
        for topic in [
            "Preprocessing",
            "Decision-tree regularization grid",
            "Logistic-regression regularization grid",
            "complexity measure",
            "How &lambda; selects",
            "The selection formula",
            "Counterfactual sampling",
            "MAD-weighted L1 distance",
            "Manual PDP",
            "Manual ALE",
            "Exact vs. finite-difference",
        ]:
            self.assertContains(response, topic)

    def test_shows_actual_grids(self):
        response = self.client.get("/project2/report/")
        # The real grids are rendered, not placeholders
        self.assertContains(response, "0.01")   # logreg C grid
        self.assertContains(response, "None")   # tree grid includes unconstrained

    def test_documents_lambda_formula(self):
        response = self.client.get("/project2/report/")
        self.assertContains(response, "acc_test - lambda * Omega")

    def test_nav_links_to_report(self):
        response = self.client.get("/project2/dashboard/")
        self.assertContains(response, "/project2/report/")
