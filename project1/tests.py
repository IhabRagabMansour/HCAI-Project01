import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from .models import Dataset
from .services.data import humanize_dtype, infer_problem_type, extract_metadata
from .services.preprocess import (
    ExperimentConfig, PreparedData, handle_missing, encode_categorical,
    encode_target, scale_features, split_train_test, prepare_experiment,
)
from .services.train import (
    build_estimator, compute_score, train_and_score,
)
from .services.evaluate import (
    evaluate_classification, evaluate_regression,
    compute_feature_importance, build_evaluation,
)
from .services.pipeline import build_preprocessing, build_full_pipeline

import numpy as np
import pandas as pd


SAMPLE_CSV = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n"


# ── Service unit tests (no DB) ─────────────────────────────────────────────

class HumanizeDtypeTest(TestCase):
    def test_int(self):
        self.assertEqual(humanize_dtype(pd.Series([1, 2]).dtype), "integer")

    def test_float(self):
        self.assertEqual(humanize_dtype(pd.Series([1.0, 2.0]).dtype), "float")

    def test_string(self):
        self.assertEqual(humanize_dtype(pd.Series(["a", "b"]).dtype), "string")

    def test_bool(self):
        self.assertEqual(humanize_dtype(pd.Series([True, False]).dtype), "boolean")


class InferProblemTypeTest(TestCase):
    def test_classification_few_unique(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        self.assertEqual(infer_problem_type(df), "classification")

    def test_regression_many_unique(self):
        df = pd.DataFrame({"x": range(50), "y": [i * 1.5 for i in range(50)]})
        self.assertEqual(infer_problem_type(df), "regression")

    def test_classification_string_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": ["cat", "dog", "cat"]})
        self.assertEqual(infer_problem_type(df), "classification")

    def test_unknown_single_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.assertEqual(infer_problem_type(df), "unknown")


class ExtractMetadataTest(TestCase):
    def test_basic_structure(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "target": [0, 1]})
        meta = extract_metadata(df)
        self.assertEqual(meta["n_rows"], 2)
        self.assertEqual(meta["n_columns"], 3)
        self.assertEqual(meta["target_name"], "target")
        self.assertEqual(len(meta["columns"]), 3)
        self.assertEqual(len(meta["head"]), 2)

    def test_head_capped_at_10(self):
        df = pd.DataFrame({"x": range(50), "y": range(50)})
        meta = extract_metadata(df)
        self.assertEqual(len(meta["head"]), 10)

    def test_nan_becomes_none(self):
        import numpy as np
        df = pd.DataFrame({"x": [1.0, float("nan")], "y": [0, 1]})
        meta = extract_metadata(df)
        self.assertIsNone(meta["head"][1][0])


# ── Upload view integration tests ──────────────────────────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DatasetSignalTest(TestCase):
    def test_file_deleted_from_disk_on_dataset_delete(self):
        uploaded = SimpleUploadedFile("test.csv", SAMPLE_CSV, content_type="text/csv")
        dataset = Dataset.objects.create(original_name="test.csv", file=uploaded)
        file_path = dataset.file.path
        self.assertTrue(os.path.isfile(file_path))
        dataset.delete()
        self.assertFalse(os.path.isfile(file_path))


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DatasetUploadViewTest(TestCase):
    def test_upload_valid_csv_parses_metadata(self):
        csv = b"a,b,species\n5.1,3.5,0\n4.9,3.0,1\n4.7,3.2,0\n"
        uploaded = SimpleUploadedFile("iris.csv", csv, content_type="text/csv")
        response = self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.assertRedirects(response, f"/project1/datasets/{dataset.pk}/")
        self.assertEqual(dataset.original_name, "iris.csv")
        self.assertEqual(dataset.n_rows, 3)
        self.assertEqual(dataset.n_columns, 3)
        self.assertEqual(dataset.target_name, "species")
        self.assertEqual(dataset.problem_type, "classification")
        self.assertIsNone(dataset.parse_error)
        self.assertTrue(dataset.is_parsed)

    def test_upload_regression_dataset(self):
        rows = "\n".join(f"{i},{i * 1.5}" for i in range(30))
        csv = f"x,y\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("reg.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.assertEqual(dataset.problem_type, "regression")

    def test_upload_rejects_non_csv_extension(self):
        uploaded = SimpleUploadedFile("data.txt", SAMPLE_CSV, content_type="text/plain")
        response = self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.assertEqual(Dataset.objects.count(), 0)
        self.assertContains(response, "csv extension")

    def test_upload_rejects_empty_file(self):
        uploaded = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.assertEqual(Dataset.objects.count(), 0)

    def test_latin1_csv_parsed_successfully(self):
        csv = "name,score\nCaf\xe9,30\nNa\xefve,80\n".encode("latin-1")
        uploaded = SimpleUploadedFile("latin.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.assertIsNone(dataset.parse_error)
        self.assertEqual(dataset.n_rows, 2)

    def test_dataset_list_view(self):
        response = self.client.get("/project1/datasets/")
        self.assertEqual(response.status_code, 200)

    def test_dataset_detail_returns_404_for_missing(self):
        response = self.client.get("/project1/datasets/999/")
        self.assertEqual(response.status_code, 404)

    def test_detail_pagination_first_page(self):
        # 60 rows → 3 pages of 25, first page has 25 rows
        rows = "\n".join(f"{i},{i * 2},{ i % 3}" for i in range(60))
        csv = f"x,y,label\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("big.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Page 1 of 3")

    def test_detail_pagination_second_page(self):
        rows = "\n".join(f"{i},{i * 2},{i % 3}" for i in range(60))
        csv = f"x,y,label\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("big2.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/?page=2")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Page 2 of 3")

    def test_detail_pagination_out_of_range_returns_last_page(self):
        rows = "\n".join(f"{i},{i * 2},{i % 3}" for i in range(60))
        csv = f"x,y,label\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("big3.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/?page=999")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Page 3 of 3")

    def test_sort_asc_puts_smallest_value_first(self):
        csv = b"x,label\n30,a\n10,b\n20,c\n"
        uploaded = SimpleUploadedFile("sort.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/?sort=x&order=asc")
        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        # 10 should appear before 20 and 30 in the rendered HTML
        self.assertLess(content.index(">10<"), content.index(">20<"))
        self.assertLess(content.index(">20<"), content.index(">30<"))

    def test_sort_desc_puts_largest_value_first(self):
        csv = b"x,label\n30,a\n10,b\n20,c\n"
        uploaded = SimpleUploadedFile("sort2.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/?sort=x&order=desc")
        content = response.content.decode()
        self.assertLess(content.index(">30<"), content.index(">20<"))
        self.assertLess(content.index(">20<"), content.index(">10<"))

    def test_invalid_sort_column_ignored(self):
        csv = b"x,label\n1,a\n2,b\n"
        uploaded = SimpleUploadedFile("sort3.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        response = self.client.get(f"/project1/datasets/{dataset.pk}/?sort=nonexistent&order=asc")
        self.assertEqual(response.status_code, 200)


# ── Stage 5a: preprocessing service unit tests ─────────────────────────────

class HandleMissingTest(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "num": [1.0, float("nan"), 3.0],
            "cat": ["x", None, "x"],
        })

    def test_drop_removes_nan_rows(self):
        result = handle_missing(self.df, "drop")
        self.assertEqual(len(result), 2)
        self.assertFalse(result.isnull().any().any())

    def test_mean_mode_fills_numeric_with_mean(self):
        result = handle_missing(self.df, "mean_mode")
        self.assertAlmostEqual(result["num"].iloc[1], 2.0)  # mean(1, 3) = 2

    def test_mean_mode_fills_categorical_with_mode(self):
        result = handle_missing(self.df, "mean_mode")
        self.assertEqual(result["cat"].iloc[1], "x")

    def test_zero_empty_fills_numeric_with_zero(self):
        result = handle_missing(self.df, "zero_empty")
        self.assertEqual(result["num"].iloc[1], 0.0)

    def test_zero_empty_fills_categorical_with_empty_string(self):
        result = handle_missing(self.df, "zero_empty")
        self.assertEqual(result["cat"].iloc[1], "")

    def test_no_nan_in_result(self):
        for strategy in ("drop", "mean_mode", "zero_empty"):
            result = handle_missing(self.df, strategy)
            self.assertFalse(result.isnull().any().any(), msg=f"NaN found after strategy={strategy!r}")

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            handle_missing(self.df, "invalid")


class EncodeCategoricalTest(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
        })

    def test_onehot_expands_column(self):
        X_enc, names = encode_categorical(self.df, "onehot")
        self.assertIn("num", names)
        self.assertNotIn("cat", names)
        self.assertTrue(any("cat" in n for n in names))

    def test_onehot_preserves_numeric(self):
        X_enc, names = encode_categorical(self.df, "onehot")
        self.assertIn("num", names)

    def test_label_encodes_to_integers(self):
        X_enc, names = encode_categorical(self.df, "label")
        self.assertIn("cat", names)
        self.assertTrue(pd.api.types.is_numeric_dtype(X_enc["cat"]))

    def test_drop_removes_categorical_columns(self):
        _, names = encode_categorical(self.df, "drop")
        self.assertEqual(names, ["num"])

    def test_onehot_no_categorical_columns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X_enc, names = encode_categorical(df, "onehot")
        self.assertEqual(names, ["a", "b"])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            encode_categorical(self.df, "bad")


class EncodeTargetTest(TestCase):
    def test_regression_returns_floats(self):
        y = pd.Series([1.5, 2.5, 3.5])
        y_enc, label_map = encode_target(y, "regression")
        self.assertIsNone(label_map)
        np.testing.assert_array_almost_equal(y_enc, [1.5, 2.5, 3.5])

    def test_classification_numeric_target_no_label_map(self):
        y = pd.Series([0, 1, 0, 1])
        y_enc, label_map = encode_target(y, "classification")
        self.assertIsNone(label_map)

    def test_classification_string_target_encoded(self):
        y = pd.Series(["cat", "dog", "cat"])
        y_enc, label_map = encode_target(y, "classification")
        self.assertIsNotNone(label_map)
        self.assertIn("cat", label_map)
        self.assertIn("dog", label_map)
        self.assertEqual(len(set(y_enc)), 2)

    def test_classification_string_labels_are_ints(self):
        y = pd.Series(["a", "b", "c"])
        y_enc, label_map = encode_target(y, "classification")
        self.assertTrue(all(isinstance(v, int) for v in label_map.values()))


class ScaleFeaturesTest(TestCase):
    def setUp(self):
        self.X_train = np.array([[1.0, 10.0], [3.0, 30.0]])
        self.X_test = np.array([[2.0, 20.0]])

    def test_none_returns_unchanged(self):
        Xt, Xte = scale_features(self.X_train, self.X_test, "none")
        np.testing.assert_array_equal(Xt, self.X_train)
        np.testing.assert_array_equal(Xte, self.X_test)

    def test_standard_zero_mean(self):
        Xt, _ = scale_features(self.X_train, self.X_test, "standard")
        np.testing.assert_array_almost_equal(Xt.mean(axis=0), [0.0, 0.0])

    def test_minmax_bounds(self):
        Xt, _ = scale_features(self.X_train, self.X_test, "minmax")
        self.assertAlmostEqual(Xt.min(), 0.0)
        self.assertAlmostEqual(Xt.max(), 1.0)

    def test_no_data_leakage_standard(self):
        # Scaler fit on train: mean of col0 = 2, std = 1
        # Test value 2.0 → (2 - 2) / 1 = 0.0
        Xt, Xte = scale_features(self.X_train, self.X_test, "standard")
        self.assertAlmostEqual(Xte[0, 0], 0.0)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            scale_features(self.X_train, self.X_test, "bad")


class SplitTrainTestTest(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.X = np.random.randn(100, 3)
        self.y = np.array([0, 1] * 50)

    def test_correct_split_sizes(self):
        X_tr, X_te, y_tr, y_te = split_train_test(self.X, self.y, 0.2, 42, False)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)

    def test_reproducible_with_same_seed(self):
        r1 = split_train_test(self.X, self.y, 0.2, 42, False)
        r2 = split_train_test(self.X, self.y, 0.2, 42, False)
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_different_seeds_differ(self):
        r1 = split_train_test(self.X, self.y, 0.2, 42, False)
        r2 = split_train_test(self.X, self.y, 0.2, 99, False)
        self.assertFalse(np.array_equal(r1[0], r2[0]))


class PrepareExperimentTest(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.df = pd.DataFrame({
            "x1": np.random.randn(20),
            "x2": np.random.randn(20),
            "target": [0, 1] * 10,
        })
        self.config = ExperimentConfig(stratify=False)

    def test_correct_split_ratio(self):
        result = prepare_experiment(self.df, "target", "classification", self.config)
        self.assertEqual(result.n_train + result.n_test, 20)
        self.assertEqual(result.n_test, 4)  # 20% of 20

    def test_target_not_in_features(self):
        result = prepare_experiment(self.df, "target", "classification", self.config)
        self.assertNotIn("target", result.feature_names)

    def test_feature_names_correct(self):
        result = prepare_experiment(self.df, "target", "classification", self.config)
        self.assertEqual(result.feature_names, ["x1", "x2"])

    def test_reproducible_with_same_config(self):
        r1 = prepare_experiment(self.df, "target", "classification", self.config)
        r2 = prepare_experiment(self.df, "target", "classification", self.config)
        np.testing.assert_array_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_equal(r1.y_train, r2.y_train)

    def test_no_nan_after_pipeline_fit(self):
        # After Stage 8: X_train is RAW (still has NaN until pipeline runs).
        # The contract is that the FITTED pipeline produces NaN-free output.
        df = self.df.copy()
        df.loc[0, "x1"] = float("nan")
        result = prepare_experiment(df, "target", "classification", self.config)
        from sklearn.base import clone
        fitted = clone(result.preprocessing).fit(result.X_train)
        transformed = fitted.transform(result.X_train)
        self.assertFalse(np.isnan(np.asarray(transformed, dtype=float)).any())

    def test_onehot_expands_features(self):
        df = pd.DataFrame({
            "num": [float(i) for i in range(10)],
            "cat": ["a", "b"] * 5,
            "target": [0, 1] * 5,
        })
        config = ExperimentConfig(categorical_encoding="onehot", stratify=False)
        result = prepare_experiment(df, "target", "classification", config)
        self.assertEqual(result.n_features_before, 2)   # num + cat
        self.assertEqual(result.n_features_after, 3)    # num + cat_a + cat_b

    def test_string_target_gets_label_map(self):
        df = pd.DataFrame({
            "x": [float(i) for i in range(10)],
            "label": ["cat", "dog"] * 5,
        })
        config = ExperimentConfig(stratify=False)
        result = prepare_experiment(df, "label", "classification", config)
        self.assertIsNotNone(result.label_map)
        self.assertIn("cat", result.label_map)
        self.assertIn("dog", result.label_map)

    def test_regression_target_no_label_map(self):
        df = pd.DataFrame({
            "x": [float(i) for i in range(20)],
            "y": [float(i) * 1.5 for i in range(20)],
        })
        config = ExperimentConfig(stratify=False)
        result = prepare_experiment(df, "y", "regression", config)
        self.assertIsNone(result.label_map)

    def test_stratify_fallback_on_rare_class(self):
        # 1 sample per class → sklearn stratify raises → should fall back silently
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0, 1, 2, 3, 4],
        })
        config = ExperimentConfig(test_size=0.4, stratify=True)
        result = prepare_experiment(df, "y", "classification", config)
        self.assertFalse(result.stratify_used)

    def test_stratify_used_when_possible(self):
        config = ExperimentConfig(stratify=True)
        result = prepare_experiment(self.df, "target", "classification", config)
        self.assertTrue(result.stratify_used)

    def test_missing_target_column_raises(self):
        with self.assertRaises(ValueError):
            prepare_experiment(self.df, "nonexistent", "classification", self.config)

    def test_all_rows_dropped_raises(self):
        df = pd.DataFrame({"x": [float("nan")], "y": [float("nan")]})
        with self.assertRaises(ValueError):
            prepare_experiment(df, "y", "regression", ExperimentConfig(missing_strategy="drop"))


# ── Stage 5b: Experiment model + create view integration tests ──────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ExperimentCreateViewTest(TestCase):
    def setUp(self):
        csv = b"a,b,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0\n"
        uploaded = SimpleUploadedFile("exp.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.dataset = Dataset.objects.first()

    def test_create_form_get_returns_200(self):
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/experiments/new/")
        self.assertEqual(response.status_code, 200)

    def test_create_form_contains_fields(self):
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/experiments/new/")
        self.assertContains(response, "missing_strategy")
        self.assertContains(response, "categorical_encoding")
        self.assertContains(response, "scaling")
        self.assertContains(response, "test_size")

    def test_post_valid_form_creates_experiment(self):
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "My Experiment",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
            "stratify": "on",
        })
        from .models import Experiment
        self.assertEqual(Experiment.objects.count(), 1)
        exp = Experiment.objects.first()
        self.assertEqual(exp.name, "My Experiment")
        self.assertEqual(exp.dataset, self.dataset)

    def test_post_populates_result_fields(self):
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Test",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        exp = Experiment.objects.first()
        self.assertTrue(exp.is_prepared)
        self.assertIsNotNone(exp.n_train)
        self.assertIsNotNone(exp.n_test)
        self.assertIsNone(exp.prepare_error)

    def test_post_auto_names_experiment(self):
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        exp = Experiment.objects.first()
        self.assertIn("Experiment", exp.name)

    def test_post_redirects_to_experiment_detail(self):
        response = self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Redir Test",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        exp = Experiment.objects.first()
        self.assertRedirects(response, f"/project1/experiments/{exp.pk}/")

    def test_invalid_test_size_rejected(self):
        response = self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Bad",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.9",
            "random_seed": "42",
        })
        from .models import Experiment
        self.assertEqual(Experiment.objects.count(), 0)
        self.assertContains(response, "0.10")

    def test_experiment_detail_returns_200(self):
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Detail Test",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        exp = Experiment.objects.first()
        response = self.client.get(f"/project1/experiments/{exp.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, exp.name)


# ── Stage 5c: experiments list, delete, and warnings ───────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ExperimentListAndDeleteTest(TestCase):
    def setUp(self):
        csv = b"a,b,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0\n"
        uploaded = SimpleUploadedFile("exp2.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Exp Alpha",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        self.experiment = Experiment.objects.first()

    def test_dataset_detail_shows_experiments_section(self):
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/")
        self.assertContains(response, "Experiments")

    def test_dataset_detail_shows_experiment_name(self):
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/")
        self.assertContains(response, "Exp Alpha")

    def test_dataset_detail_shows_new_experiment_button(self):
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/")
        self.assertContains(response, "New Experiment")

    def test_delete_confirm_page_returns_200(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/delete/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Exp Alpha")

    def test_delete_post_removes_experiment(self):
        from .models import Experiment
        self.client.post(f"/project1/experiments/{self.experiment.pk}/delete/")
        self.assertEqual(Experiment.objects.count(), 0)

    def test_delete_post_redirects_to_dataset(self):
        response = self.client.post(f"/project1/experiments/{self.experiment.pk}/delete/")
        self.assertRedirects(response, f"/project1/datasets/{self.dataset.pk}/")

    def test_experiment_detail_shows_delete_button(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertContains(response, "delete")

    def test_multiple_experiments_all_listed(self):
        from .models import Experiment
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Exp Beta",
            "missing_strategy": "drop",
            "categorical_encoding": "label",
            "scaling": "minmax",
            "test_size": "0.3",
            "random_seed": "7",
        })
        self.assertEqual(Experiment.objects.count(), 2)
        response = self.client.get(f"/project1/datasets/{self.dataset.pk}/")
        self.assertContains(response, "Exp Alpha")
        self.assertContains(response, "Exp Beta")


# ── Stage 6a: training service unit tests ──────────────────────────────────

class BuildEstimatorTest(TestCase):
    def test_classification_algorithms_return_sklearn_objects(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        self.assertIsInstance(build_estimator("logreg",  42), LogisticRegression)
        self.assertIsInstance(build_estimator("rf_clf",  42), RandomForestClassifier)
        self.assertIsInstance(build_estimator("svm",     42), SVC)
        self.assertIsInstance(build_estimator("knn_clf", 42), KNeighborsClassifier)
        self.assertIsInstance(build_estimator("dt_clf",  42), DecisionTreeClassifier)

    def test_regression_algorithms_return_sklearn_objects(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        self.assertIsInstance(build_estimator("linreg",  42), LinearRegression)
        self.assertIsInstance(build_estimator("rf_reg",  42), RandomForestRegressor)
        self.assertIsInstance(build_estimator("svr",     42), SVR)
        self.assertIsInstance(build_estimator("knn_reg", 42), KNeighborsRegressor)
        self.assertIsInstance(build_estimator("dt_reg",  42), DecisionTreeRegressor)

    def test_random_seed_is_passed(self):
        est = build_estimator("rf_clf", random_seed=99)
        self.assertEqual(est.random_state, 99)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            build_estimator("bogus_algo", 42)


class ComputeScoreTest(TestCase):
    def test_accuracy_perfect(self):
        self.assertEqual(compute_score([0, 1, 0, 1], [0, 1, 0, 1], "accuracy"), 1.0)

    def test_accuracy_half(self):
        self.assertEqual(compute_score([0, 1, 0, 1], [0, 0, 0, 0], "accuracy"), 0.5)

    def test_f1_perfect(self):
        self.assertEqual(compute_score([0, 1, 0, 1], [0, 1, 0, 1], "f1"), 1.0)

    def test_precision_recall_run(self):
        # Smoke test — both produce a number
        p = compute_score([0, 1, 0, 1], [0, 1, 1, 1], "precision")
        r = compute_score([0, 1, 0, 1], [0, 1, 1, 1], "recall")
        self.assertGreaterEqual(p, 0.0)
        self.assertGreaterEqual(r, 0.0)

    def test_r2_perfect(self):
        self.assertAlmostEqual(compute_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "r2"), 1.0)

    def test_rmse_zero_when_perfect(self):
        self.assertEqual(compute_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "rmse"), 0.0)

    def test_rmse_nonzero(self):
        # MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 1 → RMSE = 1
        self.assertAlmostEqual(compute_score([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], "rmse"), 1.0)

    def test_mae(self):
        self.assertAlmostEqual(compute_score([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], "mae"), 1.0)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            compute_score([0, 1], [0, 1], "bogus_metric")


class TrainAndScoreTest(TestCase):
    def _classification_prepared(self):
        # Two well-separated clusters → easy classification problem
        np.random.seed(0)
        df = pd.DataFrame({
            "x1": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
            "x2": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
            "target": [0] * 50 + [1] * 50,
        })
        return prepare_experiment(df, "target", "classification", ExperimentConfig(stratify=False))

    def _regression_prepared(self):
        np.random.seed(0)
        x = np.random.randn(100)
        df = pd.DataFrame({"x": x, "y": x * 2.0 + 1.0 + np.random.randn(100) * 0.1})
        return prepare_experiment(df, "y", "regression", ExperimentConfig(stratify=False, scaling="none"))

    def test_classification_logreg_high_accuracy(self):
        prepared = self._classification_prepared()
        result = train_and_score(prepared, "logreg", "accuracy", random_seed=42)
        self.assertGreater(result.test_score, 0.85)
        self.assertGreater(result.train_score, 0.85)

    def test_classification_f1_metric(self):
        prepared = self._classification_prepared()
        result = train_and_score(prepared, "rf_clf", "f1", random_seed=42)
        self.assertGreater(result.test_score, 0.85)

    def test_regression_linreg_high_r2(self):
        prepared = self._regression_prepared()
        result = train_and_score(prepared, "linreg", "r2", random_seed=42)
        self.assertGreater(result.test_score, 0.95)

    def test_regression_rmse_low(self):
        prepared = self._regression_prepared()
        result = train_and_score(prepared, "linreg", "rmse", random_seed=42)
        self.assertLess(result.test_score, 0.5)

    def test_pipeline_bytes_nonempty(self):
        prepared = self._classification_prepared()
        result = train_and_score(prepared, "logreg", "accuracy", random_seed=42)
        self.assertGreater(len(result.pipeline_bytes), 0)

    def test_pickled_pipeline_roundtrips(self):
        # The whole Pipeline (preprocessor + estimator) is what's saved now,
        # and reloading it should yield byte-for-byte identical predictions.
        import joblib, io
        prepared = self._classification_prepared()
        result = train_and_score(prepared, "logreg", "accuracy", random_seed=42)
        loaded = joblib.load(io.BytesIO(result.pipeline_bytes))
        np.testing.assert_array_equal(
            loaded.predict(prepared.X_test),
            result.pipeline.predict(prepared.X_test),
        )

    def test_duration_is_recorded(self):
        prepared = self._classification_prepared()
        result = train_and_score(prepared, "logreg", "accuracy", random_seed=42)
        self.assertGreaterEqual(result.train_duration_ms, 0)


# ── Stage 6b: TrainedModel + train form + detail view integration ──────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class TrainedModelCreateViewTest(TestCase):
    def setUp(self):
        # 20 well-separated rows for stable training
        rows = []
        for i in range(10):
            rows.append(f"{i*0.1},{i*0.2},0")
        for i in range(10):
            rows.append(f"{5 + i*0.1},{5 + i*0.2},1")
        csv = ("a,b,target\n" + "\n".join(rows) + "\n").encode()
        uploaded = SimpleUploadedFile("clf.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "Exp1",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        self.experiment = Experiment.objects.first()

    def test_create_form_get_returns_200(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/models/new/")
        self.assertEqual(response.status_code, 200)

    def test_create_form_shows_classification_algorithms(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/models/new/")
        self.assertContains(response, "Logistic Regression")
        self.assertContains(response, "Random Forest")
        self.assertContains(response, "Accuracy")
        self.assertNotContains(response, "Linear Regression")
        self.assertNotContains(response, "RMSE")

    def test_post_creates_trained_model(self):
        from .models import TrainedModel
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "My Model",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        self.assertEqual(TrainedModel.objects.count(), 1)
        m = TrainedModel.objects.first()
        self.assertEqual(m.name, "My Model")
        self.assertEqual(m.experiment, self.experiment)

    def test_post_populates_scores_and_file(self):
        from .models import TrainedModel
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "Scored",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        m = TrainedModel.objects.first()
        self.assertTrue(m.is_trained)
        self.assertIsNotNone(m.train_score)
        self.assertIsNotNone(m.test_score)
        self.assertIsNotNone(m.train_duration_ms)
        self.assertTrue(m.model_file.name)
        self.assertGreater(m.model_file.size, 0)

    def test_post_auto_names_when_blank(self):
        from .models import TrainedModel
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "",
            "algorithm": "rf_clf",
            "metric": "f1",
        })
        m = TrainedModel.objects.first()
        self.assertIn("Random Forest", m.name)

    def test_post_redirects_to_model_detail(self):
        from .models import TrainedModel
        response = self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "Redir",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        m = TrainedModel.objects.first()
        self.assertRedirects(response, f"/project1/models/{m.pk}/")

    def test_model_detail_returns_200(self):
        from .models import TrainedModel
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "Det",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        m = TrainedModel.objects.first()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Det")

    def test_experiment_detail_lists_trained_models(self):
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "ListMe",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertContains(response, "ListMe")
        self.assertContains(response, "Logistic Regression")

    def test_train_button_visible_on_prepared_experiment(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertContains(response, "Train new model")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class TrainedModelRegressionFlowTest(TestCase):
    def test_regression_form_shows_regression_options(self):
        rows = "\n".join(f"{i*0.1},{i*0.2 + 1.0}" for i in range(40))
        csv = f"x,y\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("reg.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{dataset.pk}/experiments/new/", {
            "name": "ExpReg",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "none",
            "test_size": "0.2",
            "random_seed": "42",
            "stratify": "",  # not applicable for regression
        })
        from .models import Experiment
        experiment = Experiment.objects.first()

        response = self.client.get(f"/project1/experiments/{experiment.pk}/models/new/")
        self.assertContains(response, "Linear Regression")
        self.assertContains(response, "RMSE")
        self.assertNotContains(response, "Logistic Regression")
        self.assertNotContains(response, "Accuracy")


# ── Stage 6c: model delete + cleanup signal ────────────────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class TrainedModelDeleteTest(TestCase):
    def setUp(self):
        rows = []
        for i in range(10):
            rows.append(f"{i*0.1},{i*0.2},0")
        for i in range(10):
            rows.append(f"{5 + i*0.1},{5 + i*0.2},1")
        csv = ("a,b,target\n" + "\n".join(rows) + "\n").encode()
        uploaded = SimpleUploadedFile("clf2.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "ExpDel",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment
        self.experiment = Experiment.objects.first()
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": "ToDelete",
            "algorithm": "logreg",
            "metric": "accuracy",
        })
        from .models import TrainedModel
        self.model = TrainedModel.objects.first()

    def test_delete_confirm_page_returns_200(self):
        response = self.client.get(f"/project1/models/{self.model.pk}/delete/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ToDelete")

    def test_delete_post_removes_model(self):
        from .models import TrainedModel
        self.client.post(f"/project1/models/{self.model.pk}/delete/")
        self.assertEqual(TrainedModel.objects.count(), 0)

    def test_delete_post_redirects_to_experiment(self):
        response = self.client.post(f"/project1/models/{self.model.pk}/delete/")
        self.assertRedirects(response, f"/project1/experiments/{self.experiment.pk}/")

    def test_delete_removes_file_from_disk(self):
        file_path = self.model.model_file.path
        self.assertTrue(os.path.isfile(file_path))
        self.model.delete()
        self.assertFalse(os.path.isfile(file_path))

    def test_model_detail_shows_delete_button(self):
        response = self.client.get(f"/project1/models/{self.model.pk}/")
        self.assertContains(response, "Delete")

    def test_deleting_experiment_cascades_to_models(self):
        from .models import TrainedModel
        file_path = self.model.model_file.path
        self.experiment.delete()
        self.assertEqual(TrainedModel.objects.count(), 0)
        self.assertFalse(os.path.isfile(file_path))


# ── Stage 7a: evaluation service unit tests ────────────────────────────────

def _make_classification_prepared():
    np.random.seed(0)
    df = pd.DataFrame({
        "x1": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
        "x2": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
        "target": [0] * 50 + [1] * 50,
    })
    return prepare_experiment(df, "target", "classification", ExperimentConfig(stratify=False))


def _make_regression_prepared():
    np.random.seed(0)
    x = np.random.randn(100)
    df = pd.DataFrame({"x": x, "y": x * 2.0 + 1.0 + np.random.randn(100) * 0.1})
    return prepare_experiment(df, "y", "regression", ExperimentConfig(stratify=False, scaling="none"))


def _make_string_target_prepared():
    df = pd.DataFrame({
        "x1": list(range(20)),
        "x2": [i * 0.5 for i in range(20)],
        "target": ["cat", "dog"] * 10,
    })
    return prepare_experiment(df, "target", "classification", ExperimentConfig(stratify=False, scaling="none"))


class EvaluateClassificationTest(TestCase):
    def setUp(self):
        self.prepared = _make_classification_prepared()
        self.estimator = build_estimator("logreg", 42)
        self.estimator.fit(self.prepared.X_train, self.prepared.y_train)

    def test_problem_type_set(self):
        result = evaluate_classification(self.estimator, self.prepared, None)
        self.assertEqual(result["problem_type"], "classification")

    def test_all_metrics_present_train_and_test(self):
        result = evaluate_classification(self.estimator, self.prepared, None)
        for key in ("accuracy", "f1", "precision", "recall"):
            self.assertIn(key, result["train"])
            self.assertIn(key, result["test"])

    def test_metrics_are_floats(self):
        result = evaluate_classification(self.estimator, self.prepared, None)
        for v in result["test"].values():
            self.assertIsInstance(v, float)

    def test_confusion_matrix_is_k_by_k(self):
        result = evaluate_classification(self.estimator, self.prepared, None)
        cm = result["confusion_matrix"]
        n_classes = len(result["labels"])
        self.assertEqual(len(cm), n_classes)
        self.assertEqual(len(cm[0]), n_classes)

    def test_per_class_report_one_row_per_class(self):
        result = evaluate_classification(self.estimator, self.prepared, None)
        n_classes = len(result["labels"])
        self.assertEqual(len(result["per_class"]), n_classes)
        for cls in result["per_class"]:
            self.assertIn("precision", cls)
            self.assertIn("recall", cls)
            self.assertIn("f1", cls)
            self.assertIn("support", cls)
            self.assertIsInstance(cls["support"], int)

    def test_string_labels_decoded_in_output(self):
        prepared = _make_string_target_prepared()
        est = build_estimator("logreg", 42)
        est.fit(prepared.X_train, prepared.y_train)
        result = evaluate_classification(est, prepared, prepared.label_map)
        self.assertIn("cat", result["labels"])
        self.assertIn("dog", result["labels"])
        for sample in result["predictions_sample"]:
            self.assertIn(sample["y_true"], ("cat", "dog"))
            self.assertIn(sample["y_pred"], ("cat", "dog"))

    def test_predictions_sample_capped_at_500(self):
        # Synthesize a large dataset
        np.random.seed(0)
        n = 3000
        df = pd.DataFrame({
            "x1": np.random.randn(n),
            "target": np.random.randint(0, 2, n),
        })
        prepared = prepare_experiment(df, "target", "classification",
                                      ExperimentConfig(stratify=False))
        est = build_estimator("logreg", 42)
        est.fit(prepared.X_train, prepared.y_train)
        result = evaluate_classification(est, prepared, None)
        self.assertLessEqual(len(result["predictions_sample"]), 500)


class EvaluateRegressionTest(TestCase):
    def setUp(self):
        self.prepared = _make_regression_prepared()
        self.estimator = build_estimator("linreg", 42)
        self.estimator.fit(self.prepared.X_train, self.prepared.y_train)

    def test_problem_type_set(self):
        result = evaluate_regression(self.estimator, self.prepared)
        self.assertEqual(result["problem_type"], "regression")

    def test_all_metrics_present(self):
        result = evaluate_regression(self.estimator, self.prepared)
        for key in ("r2", "rmse", "mae"):
            self.assertIn(key, result["train"])
            self.assertIn(key, result["test"])

    def test_predictions_sample_has_required_keys(self):
        result = evaluate_regression(self.estimator, self.prepared)
        for sample in result["predictions_sample"]:
            self.assertIn("y_true", sample)
            self.assertIn("y_pred", sample)
            self.assertIsInstance(sample["y_true"], float)

    def test_residuals_match_y_true_minus_y_pred(self):
        result = evaluate_regression(self.estimator, self.prepared)
        for pred, res in zip(result["predictions_sample"], result["residuals_sample"]):
            expected = pred["y_true"] - pred["y_pred"]
            self.assertAlmostEqual(res["residual"], expected, places=10)

    def test_no_confusion_matrix_for_regression(self):
        result = evaluate_regression(self.estimator, self.prepared)
        self.assertNotIn("confusion_matrix", result)


class FeatureImportanceTest(TestCase):
    def setUp(self):
        self.prepared = _make_classification_prepared()

    def _fit(self, algo):
        est = build_estimator(algo, 42)
        est.fit(self.prepared.X_train, self.prepared.y_train)
        return est

    def test_random_forest_supported(self):
        result = compute_feature_importance(self._fit("rf_clf"), self.prepared.feature_names)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.prepared.feature_names))

    def test_decision_tree_supported(self):
        result = compute_feature_importance(self._fit("dt_clf"), self.prepared.feature_names)
        self.assertIsNotNone(result)

    def test_logistic_regression_supported(self):
        result = compute_feature_importance(self._fit("logreg"), self.prepared.feature_names)
        self.assertIsNotNone(result)

    def test_knn_returns_none(self):
        result = compute_feature_importance(self._fit("knn_clf"), self.prepared.feature_names)
        self.assertIsNone(result)

    def test_svm_rbf_returns_none(self):
        # SVC default kernel is rbf — no feature importance available
        result = compute_feature_importance(self._fit("svm"), self.prepared.feature_names)
        self.assertIsNone(result)

    def test_sorted_descending(self):
        result = compute_feature_importance(self._fit("rf_clf"), self.prepared.feature_names)
        values = [item["value"] for item in result]
        self.assertEqual(values, sorted(values, reverse=True))


class BuildEvaluationTest(TestCase):
    def test_classification_orchestration(self):
        prepared = _make_classification_prepared()
        est = build_estimator("rf_clf", 42)
        est.fit(prepared.X_train, prepared.y_train)
        result = build_evaluation(
            est, prepared, "classification", prepared.label_map, prepared.feature_names
        )
        self.assertEqual(result["problem_type"], "classification")
        self.assertIn("confusion_matrix", result)
        self.assertIn("feature_importance", result)
        self.assertIsNotNone(result["feature_importance"])

    def test_regression_orchestration(self):
        prepared = _make_regression_prepared()
        est = build_estimator("linreg", 42)
        est.fit(prepared.X_train, prepared.y_train)
        result = build_evaluation(
            est, prepared, "regression", None, prepared.feature_names
        )
        self.assertEqual(result["problem_type"], "regression")
        self.assertIn("predictions_sample", result)
        self.assertIn("residuals_sample", result)

    def test_unknown_problem_type_raises(self):
        prepared = _make_regression_prepared()
        est = build_estimator("linreg", 42)
        est.fit(prepared.X_train, prepared.y_train)
        with self.assertRaises(ValueError):
            build_evaluation(est, prepared, "bogus", None, prepared.feature_names)

    def test_evaluation_is_json_serializable(self):
        import json
        prepared = _make_classification_prepared()
        est = build_estimator("rf_clf", 42)
        est.fit(prepared.X_train, prepared.y_train)
        result = build_evaluation(
            est, prepared, "classification", prepared.label_map, prepared.feature_names
        )
        # Should not raise
        json.dumps(result)


# ── Stage 7b: persisted evaluation + model detail UI ───────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class PersistedEvaluationTest(TestCase):
    def _train_classification_model(self):
        rows = []
        for i in range(15):
            rows.append(f"{i*0.1},{i*0.2},0")
        for i in range(15):
            rows.append(f"{5 + i*0.1},{5 + i*0.2},1")
        csv = ("a,b,target\n" + "\n".join(rows) + "\n").encode()
        uploaded = SimpleUploadedFile("eval_clf.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{dataset.pk}/experiments/new/", {
            "name": "ExpEval",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        from .models import Experiment, TrainedModel
        experiment = Experiment.objects.first()
        self.client.post(f"/project1/experiments/{experiment.pk}/models/new/", {
            "name": "EvalRF",
            "algorithm": "rf_clf",
            "metric": "f1",
        })
        return TrainedModel.objects.first()

    def _train_regression_model(self):
        rows = "\n".join(f"{i*0.1},{i*0.2 + 1.0}" for i in range(40))
        csv = f"x,y\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("eval_reg.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{dataset.pk}/experiments/new/", {
            "name": "ExpRegEval",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "none",
            "test_size": "0.2",
            "random_seed": "42",
            "stratify": "",
        })
        from .models import Experiment, TrainedModel
        experiment = Experiment.objects.first()
        self.client.post(f"/project1/experiments/{experiment.pk}/models/new/", {
            "name": "EvalLinReg",
            "algorithm": "linreg",
            "metric": "rmse",
        })
        return TrainedModel.objects.first()

    def test_evaluation_is_persisted_after_classification_training(self):
        m = self._train_classification_model()
        self.assertIsNotNone(m.evaluation)
        self.assertEqual(m.evaluation["problem_type"], "classification")

    def test_evaluation_contains_all_classification_metrics(self):
        m = self._train_classification_model()
        for key in ("accuracy", "f1", "precision", "recall"):
            self.assertIn(key, m.evaluation["test"])

    def test_evaluation_is_persisted_after_regression_training(self):
        m = self._train_regression_model()
        self.assertIsNotNone(m.evaluation)
        self.assertEqual(m.evaluation["problem_type"], "regression")
        for key in ("r2", "rmse", "mae"):
            self.assertIn(key, m.evaluation["test"])

    def test_classification_detail_shows_all_metrics(self):
        m = self._train_classification_model()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "Accuracy")
        self.assertContains(response, "F1 (weighted)")
        self.assertContains(response, "Precision (weighted)")
        self.assertContains(response, "Recall (weighted)")

    def test_classification_detail_shows_confusion_matrix(self):
        m = self._train_classification_model()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "Confusion matrix")

    def test_classification_detail_shows_per_class_report(self):
        m = self._train_classification_model()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "Per-class report")
        self.assertContains(response, "Support")

    def test_regression_detail_does_not_show_confusion_matrix(self):
        m = self._train_regression_model()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertNotContains(response, "Confusion matrix")
        self.assertNotContains(response, "Per-class report")

    def test_regression_detail_shows_regression_metrics(self):
        m = self._train_regression_model()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "R²")
        self.assertContains(response, "RMSE")
        self.assertContains(response, "MAE")

    def test_trained_metric_is_highlighted(self):
        m = self._train_classification_model()  # trained with f1
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "trained")

    def test_no_evaluation_for_old_model_shows_message(self):
        from .models import Experiment, TrainedModel
        # Train one normally to set up
        m = self._train_classification_model()
        # Simulate an old model with no evaluation
        m.evaluation = None
        m.save()
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "Re-train")


# ── Stage 7c: Chart.js plots on model detail ───────────────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ModelDetailPlotsTest(TestCase):
    def _train(self, algorithm, metric, problem_type="classification"):
        if problem_type == "classification":
            rows = []
            for i in range(15):
                rows.append(f"{i*0.1},{i*0.2},0")
            for i in range(15):
                rows.append(f"{5 + i*0.1},{5 + i*0.2},1")
            csv = ("a,b,target\n" + "\n".join(rows) + "\n").encode()
            scaling = "standard"
            stratify = "on"
        else:
            rows = "\n".join(f"{i*0.1},{i*0.2 + 1.0}" for i in range(40))
            csv = f"x,y\n{rows}\n".encode()
            scaling = "none"
            stratify = ""

        uploaded = SimpleUploadedFile("plots.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        from .models import Dataset, Experiment, TrainedModel
        dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{dataset.pk}/experiments/new/", {
            "name": "ExpPlots",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": scaling,
            "test_size": "0.2",
            "random_seed": "42",
            "stratify": stratify,
        })
        experiment = Experiment.objects.first()
        self.client.post(f"/project1/experiments/{experiment.pk}/models/new/", {
            "name": "PlotModel",
            "algorithm": algorithm,
            "metric": metric,
        })
        return TrainedModel.objects.first()

    def test_classification_detail_has_confusion_matrix_canvas(self):
        m = self._train("rf_clf", "f1", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, 'id="cm-chart"')

    def test_classification_detail_does_not_have_regression_canvases(self):
        m = self._train("rf_clf", "f1", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertNotContains(response, 'id="pva-chart"')
        self.assertNotContains(response, 'id="res-chart"')

    def test_regression_detail_has_pred_vs_actual_canvas(self):
        m = self._train("linreg", "rmse", "regression")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, 'id="pva-chart"')

    def test_regression_detail_has_residuals_canvas(self):
        m = self._train("linreg", "rmse", "regression")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, 'id="res-chart"')

    def test_regression_detail_does_not_have_confusion_matrix_canvas(self):
        m = self._train("linreg", "rmse", "regression")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertNotContains(response, 'id="cm-chart"')

    def test_feature_importance_canvas_for_random_forest(self):
        m = self._train("rf_clf", "f1", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, 'id="fi-chart"')

    def test_feature_importance_not_available_for_knn(self):
        m = self._train("knn_clf", "accuracy", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "Not available")
        self.assertNotContains(response, 'id="fi-chart"')

    def test_chart_js_scripts_loaded(self):
        m = self._train("rf_clf", "f1", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, "chart.js")
        self.assertContains(response, "chartjs-chart-matrix")
        self.assertContains(response, "model_chart.js")

    def test_evaluation_json_embedded(self):
        m = self._train("rf_clf", "f1", "classification")
        response = self.client.get(f"/project1/models/{m.pk}/")
        self.assertContains(response, 'id="model-evaluation"')
        # The matrix data should be in the JSON
        self.assertContains(response, "confusion_matrix")


# ── Stage 7d: experiment compare view ──────────────────────────────────────

@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ExperimentCompareTest(TestCase):
    def setUp(self):
        rows = []
        for i in range(15):
            rows.append(f"{i*0.1},{i*0.2},0")
        for i in range(15):
            rows.append(f"{5 + i*0.1},{5 + i*0.2},1")
        csv = ("a,b,target\n" + "\n".join(rows) + "\n").encode()
        uploaded = SimpleUploadedFile("compare.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        from .models import Dataset, Experiment
        self.dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{self.dataset.pk}/experiments/new/", {
            "name": "ExpCmp",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "standard",
            "test_size": "0.2",
            "random_seed": "42",
        })
        self.experiment = Experiment.objects.first()

    def _train(self, name, algorithm, metric):
        self.client.post(f"/project1/experiments/{self.experiment.pk}/models/new/", {
            "name": name,
            "algorithm": algorithm,
            "metric": metric,
        })

    def test_compare_view_returns_200_with_zero_models(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No trained models")

    def test_compare_view_lists_models(self):
        self._train("LogReg M", "logreg", "accuracy")
        self._train("RF M", "rf_clf", "f1")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/")
        self.assertContains(response, "LogReg M")
        self.assertContains(response, "RF M")

    def test_compare_view_shows_all_metrics(self):
        self._train("M1", "logreg", "accuracy")
        self._train("M2", "rf_clf", "f1")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/")
        for label in ("Accuracy", "F1 (weighted)", "Precision", "Recall"):
            self.assertContains(response, label)

    def test_compare_view_marks_best_metric(self):
        self._train("M1", "logreg", "accuracy")
        self._train("M2", "rf_clf", "f1")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/")
        # Best cell should appear (★ marker)
        self.assertContains(response, "★")

    def test_compare_button_hidden_with_zero_models(self):
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertNotContains(response, "Compare models")

    def test_compare_button_hidden_with_one_model(self):
        self._train("Only", "logreg", "accuracy")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertNotContains(response, "Compare models")

    def test_compare_button_visible_with_two_models(self):
        self._train("A", "logreg", "accuracy")
        self._train("B", "rf_clf", "f1")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/")
        self.assertContains(response, "Compare models")

    def test_compare_view_has_chart_canvas_and_payload(self):
        self._train("A", "logreg", "accuracy")
        self._train("B", "rf_clf", "accuracy")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/")
        self.assertContains(response, 'id="compare-chart"')
        self.assertContains(response, 'id="compare-payload"')
        self.assertContains(response, "compare_chart.js")

    def test_compare_view_metric_param_persisted(self):
        self._train("A", "logreg", "accuracy")
        self._train("B", "rf_clf", "accuracy")
        response = self.client.get(f"/project1/experiments/{self.experiment.pk}/compare/?metric=f1")
        # Selected metric should be f1
        self.assertContains(response, '<option value="f1" selected')


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ExperimentCompareRegressionTest(TestCase):
    def test_regression_compare_shows_regression_metrics(self):
        rows = "\n".join(f"{i*0.1},{i*0.2 + 1.0}" for i in range(40))
        csv = f"x,y\n{rows}\n".encode()
        uploaded = SimpleUploadedFile("compreg.csv", csv, content_type="text/csv")
        self.client.post("/project1/datasets/upload/", {"file": uploaded})
        from .models import Dataset, Experiment
        dataset = Dataset.objects.first()
        self.client.post(f"/project1/datasets/{dataset.pk}/experiments/new/", {
            "name": "RegCmp",
            "missing_strategy": "mean_mode",
            "categorical_encoding": "onehot",
            "scaling": "none",
            "test_size": "0.2",
            "random_seed": "42",
            "stratify": "",
        })
        experiment = Experiment.objects.first()
        for name, algo in [("LR", "linreg"), ("RF", "rf_reg")]:
            self.client.post(f"/project1/experiments/{experiment.pk}/models/new/", {
                "name": name, "algorithm": algo, "metric": "r2",
            })
        response = self.client.get(f"/project1/experiments/{experiment.pk}/compare/")
        self.assertContains(response, "R²")
        self.assertContains(response, "RMSE")
        self.assertContains(response, "MAE")
        self.assertNotContains(response, "Accuracy")


# ── Stage 8a: pipeline construction unit tests ─────────────────────────────

class BuildPreprocessingTest(TestCase):
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        ct = build_preprocessing(ExperimentConfig(), ["a", "b"], ["c"])
        self.assertIsInstance(ct, ColumnTransformer)

    def test_numeric_only_has_one_branch(self):
        ct = build_preprocessing(ExperimentConfig(), ["a", "b"], [])
        self.assertEqual(len(ct.transformers), 1)
        self.assertEqual(ct.transformers[0][0], "num")

    def test_categorical_only_has_one_branch(self):
        ct = build_preprocessing(ExperimentConfig(), [], ["c"])
        self.assertEqual(len(ct.transformers), 1)
        self.assertEqual(ct.transformers[0][0], "cat")

    def test_both_branches_when_both_column_types_present(self):
        ct = build_preprocessing(ExperimentConfig(), ["a"], ["b"])
        names = [t[0] for t in ct.transformers]
        self.assertIn("num", names)
        self.assertIn("cat", names)

    def test_drop_encoding_excludes_categorical_branch(self):
        ct = build_preprocessing(
            ExperimentConfig(categorical_encoding="drop"), ["a"], ["b"]
        )
        names = [t[0] for t in ct.transformers]
        self.assertIn("num", names)
        self.assertNotIn("cat", names)

    def test_onehot_encoder_present_for_onehot_strategy(self):
        ct = build_preprocessing(
            ExperimentConfig(categorical_encoding="onehot"), [], ["c"]
        )
        from sklearn.preprocessing import OneHotEncoder
        cat_pipe = ct.transformers[0][1]
        encoder = dict(cat_pipe.steps).get("encoder")
        self.assertIsInstance(encoder, OneHotEncoder)

    def test_ordinal_encoder_present_for_label_strategy(self):
        ct = build_preprocessing(
            ExperimentConfig(categorical_encoding="label"), [], ["c"]
        )
        from sklearn.preprocessing import OrdinalEncoder
        cat_pipe = ct.transformers[0][1]
        encoder = dict(cat_pipe.steps).get("encoder")
        self.assertIsInstance(encoder, OrdinalEncoder)

    def test_standard_scaler_present_for_standard_scaling(self):
        ct = build_preprocessing(
            ExperimentConfig(scaling="standard"), ["x"], []
        )
        from sklearn.preprocessing import StandardScaler
        num_pipe = ct.transformers[0][1]
        scaler = dict(num_pipe.steps).get("scaler")
        self.assertIsInstance(scaler, StandardScaler)

    def test_minmax_scaler_present_for_minmax_scaling(self):
        ct = build_preprocessing(
            ExperimentConfig(scaling="minmax"), ["x"], []
        )
        from sklearn.preprocessing import MinMaxScaler
        num_pipe = ct.transformers[0][1]
        scaler = dict(num_pipe.steps).get("scaler")
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_no_scaler_when_scaling_none(self):
        ct = build_preprocessing(
            ExperimentConfig(scaling="none", missing_strategy="drop"), ["x"], []
        )
        # With missing="drop" + scaling="none", the numeric branch is just passthrough.
        num_pipe = ct.transformers[0][1]
        step_names = [name for name, _ in num_pipe.steps]
        self.assertNotIn("scaler", step_names)

    def test_imputer_present_for_mean_mode(self):
        ct = build_preprocessing(
            ExperimentConfig(missing_strategy="mean_mode"), ["x"], ["c"]
        )
        from sklearn.impute import SimpleImputer
        num_pipe = ct.transformers[0][1]
        imputer = dict(num_pipe.steps).get("imputer")
        self.assertIsInstance(imputer, SimpleImputer)
        self.assertEqual(imputer.strategy, "mean")
        cat_pipe = ct.transformers[1][1]
        cat_imputer = dict(cat_pipe.steps).get("imputer")
        self.assertEqual(cat_imputer.strategy, "most_frequent")

    def test_imputer_with_zero_empty_strategy(self):
        ct = build_preprocessing(
            ExperimentConfig(missing_strategy="zero_empty"), ["x"], ["c"]
        )
        num_pipe = ct.transformers[0][1]
        imputer = dict(num_pipe.steps).get("imputer")
        self.assertEqual(imputer.strategy, "constant")
        self.assertEqual(imputer.fill_value, 0)

    def test_fit_transform_smoke(self):
        # ColumnTransformer should fit + transform a small DataFrame without error.
        ct = build_preprocessing(ExperimentConfig(), ["x1"], ["c1"])
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, float("nan")],
            "c1": ["a", "b", "a", "b"],
        })
        out = ct.fit_transform(df)
        self.assertEqual(out.shape[0], 4)


class BuildFullPipelineTest(TestCase):
    def test_returns_pipeline_with_preprocessor_and_estimator(self):
        from sklearn.pipeline import Pipeline as SkPipeline
        ct = build_preprocessing(ExperimentConfig(), ["a"], [])
        from sklearn.linear_model import LogisticRegression
        pipe = build_full_pipeline(ct, LogisticRegression())
        self.assertIsInstance(pipe, SkPipeline)
        self.assertEqual([n for n, _ in pipe.steps], ["preprocessor", "estimator"])

    def test_end_to_end_classification_on_two_cluster_data(self):
        # Two well-separated clusters → near-perfect classification accuracy.
        np.random.seed(0)
        df = pd.DataFrame({
            "x1": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
            "x2": np.concatenate([np.random.randn(50), np.random.randn(50) + 5]),
        })
        y = np.array([0] * 50 + [1] * 50)

        ct = build_preprocessing(ExperimentConfig(), ["x1", "x2"], [])
        from sklearn.linear_model import LogisticRegression
        pipe = build_full_pipeline(ct, LogisticRegression(random_state=42, max_iter=1000))
        pipe.fit(df, y)
        accuracy = (pipe.predict(df) == y).mean()
        self.assertGreater(accuracy, 0.9)

    def test_pipeline_accepts_mixed_dtype_dataframe(self):
        # Confirm the pipeline can fit + predict on a DataFrame with both
        # numeric and categorical columns (the main motivation for the refactor).
        ct = build_preprocessing(
            ExperimentConfig(categorical_encoding="onehot", scaling="standard"),
            ["x1"], ["cat"],
        )
        from sklearn.linear_model import LogisticRegression
        pipe = build_full_pipeline(ct, LogisticRegression(random_state=42))
        df = pd.DataFrame({
            "x1": [float(i) for i in range(10)],
            "cat": ["a", "b"] * 5,
        })
        y = np.array([0, 1] * 5)
        pipe.fit(df, y)
        # Should run without error
        pipe.predict(df)

    def test_pipeline_predict_handles_unseen_category(self):
        # OneHotEncoder with handle_unknown='ignore' should let an unseen
        # category come through at predict time without crashing.
        ct = build_preprocessing(
            ExperimentConfig(categorical_encoding="onehot"),
            ["x1"], ["cat"],
        )
        from sklearn.linear_model import LogisticRegression
        pipe = build_full_pipeline(ct, LogisticRegression(random_state=42))
        df_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
        df_pred  = pd.DataFrame({"x1": [5.0],                 "cat": ["c"]})  # unseen
        pipe.fit(df_train, np.array([0, 1, 0, 1]))
        pipe.predict(df_pred)  # must not raise

    def test_serialization_roundtrip_preserves_predictions(self):
        # joblib-pickle the whole pipeline, reload, predictions must be identical.
        import joblib, io
        np.random.seed(0)
        df = pd.DataFrame({
            "x1": np.concatenate([np.random.randn(30), np.random.randn(30) + 4]),
        })
        y = np.array([0] * 30 + [1] * 30)
        ct = build_preprocessing(ExperimentConfig(), ["x1"], [])
        from sklearn.linear_model import LogisticRegression
        pipe = build_full_pipeline(ct, LogisticRegression(random_state=42))
        pipe.fit(df, y)
        before = pipe.predict(df)

        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        loaded = joblib.load(io.BytesIO(buf.getvalue()))
        after = loaded.predict(df)
        np.testing.assert_array_equal(before, after)


