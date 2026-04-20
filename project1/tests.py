import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from .models import Dataset
from .services.data import humanize_dtype, infer_problem_type, extract_metadata
from .services.preprocess import (
    ExperimentConfig, handle_missing, encode_categorical,
    encode_target, scale_features, split_train_test, prepare_experiment,
)

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

    def test_no_nan_in_output(self):
        df = self.df.copy()
        df.loc[0, "x1"] = float("nan")
        result = prepare_experiment(df, "target", "classification", self.config)
        self.assertFalse(np.isnan(result.X_train).any())
        self.assertFalse(np.isnan(result.X_test).any())

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
