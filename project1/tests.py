import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from .models import Dataset
from .services.data import humanize_dtype, infer_problem_type, extract_metadata

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
