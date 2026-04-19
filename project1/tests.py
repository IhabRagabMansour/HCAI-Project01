import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from .models import Dataset


SAMPLE_CSV = b"feature1,feature2,target\n1.0,2.0,0\n3.0,4.0,1\n"


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DatasetSignalTest(TestCase):
    def test_file_deleted_from_disk_on_dataset_delete(self):
        uploaded = SimpleUploadedFile("test.csv", SAMPLE_CSV, content_type="text/csv")
        dataset = Dataset.objects.create(original_name="test.csv", file=uploaded)

        file_path = dataset.file.path
        self.assertTrue(os.path.isfile(file_path), "File should exist after upload")

        dataset.delete()
        self.assertFalse(os.path.isfile(file_path), "File should be removed after dataset delete")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class DatasetUploadViewTest(TestCase):
    def test_upload_valid_csv(self):
        uploaded = SimpleUploadedFile("iris.csv", SAMPLE_CSV, content_type="text/csv")
        response = self.client.post(
            "/project1/datasets/upload/",
            {"file": uploaded},
        )
        self.assertEqual(Dataset.objects.count(), 1)
        dataset = Dataset.objects.first()
        self.assertEqual(dataset.original_name, "iris.csv")
        self.assertRedirects(response, f"/project1/datasets/{dataset.pk}/")

    def test_upload_rejects_non_csv_extension(self):
        uploaded = SimpleUploadedFile("data.txt", SAMPLE_CSV, content_type="text/plain")
        response = self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.assertEqual(Dataset.objects.count(), 0)
        self.assertContains(response, "csv extension")

    def test_upload_rejects_empty_file(self):
        uploaded = SimpleUploadedFile("empty.csv", b"", content_type="text/csv")
        response = self.client.post("/project1/datasets/upload/", {"file": uploaded})
        self.assertEqual(Dataset.objects.count(), 0)

    def test_dataset_list_view(self):
        response = self.client.get("/project1/datasets/")
        self.assertEqual(response.status_code, 200)

    def test_dataset_detail_returns_404_for_missing(self):
        response = self.client.get("/project1/datasets/999/")
        self.assertEqual(response.status_code, 404)
