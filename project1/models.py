from django.db import models
from django.utils import timezone


def dataset_upload_path(instance, filename):
    today = timezone.now().strftime("%Y-%m-%d")
    return f"datasets/{today}/{filename}"


class Dataset(models.Model):
    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.original_name} ({self.uploaded_at:%Y-%m-%d %H:%M})"

    def filesize_kb(self):
        try:
            return round(self.file.size / 1024, 1)
        except (FileNotFoundError, ValueError):
            return None
