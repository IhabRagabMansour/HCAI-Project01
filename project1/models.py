from django.db import models
from django.utils import timezone


def dataset_upload_path(instance, filename):
    today = timezone.now().strftime("%Y-%m-%d")
    return f"datasets/{today}/{filename}"


PROBLEM_TYPE_CHOICES = [
    ("classification", "Classification"),
    ("regression", "Regression"),
    ("unknown", "Unknown"),
]


class Dataset(models.Model):
    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Metadata populated at upload time by the parsing service
    n_rows = models.PositiveIntegerField(null=True, blank=True)
    n_columns = models.PositiveIntegerField(null=True, blank=True)
    columns = models.JSONField(null=True, blank=True)       # [{name, dtype}, ...]
    head_preview = models.JSONField(null=True, blank=True)  # list of rows (10 max)
    target_name = models.CharField(max_length=255, null=True, blank=True)
    problem_type = models.CharField(
        max_length=20, choices=PROBLEM_TYPE_CHOICES, null=True, blank=True
    )
    parse_error = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.original_name} ({self.uploaded_at:%Y-%m-%d %H:%M})"

    def filesize_kb(self):
        try:
            return round(self.file.size / 1024, 1)
        except (FileNotFoundError, ValueError):
            return None

    @property
    def is_parsed(self):
        return self.parse_error is None and self.n_rows is not None
