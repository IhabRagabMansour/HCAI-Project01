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

    @property
    def target_dtype(self):
        """Return the humanized dtype of the target column, or None."""
        if not self.columns or not self.target_name:
            return None
        for col in self.columns:
            if col["name"] == self.target_name:
                return col["dtype"]
        return None


class Experiment(models.Model):
    MISSING_CHOICES = [
        ("mean_mode", "Mean / Mode"),
        ("drop", "Drop rows with missing values"),
        ("zero_empty", "Fill with 0 / empty string"),
    ]
    ENCODING_CHOICES = [
        ("onehot", "One-hot encoding"),
        ("label", "Label encoding"),
        ("drop", "Drop categorical columns"),
    ]
    SCALING_CHOICES = [
        ("standard", "Standardize (z-score)"),
        ("minmax", "Min-max (0–1)"),
        ("none", "None"),
    ]

    dataset = models.ForeignKey(
        Dataset, on_delete=models.CASCADE, related_name="experiments"
    )
    name = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Preprocessing config
    missing_strategy = models.CharField(
        max_length=20, choices=MISSING_CHOICES, default="mean_mode"
    )
    categorical_encoding = models.CharField(
        max_length=20, choices=ENCODING_CHOICES, default="onehot"
    )
    scaling = models.CharField(
        max_length=20, choices=SCALING_CHOICES, default="standard"
    )
    test_size = models.FloatField(default=0.2)
    random_seed = models.PositiveIntegerField(default=42)
    stratify = models.BooleanField(default=True)

    # Results (populated by prepare_experiment)
    n_train = models.PositiveIntegerField(null=True, blank=True)
    n_test = models.PositiveIntegerField(null=True, blank=True)
    n_features_before = models.PositiveIntegerField(null=True, blank=True)
    n_features_after = models.PositiveIntegerField(null=True, blank=True)
    feature_names = models.JSONField(null=True, blank=True)
    stratify_used = models.BooleanField(null=True, blank=True)
    prepare_error = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name or f"Experiment #{self.pk}"

    @property
    def is_prepared(self):
        return self.prepare_error is None and self.n_train is not None

    @property
    def test_size_pct(self):
        return round(self.test_size * 100)

    @property
    def has_high_cardinality_warning(self):
        return (
            self.is_prepared
            and self.categorical_encoding == "onehot"
            and self.n_features_after is not None
            and self.n_features_after > 50
        )

    def as_config(self):
        from .services.preprocess import ExperimentConfig
        return ExperimentConfig(
            missing_strategy=self.missing_strategy,
            categorical_encoding=self.categorical_encoding,
            scaling=self.scaling,
            test_size=self.test_size,
            random_seed=self.random_seed,
            stratify=self.stratify,
        )
