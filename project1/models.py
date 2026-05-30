from django.db import models
from django.utils import timezone


def dataset_upload_path(instance, filename):
    today = timezone.now().strftime("%Y-%m-%d")
    return f"datasets/{today}/{filename}"


def trained_model_upload_path(instance, filename):
    today = timezone.now().strftime("%Y-%m-%d")
    return f"trained_models/{today}/{filename}"


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
    excluded_columns = models.JSONField(default=list, blank=True)

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
            excluded_columns=list(self.excluded_columns or []),
        )


class TrainedModel(models.Model):
    CLASSIFICATION_ALGOS = [
        ("logreg",  "Logistic Regression"),
        ("rf_clf",  "Random Forest"),
        ("svm",     "Support Vector Machine"),
        ("knn_clf", "K-Nearest Neighbors"),
        ("dt_clf",  "Decision Tree"),
    ]
    REGRESSION_ALGOS = [
        ("linreg",  "Linear Regression"),
        ("rf_reg",  "Random Forest"),
        ("svr",     "Support Vector Regressor"),
        ("knn_reg", "K-Nearest Neighbors"),
        ("dt_reg",  "Decision Tree"),
    ]
    CLASSIFICATION_METRICS = [
        ("accuracy",  "Accuracy"),
        ("f1",        "F1 (weighted)"),
        ("precision", "Precision (weighted)"),
        ("recall",    "Recall (weighted)"),
    ]
    REGRESSION_METRICS = [
        ("r2",   "R²"),
        ("rmse", "RMSE"),
        ("mae",  "MAE"),
    ]
    ALL_ALGOS = CLASSIFICATION_ALGOS + REGRESSION_ALGOS
    ALL_METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="models"
    )
    name = models.CharField(max_length=255, blank=True)
    algorithm = models.CharField(max_length=20, choices=ALL_ALGOS)
    metric = models.CharField(max_length=20, choices=ALL_METRICS)
    created_at = models.DateTimeField(auto_now_add=True)

    model_file = models.FileField(
        upload_to=trained_model_upload_path, null=True, blank=True
    )
    train_score = models.FloatField(null=True, blank=True)
    test_score = models.FloatField(null=True, blank=True)
    train_duration_ms = models.PositiveIntegerField(null=True, blank=True)
    train_error = models.TextField(null=True, blank=True)

    evaluation = models.JSONField(null=True, blank=True)
    eval_error = models.TextField(null=True, blank=True)

    hyperparameters = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name or f"Model #{self.pk}"

    @property
    def is_trained(self):
        return self.train_error is None and self.train_score is not None

    @property
    def algorithm_display(self):
        return dict(self.ALL_ALGOS).get(self.algorithm, self.algorithm)

    @property
    def metric_display(self):
        return dict(self.ALL_METRICS).get(self.metric, self.metric)

    @property
    def model_file_size_kb(self):
        try:
            return round(self.model_file.size / 1024, 1)
        except (FileNotFoundError, ValueError, AttributeError):
            return None
