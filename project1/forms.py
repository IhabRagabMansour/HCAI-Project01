from django import forms
from .models import Dataset, Experiment, TrainedModel

MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ["file"]
        labels = {"file": "Select a CSV file"}

    def clean_file(self):
        f = self.cleaned_data["file"]

        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("File must have a .csv extension.")

        if f.size == 0:
            raise forms.ValidationError("Uploaded file is empty.")

        if f.size > MAX_UPLOAD_SIZE_BYTES:
            raise forms.ValidationError(
                f"File exceeds the {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB limit."
            )

        try:
            raw_head = f.read(1024)
            f.seek(0)
            try:
                head = raw_head.decode("utf-8-sig")
            except UnicodeDecodeError:
                head = raw_head.decode("latin-1")
        except UnicodeDecodeError:
            raise forms.ValidationError("File encoding is not supported.")

        if not any(sep in head for sep in (",", ";", "\t")):
            raise forms.ValidationError("File does not appear to contain CSV-style delimiters.")

        return f

    def save(self, commit=True):
        from .services.data import read_csv_safely, extract_metadata

        instance = super().save(commit=False)
        instance.original_name = self.cleaned_data["file"].name

        # Save first so FileField is written to disk and file.path is available
        if commit:
            instance.save()

        try:
            df = read_csv_safely(instance.file)
            meta = extract_metadata(df)
            instance.n_rows = meta["n_rows"]
            instance.n_columns = meta["n_columns"]
            instance.columns = meta["columns"]
            instance.head_preview = meta["head"]
            instance.target_name = meta["target_name"]
            instance.problem_type = meta["problem_type"]
            instance.parse_error = None
        except ValueError as e:
            instance.parse_error = str(e)

        if commit:
            instance.save()

        return instance


class ExperimentForm(forms.ModelForm):
    class Meta:
        model = Experiment
        fields = [
            "name", "missing_strategy", "categorical_encoding",
            "scaling", "test_size", "random_seed", "stratify", "oversampling",
        ]
        widgets = {
            "test_size": forms.NumberInput(attrs={"step": "0.05", "min": "0.10", "max": "0.50"}),
            "random_seed": forms.NumberInput(attrs={"min": "0"}),
            "name": forms.TextInput(attrs={"placeholder": "Leave blank to auto-name"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # oversampling is optional — falls back to model default "none" if blank
        self.fields["oversampling"].required = False

    def clean_test_size(self):
        v = self.cleaned_data["test_size"]
        if not (0.10 <= v <= 0.50):
            raise forms.ValidationError("Test size must be between 0.10 and 0.50.")
        return v

    def clean_oversampling(self):
        v = self.cleaned_data.get("oversampling")
        return v or "none"


class TrainedModelForm(forms.ModelForm):
    class Meta:
        model = TrainedModel
        fields = ["name", "algorithm", "metric", "cv_folds"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Leave blank to auto-name"}),
            "cv_folds": forms.NumberInput(attrs={"min": "0", "max": "10", "step": "1"}),
        }

    def __init__(self, *args, problem_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        if problem_type == "classification":
            self.fields["algorithm"].choices = TrainedModel.CLASSIFICATION_ALGOS
            self.fields["metric"].choices    = TrainedModel.CLASSIFICATION_METRICS
        elif problem_type == "regression":
            self.fields["algorithm"].choices = TrainedModel.REGRESSION_ALGOS
            self.fields["metric"].choices    = TrainedModel.REGRESSION_METRICS

        # cv_folds is optional in the form — falls back to 5 if blank
        self.fields["cv_folds"].required = False
        self.fields["cv_folds"].initial = 5

    def clean_cv_folds(self):
        v = self.cleaned_data.get("cv_folds")
        if v is None:
            return 5
        if v == 1 or v > 10:
            raise forms.ValidationError("CV folds must be 0 (skip) or 2–10.")
        return v
