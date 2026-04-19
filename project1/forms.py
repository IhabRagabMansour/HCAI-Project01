from django import forms
from .models import Dataset

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
            head = f.read(1024).decode("utf-8-sig")
            f.seek(0)
        except UnicodeDecodeError:
            raise forms.ValidationError("File is not valid UTF-8 encoded text.")

        if not any(sep in head for sep in (",", ";", "\t")):
            raise forms.ValidationError("File does not appear to contain CSV-style delimiters.")

        return f

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.original_name = self.cleaned_data["file"].name
        if commit:
            instance.save()
        return instance
