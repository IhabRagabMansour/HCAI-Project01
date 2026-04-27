"""Clear all existing TrainedModel rows.

The Stage 8 pipeline refactor changed what model_file contains: it used to be
a joblib pickle of the bare sklearn estimator, now it's the full Pipeline
(preprocessor + estimator). Old files would fail to load correctly via
pipeline.predict(raw_X) because the saved object has no preprocessing step.

Rather than carry a compatibility shim, this migration deletes all
pre-refactor TrainedModel rows and their .joblib files. Users will re-train.
"""
import os

from django.db import migrations


def clear_old_trained_models(apps, schema_editor):
    TrainedModel = apps.get_model("project1", "TrainedModel")
    for m in TrainedModel.objects.all():
        if m.model_file:
            try:
                path = m.model_file.path
                if os.path.isfile(path):
                    os.remove(path)
            except (ValueError, NotImplementedError, FileNotFoundError):
                pass
    TrainedModel.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("project1", "0005_trainedmodel_eval_error_trainedmodel_evaluation"),
    ]
    operations = [
        migrations.RunPython(
            clear_old_trained_models,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
