import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import Dataset, TrainedModel


@receiver(post_delete, sender=Dataset)
def delete_file_on_dataset_delete(sender, instance, **kwargs):
    if instance.file and os.path.isfile(instance.file.path):
        os.remove(instance.file.path)


@receiver(post_delete, sender=TrainedModel)
def delete_file_on_trained_model_delete(sender, instance, **kwargs):
    if instance.model_file:
        try:
            path = instance.model_file.path
        except (ValueError, NotImplementedError):
            return
        if os.path.isfile(path):
            os.remove(path)
