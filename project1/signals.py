import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import Dataset


@receiver(post_delete, sender=Dataset)
def delete_file_on_dataset_delete(sender, instance, **kwargs):
    if instance.file and os.path.isfile(instance.file.path):
        os.remove(instance.file.path)
