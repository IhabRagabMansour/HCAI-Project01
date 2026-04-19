from django.apps import AppConfig


class Project1Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "project1"

    def ready(self):
        from . import signals  # noqa: F401
