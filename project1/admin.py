from django.contrib import admin
from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("original_name", "uploaded_at", "filesize_kb")
    readonly_fields = ("uploaded_at",)
    ordering = ("-uploaded_at",)
