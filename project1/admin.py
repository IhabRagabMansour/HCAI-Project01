from django.contrib import admin
from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("original_name", "uploaded_at", "n_rows", "n_columns", "problem_type", "parse_error_short", "filesize_kb")
    list_filter = ("problem_type",)
    readonly_fields = ("uploaded_at",)
    ordering = ("-uploaded_at",)

    def parse_error_short(self, obj):
        if obj.parse_error and len(obj.parse_error) > 40:
            return obj.parse_error[:40] + "…"
        return obj.parse_error
    parse_error_short.short_description = "Parse error"
