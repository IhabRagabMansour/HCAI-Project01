from django.contrib import admin

from .models import HumanLabel


@admin.register(HumanLabel)
class HumanLabelAdmin(admin.ModelAdmin):
    list_display = ("article_index", "true_label", "chosen_label", "is_correct", "created_at")
    list_filter = ("true_label", "chosen_label")
