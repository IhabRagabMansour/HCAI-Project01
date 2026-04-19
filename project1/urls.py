from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("datasets/", views.dataset_list, name="dataset_list"),
    path("datasets/upload/", views.dataset_upload, name="dataset_upload"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:pk>/delete/", views.dataset_delete, name="dataset_delete"),
]
