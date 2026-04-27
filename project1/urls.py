from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("datasets/", views.dataset_list, name="dataset_list"),
    path("datasets/upload/", views.dataset_upload, name="dataset_upload"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:pk>/delete/", views.dataset_delete, name="dataset_delete"),
    path("datasets/<int:pk>/chart/", views.dataset_chart_data, name="dataset_chart_data"),
    path("datasets/<int:pk>/set-type/", views.dataset_set_problem_type, name="dataset_set_problem_type"),
    path("datasets/<int:dataset_pk>/experiments/new/", views.experiment_create, name="experiment_create"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
    path("experiments/<int:pk>/delete/", views.experiment_delete, name="experiment_delete"),
    path("experiments/<int:experiment_pk>/models/new/", views.model_create, name="model_create"),
    path("models/<int:pk>/", views.model_detail, name="model_detail"),
]
