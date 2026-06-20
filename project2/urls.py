from django.urls import path
from . import views

app_name = "project2"

urlpatterns = [
    path("", views.index, name="index"),
    path("decision-tree/", views.decision_tree, name="decision_tree"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("pdp/", views.pdp_data, name="pdp_data"),
    path("ale/", views.ale_data, name="ale_data"),
    path("report/", views.report, name="report"),
]
