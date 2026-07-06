from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("baseline/", views.baseline, name="baseline"),
    path("expert/", views.expert, name="expert"),
]
