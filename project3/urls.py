from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("baseline/", views.baseline, name="baseline"),
    path("expert/", views.expert, name="expert"),
    path("defer/", views.defer, name="defer"),
    path("active/", views.active, name="active"),
    path("human/", views.human, name="human"),
    path("report/download/", views.report_download, name="report_download"),
]
