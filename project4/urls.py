from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("features/", views.features, name="features"),

    # Study flow
    path("study/start/", views.start_study, name="start_study"),
    path("study/", views.study, name="study"),
    path("study/consent/", views.consent, name="consent"),
    path("study/background/", views.background, name="background"),
    path("study/instructions/", views.instructions, name="instructions"),
    path("study/practice/", views.practice, name="practice"),
    path("study/condition/", views.condition, name="condition"),
]
