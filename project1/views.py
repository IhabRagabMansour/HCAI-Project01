from django.shortcuts import render


def index(request):
    context = {
        "title": "Project 1 — Automated Machine Learning",
        "description": "Upload a CSV dataset, visualize it, and train supervised ML models.",
    }
    return render(request, "project1/index.html", context)
