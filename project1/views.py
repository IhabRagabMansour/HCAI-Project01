from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages

from .forms import DatasetUploadForm
from .models import Dataset


def index(request):
    recent_datasets = Dataset.objects.all()[:5]
    return render(request, "project1/index.html", {
        "title": "Project 1 — Automated Machine Learning",
        "description": "Upload a CSV dataset, visualize it, and train supervised ML models.",
        "recent_datasets": recent_datasets,
    })


def dataset_upload(request):
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            messages.success(request, f"'{dataset.original_name}' uploaded successfully.")
            return redirect("project1:dataset_detail", pk=dataset.pk)
    else:
        form = DatasetUploadForm()

    return render(request, "project1/dataset_upload.html", {"form": form})


def dataset_list(request):
    datasets = Dataset.objects.all()
    return render(request, "project1/dataset_list.html", {"datasets": datasets})


def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    return render(request, "project1/dataset_detail.html", {"dataset": dataset})


def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        name = dataset.original_name
        dataset.delete()
        messages.success(request, f"'{name}' deleted successfully.")
        return redirect("project1:dataset_list")
    return render(request, "project1/dataset_confirm_delete.html", {"dataset": dataset})
