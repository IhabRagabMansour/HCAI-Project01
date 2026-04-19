from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator

from .forms import DatasetUploadForm
from .models import Dataset
from .services.data import read_csv_safely

ROWS_PER_PAGE = 25


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

    page_obj = None
    column_names = [col["name"] for col in dataset.columns] if dataset.columns else []
    sort_col = None
    sort_order = "asc"

    if dataset.is_parsed:
        try:
            df = read_csv_safely(dataset.file)

            # Resolve sort params
            requested_sort = request.GET.get("sort", "")
            requested_order = request.GET.get("order", "asc")
            if requested_sort in df.columns:
                sort_col = requested_sort
                sort_order = "desc" if requested_order == "desc" else "asc"
                df = df.sort_values(by=sort_col, ascending=(sort_order == "asc"))

            rows = df.values.tolist()
            paginator = Paginator(rows, ROWS_PER_PAGE)
            page_number = request.GET.get("page", 1)
            page_obj = paginator.get_page(page_number)
        except Exception:
            pass

    # Build per-column sort URLs for the template
    sort_headers = []
    for name in column_names:
        if name == sort_col:
            # Already sorted by this column — toggle direction
            next_order = "asc" if sort_order == "desc" else "desc"
            indicator = "▲" if sort_order == "asc" else "▼"
        else:
            next_order = "asc"
            indicator = ""
        sort_headers.append({
            "name": name,
            "url": f"?sort={name}&order={next_order}&page=1#data-section",
            "indicator": indicator,
            "active": name == sort_col,
        })

    return render(request, "project1/dataset_detail.html", {
        "dataset": dataset,
        "page_obj": page_obj,
        "column_names": column_names,
        "sort_headers": sort_headers,
        "sort_col": sort_col,
        "sort_order": sort_order,
    })


def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        name = dataset.original_name
        dataset.delete()
        messages.success(request, f"'{name}' deleted successfully.")
        return redirect("project1:dataset_list")
    return render(request, "project1/dataset_confirm_delete.html", {"dataset": dataset})
