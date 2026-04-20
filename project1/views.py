import json

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse

from .forms import DatasetUploadForm
from .models import Dataset
from .services.data import (
    read_csv_safely, numeric_column_names,
    build_chart_data, build_histogram_data, build_boxplot_data, build_heatmap_data,
)

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
    numeric_cols = []
    histogram_cols = []
    class_names = []
    chart_type = "scatter"
    chart_x = ""
    chart_y = ""
    chart_mode = ""

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

            numeric_cols = numeric_column_names(df)

            # Histogram also allows the target column even if non-numeric
            histogram_cols = list(numeric_cols)
            if dataset.target_name and dataset.target_name not in histogram_cols and dataset.target_name in df.columns:
                histogram_cols.append(dataset.target_name)

            if dataset.problem_type == "classification" and dataset.target_name in df.columns:
                class_names = sorted(df[dataset.target_name].astype(str).unique().tolist())

            # Chart state from URL
            default_mode = dataset.problem_type if dataset.problem_type in ("classification", "regression") else "regression"
            chart_type = request.GET.get("type", "scatter")
            if chart_type not in ("scatter", "histogram", "boxplot", "heatmap"):
                chart_type = "scatter"

            valid_x_cols = histogram_cols if chart_type == "histogram" else numeric_cols
            chart_x = request.GET.get("x", valid_x_cols[0] if valid_x_cols else "")
            chart_y = request.GET.get("y", numeric_cols[1] if len(numeric_cols) >= 2 else "")
            chart_mode = request.GET.get("mode", default_mode)
            if chart_x not in valid_x_cols:
                chart_x = valid_x_cols[0] if valid_x_cols else ""
            if chart_y not in numeric_cols:
                chart_y = numeric_cols[1] if len(numeric_cols) >= 2 else (numeric_cols[0] if numeric_cols else "")
            if chart_mode not in ("classification", "regression"):
                chart_mode = default_mode
        except Exception:
            pass

    # Build per-column sort URLs for the template
    sort_headers = []
    for name in column_names:
        if name == sort_col:
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
        "numeric_cols": numeric_cols,
        "histogram_cols": histogram_cols,
        "class_names": class_names,
        "chart_type": chart_type,
        "chart_x": chart_x,
        "chart_y": chart_y,
        "chart_mode": chart_mode,
    })


def dataset_chart_data(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if not dataset.is_parsed:
        return JsonResponse({"error": "Dataset not parsed."}, status=400)

    chart_type = request.GET.get("type", "scatter")
    x_col = request.GET.get("x", "")
    y_col = request.GET.get("y", "")
    mode = request.GET.get("mode", "regression")

    try:
        df = read_csv_safely(dataset.file)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    numeric_cols = numeric_column_names(df)

    if chart_type == "histogram":
        if x_col not in df.columns:
            return JsonResponse({"error": "Invalid column."}, status=400)
        return JsonResponse(build_histogram_data(df, x_col))

    if chart_type == "boxplot":
        if x_col not in numeric_cols:
            return JsonResponse({"error": "Invalid column."}, status=400)
        return JsonResponse(build_boxplot_data(df, x_col, dataset.target_name, mode))

    if chart_type == "heatmap":
        if len(numeric_cols) < 2:
            return JsonResponse({"error": "Need at least 2 numeric columns."}, status=400)
        return JsonResponse(build_heatmap_data(df, numeric_cols))

    # scatter (default)
    if x_col not in numeric_cols or y_col not in numeric_cols:
        return JsonResponse({"error": "Invalid column selection."}, status=400)
    return JsonResponse(build_chart_data(df, x_col, y_col, mode, dataset.target_name))


def dataset_set_problem_type(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        new_type = request.POST.get("problem_type")
        if new_type in ("classification", "regression"):
            dataset.problem_type = new_type
            dataset.save(update_fields=["problem_type"])
            messages.success(request, f"Problem type changed to {new_type}.")
    return redirect("project1:dataset_detail", pk=pk)


def dataset_delete(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        name = dataset.original_name
        dataset.delete()
        messages.success(request, f"'{name}' deleted successfully.")
        return redirect("project1:dataset_list")
    return render(request, "project1/dataset_confirm_delete.html", {"dataset": dataset})
