import json

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import JsonResponse

from .forms import DatasetUploadForm, ExperimentForm, TrainedModelForm
from .models import Dataset, Experiment, TrainedModel
from .services.data import (
    read_csv_safely, numeric_column_names,
    build_chart_data, build_histogram_data, build_boxplot_data, build_heatmap_data,
)
from .services.preprocess import prepare_experiment
from .services.train import build_estimator, train_and_score
from .services.evaluate import build_evaluation

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

    experiments = list(dataset.experiments.all()) if dataset.is_parsed else []

    return render(request, "project1/dataset_detail.html", {
        "dataset": dataset,
        "page_obj": page_obj,
        "column_names": column_names,
        "sort_headers": sort_headers,
        "sort_col": sort_col,
        "sort_order": sort_order,
        "experiments": experiments,
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


def experiment_create(request, dataset_pk):
    dataset = get_object_or_404(Dataset, pk=dataset_pk)

    if not dataset.is_parsed:
        messages.error(request, "Dataset must be successfully parsed before creating an experiment.")
        return redirect("project1:dataset_detail", pk=dataset_pk)

    if not dataset.target_name:
        messages.error(request, "Dataset has no identified target column.")
        return redirect("project1:dataset_detail", pk=dataset_pk)

    if request.method == "POST":
        form = ExperimentForm(request.POST)
        if form.is_valid():
            experiment = form.save(commit=False)
            experiment.dataset = dataset
            if not experiment.name:
                count = dataset.experiments.count()
                experiment.name = f"Experiment {count + 1}"
            experiment.save()

            try:
                df = read_csv_safely(dataset.file)
                result = prepare_experiment(
                    df, dataset.target_name, dataset.problem_type, experiment.as_config()
                )
                experiment.n_train = result.n_train
                experiment.n_test = result.n_test
                experiment.n_features_before = result.n_features_before
                experiment.n_features_after = result.n_features_after
                experiment.feature_names = result.feature_names
                experiment.stratify_used = result.stratify_used
                experiment.prepare_error = None
            except Exception as e:
                experiment.prepare_error = str(e)

            experiment.save()
            messages.success(request, f"'{experiment.name}' prepared successfully.")
            return redirect("project1:experiment_detail", pk=experiment.pk)
    else:
        initial = {"stratify": dataset.problem_type == "classification"}
        form = ExperimentForm(initial=initial)

    return render(request, "project1/experiment_form.html", {
        "form": form,
        "dataset": dataset,
    })


def experiment_detail(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    return render(request, "project1/experiment_detail.html", {
        "experiment": experiment,
        "dataset": experiment.dataset,
        "models": list(experiment.models.all()),
    })


def experiment_delete(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    if request.method == "POST":
        name = experiment.name
        dataset_pk = experiment.dataset_id
        experiment.delete()
        messages.success(request, f"'{name}' deleted.")
        return redirect("project1:dataset_detail", pk=dataset_pk)
    return render(request, "project1/experiment_confirm_delete.html", {"experiment": experiment})


def model_create(request, experiment_pk):
    from django.core.files.base import ContentFile
    experiment = get_object_or_404(Experiment, pk=experiment_pk)

    if not experiment.is_prepared:
        messages.error(request, "Experiment must be prepared successfully before training.")
        return redirect("project1:experiment_detail", pk=experiment_pk)

    problem_type = experiment.dataset.problem_type
    if problem_type not in ("classification", "regression"):
        messages.error(request, "Cannot train: dataset's problem type is unknown.")
        return redirect("project1:experiment_detail", pk=experiment_pk)

    if request.method == "POST":
        form = TrainedModelForm(request.POST, problem_type=problem_type)
        if form.is_valid():
            model = form.save(commit=False)
            model.experiment = experiment
            if not model.name:
                count = experiment.models.count()
                model.name = f"{model.algorithm_display} #{count + 1}"
            model.save()

            estimator = None
            prepared = None
            try:
                df = read_csv_safely(experiment.dataset.file)
                prepared = prepare_experiment(
                    df,
                    experiment.dataset.target_name,
                    experiment.dataset.problem_type,
                    experiment.as_config(),
                )
                estimator = build_estimator(model.algorithm, experiment.random_seed)
                result = train_and_score(prepared, estimator, model.metric)
                model.model_file.save(
                    f"model_{model.pk}.joblib",
                    ContentFile(result.estimator_bytes),
                    save=False,
                )
                model.train_score = result.train_score
                model.test_score = result.test_score
                model.train_duration_ms = result.train_duration_ms
                model.train_error = None
            except Exception as e:
                model.train_error = str(e)

            # Evaluation runs only if training succeeded
            if model.train_error is None and estimator is not None and prepared is not None:
                try:
                    model.evaluation = build_evaluation(
                        estimator,
                        prepared,
                        experiment.dataset.problem_type,
                        prepared.label_map,
                        prepared.feature_names,
                    )
                    model.eval_error = None
                except Exception as e:
                    model.eval_error = str(e)

            model.save()
            messages.success(request, f"'{model.name}' trained.")
            return redirect("project1:model_detail", pk=model.pk)
    else:
        if problem_type == "classification":
            initial = {"algorithm": "logreg", "metric": "accuracy"}
        else:
            initial = {"algorithm": "linreg", "metric": "r2"}
        form = TrainedModelForm(initial=initial, problem_type=problem_type)

    return render(request, "project1/model_form.html", {
        "form": form,
        "experiment": experiment,
        "dataset": experiment.dataset,
    })


def model_detail(request, pk):
    model = get_object_or_404(TrainedModel, pk=pk)

    metric_rows = []
    if model.evaluation:
        ev = model.evaluation
        if ev.get("problem_type") == "classification":
            metric_specs = [
                ("accuracy",  "Accuracy"),
                ("f1",        "F1 (weighted)"),
                ("precision", "Precision (weighted)"),
                ("recall",    "Recall (weighted)"),
            ]
        else:
            metric_specs = [
                ("r2",   "R²"),
                ("rmse", "RMSE"),
                ("mae",  "MAE"),
            ]
        for key, display in metric_specs:
            metric_rows.append({
                "key": key,
                "display": display,
                "train": ev.get("train", {}).get(key),
                "test": ev.get("test", {}).get(key),
                "is_trained_metric": (key == model.metric),
            })

    return render(request, "project1/model_detail.html", {
        "model": model,
        "experiment": model.experiment,
        "dataset": model.experiment.dataset,
        "metric_rows": metric_rows,
    })


def model_delete(request, pk):
    model = get_object_or_404(TrainedModel, pk=pk)
    if request.method == "POST":
        name = model.name
        experiment_pk = model.experiment_id
        model.delete()
        messages.success(request, f"'{name}' deleted.")
        return redirect("project1:experiment_detail", pk=experiment_pk)
    return render(request, "project1/model_confirm_delete.html", {
        "model": model,
        "experiment": model.experiment,
    })


def _best_index(values, direction):
    """Return index of best value (higher or lower), or None if all None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if direction == "higher":
        return max(valid, key=lambda iv: iv[1])[0]
    return min(valid, key=lambda iv: iv[1])[0]


def experiment_compare(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    models = list(experiment.models.exclude(evaluation__isnull=True).order_by("created_at"))

    problem_type = experiment.dataset.problem_type
    if problem_type == "classification":
        metric_specs = [
            ("accuracy",  "Accuracy",            "higher"),
            ("f1",        "F1 (weighted)",       "higher"),
            ("precision", "Precision (weighted)", "higher"),
            ("recall",    "Recall (weighted)",   "higher"),
        ]
    else:
        metric_specs = [
            ("r2",   "R²",   "higher"),
            ("rmse", "RMSE", "lower"),
            ("mae",  "MAE",  "lower"),
        ]

    metric_rows = []
    chart_data_metrics = {}

    if models:
        # Context rows (algorithm + training metric)
        metric_rows.append({
            "display": "Algorithm",
            "cells": [{"display_value": m.algorithm_display, "is_best": False} for m in models],
        })
        metric_rows.append({
            "display": "Trained with",
            "cells": [{"display_value": m.metric_display, "is_best": False} for m in models],
        })

        for key, display, direction in metric_specs:
            train_values = [m.evaluation.get("train", {}).get(key) for m in models]
            test_values  = [m.evaluation.get("test",  {}).get(key) for m in models]
            best_train = _best_index(train_values, direction)
            best_test  = _best_index(test_values,  direction)

            metric_rows.append({
                "display": f"Train {display}",
                "cells": [
                    {
                        "display_value": "—" if v is None else f"{v:.4f}",
                        "is_best": (i == best_train),
                    }
                    for i, v in enumerate(train_values)
                ],
            })
            metric_rows.append({
                "display": f"Test {display}",
                "cells": [
                    {
                        "display_value": "—" if v is None else f"{v:.4f}",
                        "is_best": (i == best_test),
                    }
                    for i, v in enumerate(test_values)
                ],
            })

            chart_data_metrics[key] = {
                "display": display,
                "train": train_values,
                "test": test_values,
            }

        # Training time (lower is better)
        times = [m.train_duration_ms for m in models]
        best_time = _best_index(times, "lower")
        metric_rows.append({
            "display": "Training time (ms)",
            "cells": [
                {
                    "display_value": "—" if v is None else f"{v}",
                    "is_best": (i == best_time),
                }
                for i, v in enumerate(times)
            ],
        })

    # Selected metric for the bar chart (URL-persisted)
    valid_keys = [k for k, _, _ in metric_specs]
    selected_metric_key = request.GET.get("metric")
    if selected_metric_key not in valid_keys:
        selected_metric_key = valid_keys[0]

    chart_payload = {
        "labels": [m.name for m in models],
        "metrics": chart_data_metrics,
        "selected": selected_metric_key,
    }

    return render(request, "project1/experiment_compare.html", {
        "experiment": experiment,
        "dataset": experiment.dataset,
        "models": models,
        "metric_rows": metric_rows,
        "metric_specs": metric_specs,
        "selected_metric_key": selected_metric_key,
        "chart_payload": chart_payload,
        "problem_type": problem_type,
    })
