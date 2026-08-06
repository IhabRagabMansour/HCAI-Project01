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
from .services.train import train_and_score, train_with_random_search, HYPERPARAM_SPECS
from .services.evaluate import build_evaluation, cross_validate_pipeline, compute_learning_curve
from .services.predict import build_input_form_spec, pick_random_row_values, predict_single

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
    trained_models_count = (
        TrainedModel.objects
        .filter(experiment__dataset=dataset)
        .exclude(evaluation__isnull=True)
        .count()
        if dataset.is_parsed else 0
    )

    return render(request, "project1/dataset_detail.html", {
        "dataset": dataset,
        "page_obj": page_obj,
        "column_names": column_names,
        "sort_headers": sort_headers,
        "sort_col": sort_col,
        "sort_order": sort_order,
        "experiments": experiments,
        "trained_models_count": trained_models_count,
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

            # Capture excluded-column checkboxes (target is never excludable)
            valid_cols = {c["name"] for c in (dataset.columns or [])}
            valid_cols.discard(dataset.target_name)
            excluded = [c for c in request.POST.getlist("excluded_columns") if c in valid_cols]
            experiment.excluded_columns = excluded

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
                experiment.n_outliers_removed = result.n_outliers_removed
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


def _parse_hyperparameters(post, algorithm: str):
    """Extract HP values for the given algorithm from POST, validate against
    the spec, and return (hp_dict, error_list). Only values that differ from
    the spec's default are kept in hp_dict."""
    specs = HYPERPARAM_SPECS.get(algorithm, [])

    # If no hp fields for this algorithm are in POST at all, the form section
    # was never rendered/submitted — fall back to defaults entirely rather
    # than misinterpreting absent checkboxes as False.
    prefix = f"hp__{algorithm}__"
    if not any(k.startswith(prefix) for k in post.keys()):
        return {}, []

    hp: dict = {}
    errors: list = []

    for spec in specs:
        field_name = f"hp__{algorithm}__{spec['key']}"

        if spec["type"] == "bool":
            # Checkboxes: present in POST → True; absent → False
            value = field_name in post
            if value != spec["default"]:
                hp[spec["key"]] = value
            continue

        raw = (post.get(field_name) or "").strip()
        if not raw:
            continue  # leave at sklearn default

        if spec["type"] == "int":
            try:
                v = int(raw)
            except ValueError:
                errors.append(f"{spec['label']}: must be an integer.")
                continue
            if v < spec["min"] or v > spec["max"]:
                errors.append(
                    f"{spec['label']}: must be between {spec['min']} and {spec['max']}."
                )
                continue
            if v != spec["default"]:
                hp[spec["key"]] = v

        elif spec["type"] == "float":
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"{spec['label']}: must be a number.")
                continue
            if v < spec["min"] or v > spec["max"]:
                errors.append(
                    f"{spec['label']}: must be between {spec['min']} and {spec['max']}."
                )
                continue
            if v != spec["default"]:
                hp[spec["key"]] = v

        elif spec["type"] == "choice":
            valid = {c[0] for c in spec["choices"]}
            if raw not in valid:
                errors.append(f"{spec['label']}: invalid choice.")
                continue
            if raw != spec["default"]:
                hp[spec["key"]] = raw

    return hp, errors


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

    hp_errors: list = []
    if request.method == "POST":
        form = TrainedModelForm(request.POST, problem_type=problem_type)
        training_mode = request.POST.get("training_mode", "manual")
        # Parse + validate hyperparameters only for the manual path.
        if training_mode == "random_search":
            hp_dict, hp_errors = {}, []
        else:
            algo_for_hp = request.POST.get("algorithm", "")
            hp_dict, hp_errors = _parse_hyperparameters(request.POST, algo_for_hp)

        if form.is_valid() and not hp_errors:
            training_mode = form.cleaned_data.get("training_mode", "manual")
            model = form.save(commit=False)
            model.experiment = experiment
            if not model.name:
                count = experiment.models.count()
                model.name = f"{model.algorithm_display} #{count + 1}"
            search_info = None
            model.save()

            pipeline = None
            prepared = None
            try:
                df = read_csv_safely(experiment.dataset.file)
                prepared = prepare_experiment(
                    df,
                    experiment.dataset.target_name,
                    experiment.dataset.problem_type,
                    experiment.as_config(),
                )
                if training_mode == "random_search":
                    result, hp_dict, search_info = train_with_random_search(
                        prepared,
                        model.algorithm,
                        model.metric,
                        problem_type,
                        random_seed=experiment.random_seed,
                        cv_folds=model.cv_folds,
                        n_iter=form.cleaned_data.get("random_search_iterations", 20),
                    )
                else:
                    result = train_and_score(
                        prepared, model.algorithm, model.metric,
                        random_seed=experiment.random_seed,
                        hyperparameters=hp_dict,
                    )
                pipeline = result.pipeline
                model.model_file.save(
                    f"model_{model.pk}.joblib",
                    ContentFile(result.pipeline_bytes),
                    save=False,
                )
                model.hyperparameters = hp_dict
                model.train_score = result.train_score
                model.test_score = result.test_score
                model.train_duration_ms = result.train_duration_ms
                model.train_error = None
            except Exception as e:
                model.train_error = str(e)

            # Evaluation runs only if training succeeded
            if model.train_error is None and pipeline is not None and prepared is not None:
                try:
                    model.evaluation = build_evaluation(
                        pipeline,
                        prepared,
                        experiment.dataset.problem_type,
                        prepared.label_map,
                        prepared.feature_names,
                    )
                    if search_info is not None:
                        model.evaluation["search"] = search_info
                        model.cv_scores = {
                            model.metric: {
                                "mean": search_info.get("best_score"),
                                "std": search_info.get("best_std", 0.0),
                                "scores": [],
                            }
                        }
                    elif model.cv_folds >= 2:
                        # Cross-validation (optional; cv_folds=0 skips)
                        try:
                            model.cv_scores = cross_validate_pipeline(
                                prepared,
                                model.algorithm,
                                experiment.dataset.problem_type,
                                cv_folds=model.cv_folds,
                                random_seed=experiment.random_seed,
                                hyperparameters=hp_dict,
                            )
                        except Exception as e:
                            model.cv_scores = {"_error": str(e)}

                        # Learning curve (also tied to cv_folds; same fold count)
                        try:
                            lc = compute_learning_curve(
                                prepared,
                                model.algorithm,
                                experiment.dataset.problem_type,
                                cv_folds=model.cv_folds,
                                metric=model.metric,
                                random_seed=experiment.random_seed,
                                hyperparameters=hp_dict,
                            )
                            if lc and model.evaluation is not None:
                                model.evaluation["learning_curve"] = lc
                        except Exception:
                            pass  # Learning curve is optional, don't block training
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

    # Filter HYPERPARAM_SPECS to only the algorithms valid for this problem type
    if problem_type == "classification":
        valid_algos = ("logreg", "rf_clf", "svm", "knn_clf", "dt_clf")
    else:
        valid_algos = ("linreg", "rf_reg", "svr", "knn_reg", "dt_reg")
    hp_sections = [
        {"algorithm": a, "specs": HYPERPARAM_SPECS.get(a, [])}
        for a in valid_algos
    ]

    return render(request, "project1/model_form.html", {
        "form": form,
        "experiment": experiment,
        "dataset": experiment.dataset,
        "hp_sections": hp_sections,
        "hp_errors": hp_errors,
        "posted": request.POST if request.method == "POST" else None,
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
        cv = model.cv_scores if isinstance(model.cv_scores, dict) else {}
        for key, display in metric_specs:
            cv_entry = cv.get(key)
            cv_display = None
            if isinstance(cv_entry, dict) and "mean" in cv_entry:
                cv_display = f"{cv_entry['mean']:.4f} ± {cv_entry['std']:.4f}"
            metric_rows.append({
                "key": key,
                "display": display,
                "train": ev.get("train", {}).get(key),
                "test": ev.get("test", {}).get(key),
                "cv": cv_display,
                "is_trained_metric": (key == model.metric),
            })

    cv_error_msg = None
    if isinstance(model.cv_scores, dict):
        cv_error_msg = model.cv_scores.get("_error")

    return render(request, "project1/model_detail.html", {
        "model": model,
        "experiment": model.experiment,
        "dataset": model.experiment.dataset,
        "metric_rows": metric_rows,
        "cv_error_msg": cv_error_msg,
    })


def model_predict(request, pk):
    import joblib
    model = get_object_or_404(TrainedModel, pk=pk)
    experiment = model.experiment
    dataset = experiment.dataset

    if not model.is_trained:
        messages.error(request, "Model must be trained successfully before predicting.")
        return redirect("project1:model_detail", pk=pk)

    try:
        df = read_csv_safely(dataset.file)
    except Exception as e:
        messages.error(request, f"Could not read dataset: {e}")
        return redirect("project1:model_detail", pk=pk)

    excluded = set(experiment.excluded_columns or [])
    if dataset.target_name:
        excluded.add(dataset.target_name)
    specs = build_input_form_spec(df, dataset.columns or [], excluded)

    current_values: dict = {s["name"]: s["default"] for s in specs}
    pred_errors: list = []
    pred_label = None
    pred_probs_list = None

    if request.method == "POST":
        action = request.POST.get("action", "predict")

        if action == "sample":
            current_values = pick_random_row_values(df, specs)
        else:
            # Parse + validate user input
            row: dict = {}
            for spec in specs:
                raw = (request.POST.get(f"f__{spec['name']}") or "").strip()
                if raw == "":
                    pred_errors.append(f"{spec['name']}: value required")
                    continue
                if spec["input_type"] == "number":
                    try:
                        row[spec["name"]] = float(raw)
                    except ValueError:
                        pred_errors.append(f"{spec['name']}: must be a number")
                        continue
                else:
                    row[spec["name"]] = raw
                current_values[spec["name"]] = raw

            if not pred_errors:
                try:
                    with model.model_file.open("rb") as f:
                        pipeline = joblib.load(f)
                    feature_order = [s["name"] for s in specs]
                    result = predict_single(pipeline, row, feature_order)

                    if dataset.problem_type == "classification":
                        labels = (model.evaluation or {}).get("labels", [])
                        try:
                            pred_int = int(result["prediction"])
                            pred_label = labels[pred_int] if 0 <= pred_int < len(labels) else str(result["prediction"])
                        except (TypeError, ValueError):
                            pred_label = str(result["prediction"])

                        if result["probabilities"] is not None:
                            pred_probs_list = [
                                {"label": labels[i] if i < len(labels) else str(i),
                                 "prob": float(p)}
                                for i, p in enumerate(result["probabilities"])
                            ]
                    else:
                        pred_label = f"{float(result['prediction']):.4f}"
                except Exception as e:
                    pred_errors.append(f"Prediction failed: {e}")

    # Attach the current value to each spec for easy template rendering
    for spec in specs:
        spec["current"] = current_values.get(spec["name"], spec["default"])

    test_rmse = None
    if dataset.problem_type == "regression":
        test_rmse = (model.evaluation or {}).get("test", {}).get("rmse")

    return render(request, "project1/model_predict.html", {
        "model": model,
        "experiment": experiment,
        "dataset": dataset,
        "specs": specs,
        "pred_label": pred_label,
        "pred_probs_list": pred_probs_list,
        "pred_errors": pred_errors,
        "is_classification": dataset.problem_type == "classification",
        "test_rmse": test_rmse,
    })


def model_reevaluate(request, pk):
    """Recompute evaluation from the saved pipeline without retraining."""
    import joblib
    model = get_object_or_404(TrainedModel, pk=pk)
    if request.method != "POST":
        return redirect("project1:model_detail", pk=pk)

    if not model.model_file:
        messages.error(request, "Model has no saved file to re-evaluate.")
        return redirect("project1:model_detail", pk=pk)

    try:
        with model.model_file.open("rb") as f:
            pipeline = joblib.load(f)
        df = read_csv_safely(model.experiment.dataset.file)
        prepared = prepare_experiment(
            df,
            model.experiment.dataset.target_name,
            model.experiment.dataset.problem_type,
            model.experiment.as_config(),
        )
        model.evaluation = build_evaluation(
            pipeline,
            prepared,
            model.experiment.dataset.problem_type,
            prepared.label_map,
            prepared.feature_names,
        )
        model.eval_error = None
        model.save()
        messages.success(request, f"Evaluation recomputed for '{model.name}'.")
    except Exception as e:
        model.eval_error = str(e)
        model.save()
        messages.error(request, f"Re-evaluation failed: {e}")

    return redirect("project1:model_detail", pk=pk)


def model_download(request, pk):
    """Serve the saved joblib pipeline as a downloadable file."""
    from django.http import FileResponse
    model = get_object_or_404(TrainedModel, pk=pk)
    if not model.model_file:
        messages.error(request, "No model file to download.")
        return redirect("project1:model_detail", pk=pk)

    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model.name)
    filename = f"{safe_name or 'model'}.joblib"
    return FileResponse(
        model.model_file.open("rb"),
        as_attachment=True,
        filename=filename,
    )


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


def dataset_compare(request, pk):
    """Compare trained models across ALL experiments belonging to a dataset.
    Selection state lives in the URL via ?ids=1&ids=3 (Django getlist)."""
    dataset = get_object_or_404(Dataset, pk=pk)

    all_models = list(
        TrainedModel.objects
        .filter(experiment__dataset=dataset)
        .exclude(evaluation__isnull=True)
        .select_related("experiment")
        .order_by("experiment_id", "created_at")
    )

    # Parse selection from query string
    raw_ids = request.GET.getlist("ids")
    selected_ids: set = set()
    for s in raw_ids:
        if s.isdigit():
            selected_ids.add(int(s))
    selected_models = [m for m in all_models if m.pk in selected_ids]

    # Group models by experiment for the selection table
    grouped: dict = {}
    for m in all_models:
        grouped.setdefault(m.experiment_id, {"experiment": m.experiment, "models": []})
        grouped[m.experiment_id]["models"].append(m)
    grouped_list = list(grouped.values())

    problem_type = dataset.problem_type
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
    chart_data_metrics: dict = {}

    if len(selected_models) >= 2:
        # Recipe rows (one per dimension that may differ between experiments)
        recipe_specs = [
            ("Experiment", lambda m: m.experiment.name),
            ("Algorithm",  lambda m: m.algorithm_display),
            ("Trained with", lambda m: m.metric_display),
            ("Encoding",   lambda m: m.experiment.get_categorical_encoding_display()),
            ("Scaling",    lambda m: m.experiment.get_scaling_display()),
            ("Missing",    lambda m: m.experiment.get_missing_strategy_display()),
            ("Oversampling", lambda m: m.experiment.get_oversampling_display()),
            ("Test size",  lambda m: f"{m.experiment.test_size_pct}%"),
        ]
        for label, getter in recipe_specs:
            metric_rows.append({
                "display": label,
                "cells": [{"display_value": getter(m), "is_best": False} for m in selected_models],
            })

        # Metric rows (train + test)
        for key, display, direction in metric_specs:
            train_values = [m.evaluation.get("train", {}).get(key) for m in selected_models]
            test_values  = [m.evaluation.get("test",  {}).get(key) for m in selected_models]
            best_train = _best_index(train_values, direction)
            best_test  = _best_index(test_values, direction)

            metric_rows.append({
                "display": f"Train {display}",
                "cells": [
                    {"display_value": "—" if v is None else f"{v:.4f}",
                     "is_best": (i == best_train)}
                    for i, v in enumerate(train_values)
                ],
            })
            metric_rows.append({
                "display": f"Test {display}",
                "cells": [
                    {"display_value": "—" if v is None else f"{v:.4f}",
                     "is_best": (i == best_test)}
                    for i, v in enumerate(test_values)
                ],
            })

            chart_data_metrics[key] = {
                "display": display,
                "train": train_values,
                "test": test_values,
            }

        # Training time row
        times = [m.train_duration_ms for m in selected_models]
        best_time = _best_index(times, "lower")
        metric_rows.append({
            "display": "Training time (ms)",
            "cells": [
                {"display_value": "—" if v is None else f"{v}",
                 "is_best": (i == best_time)}
                for i, v in enumerate(times)
            ],
        })

    valid_keys = [k for k, _, _ in metric_specs]
    selected_metric_key = request.GET.get("metric")
    if selected_metric_key not in valid_keys:
        selected_metric_key = valid_keys[0] if valid_keys else ""

    chart_payload = {
        "labels": [m.name for m in selected_models],
        "metrics": chart_data_metrics,
        "selected": selected_metric_key,
    }

    return render(request, "project1/dataset_compare.html", {
        "dataset": dataset,
        "grouped_list": grouped_list,
        "all_models": all_models,
        "selected_models": selected_models,
        "selected_ids": selected_ids,
        "metric_rows": metric_rows,
        "metric_specs": metric_specs,
        "selected_metric_key": selected_metric_key,
        "chart_payload": chart_payload,
        "problem_type": problem_type,
    })
