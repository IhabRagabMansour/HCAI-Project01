import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render

from .services.data import (
    get_penguin_data, SPECIES_ORDER, INPUT_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
)
from .services.feature_effects import compute_pdp, compute_ale
from .services.grids import unconstrained_tree_entry, get_grid
from .services.treeviz import tree_to_png_base64, tree_to_text
from .services.coefs import coefficient_table
from .services.selection import (
    get_selected_model, selection_score, clamp_lambda, normalize_model_class,
    LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP, LAMBDA_DEFAULT, MODEL_CLASSES,
)
from .services.counterfactuals import (
    generate_counterfactuals, CounterfactualConfig,
)

DEFAULT_SEED = 42
CF_N_CANDIDATES = 600
CF_K_DEFAULT = 5


def _fmt_value(feat, val):
    """Human-readable display string for a feature value."""
    if feat in CATEGORICAL_FEATURES:
        return str(val)
    if feat == "year":
        return str(int(val))
    return f"{float(val):.1f}"


def report(request):
    """Written explanation of design choices"""
    from .services.grids import TREE_MAX_LEAF_NODES_GRID, LOGREG_C_GRID
    context = {
        "title": "Project 2 — Design Report",
        "tree_grid": ", ".join("None" if v is None else str(v) for v in TREE_MAX_LEAF_NODES_GRID),
        "logreg_grid": ", ".join(str(v) for v in LOGREG_C_GRID),
        "lambda_min": LAMBDA_MIN,
        "lambda_max": LAMBDA_MAX,
        "lambda_step": LAMBDA_STEP,
        "biometric_features": BIOMETRIC_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    return render(request, "project2/report.html", context)


def index(request):
    data = get_penguin_data(DEFAULT_SEED)
    context = {
        "title": "Project 2 — Explainability Dashboard",
        "description": (
            "An interactive explainability dashboard for models trained on the "
            "Palmer Penguins dataset. Compare model accuracy against complexity, "
            "generate counterfactual explanations, and inspect global feature "
            "effects with PDP and ALE plots."
        ),
        "n_train": len(data.X_train),
        "n_test": len(data.X_test),
        "n_total": len(data.X_all),
        "species": SPECIES_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "biometric_features": BIOMETRIC_FEATURES,
        "input_features": INPUT_FEATURES,
    }
    return render(request, "project2/index.html", context)


def decision_tree(request):
    """Task 1: fit a decision tree, show the tree, its test accuracy, and the
    number of leaves (= the tree's complexity Omega)."""
    seed = DEFAULT_SEED
    entry = unconstrained_tree_entry(seed)

    context = {
        "title": "Task 1 — Decision Tree",
        "entry": entry,
        "test_accuracy": entry.test_accuracy,
        "n_leaves": entry.complexity,
        "tree_png": tree_to_png_base64(entry.pipeline),
        "tree_text": tree_to_text(entry.pipeline),
        "species": SPECIES_ORDER,
    }
    return render(request, "project2/decision_tree.html", context)


def dashboard(request):
    """The connected interactive dashboard. Model class + lambda drive the
    selected model, the grid table, and the accuracy/complexity frontier.
    (Counterfactuals and PDP/ALE regions are added in later stages.)"""
    model_class = normalize_model_class(request.GET.get("model"))
    lam = clamp_lambda(request.GET.get("lambda", LAMBDA_DEFAULT))
    seed = DEFAULT_SEED

    selected = get_selected_model(model_class, lam, seed)
    grid = get_grid(model_class, seed)

    # Task 2/3: score every model at the current lambda; mark the minimizer.
    grid_rows = []
    frontier_points = []
    for entry in grid:
        score = selection_score(entry.test_accuracy, entry.complexity, lam)
        is_selected = entry is selected.entry
        grid_rows.append({
            "param_display": entry.param_display,
            "test_accuracy": entry.test_accuracy,
            "complexity": entry.complexity,
            "score": score,
            "is_selected": is_selected,
        })
        frontier_points.append({
            "x": entry.complexity,
            "y": entry.test_accuracy,
            "label": entry.param_display,
            "selected": is_selected,
        })

    param_label = "max_leaf_nodes" if model_class == "tree" else "C"
    complexity_label = "Number of leaves" if model_class == "tree" else "Nonzero coefficients"

    # ── Region B: counterfactual explanations (uses the SAME selected model) ──
    data = get_penguin_data(seed)
    n_rows = len(data.X_all)

    cf_row = _parse_int(request.GET.get("cf_row"), default=0, lo=0, hi=n_rows - 1)
    cf_target = request.GET.get("cf_target", "")
    cf_k = _parse_int(request.GET.get("cf_k"), default=CF_K_DEFAULT, lo=1, hi=10)

    x = data.X_all.iloc[cf_row].to_dict()
    x_true_class = str(data.y_all.iloc[cf_row])
    x_pred_class = str(selected.pipeline.predict(pd.DataFrame([x], columns=INPUT_FEATURES))[0])
    cf_target_choices = [sp for sp in SPECIES_ORDER if sp != x_true_class]
    if cf_target not in cf_target_choices:
        cf_target = ""

    original_cells = [
        {"feature": f, "display": _fmt_value(f, x[f])} for f in INPUT_FEATURES
    ]

    cf_table = []
    if cf_target:
        cfs = generate_counterfactuals(
            selected.pipeline, x, cf_target, data,
            CounterfactualConfig(n_candidates=CF_N_CANDIDATES, k=cf_k, seed=0),
        )
        for cf in cfs:
            cells = [
                {
                    "feature": f,
                    "display": _fmt_value(f, cf.row[f]),
                    "changed": f in cf.changed_features,
                }
                for f in INPUT_FEATURES
            ]
            cf_table.append({
                "cells": cells,
                "predicted_class": cf.predicted_class,
                "target_proba": cf.target_proba,
                "distance": cf.distance,
                "n_changed": cf.n_changed,
            })

    context = {
        "title": "Project 2 — Dashboard",
        "model_class": model_class,
        "model_classes": MODEL_CLASSES,
        "lam": lam,
        "lambda_min": LAMBDA_MIN,
        "lambda_max": LAMBDA_MAX,
        "lambda_step": LAMBDA_STEP,
        "selected": selected,
        "param_label": param_label,
        "complexity_label": complexity_label,
        "grid_rows": grid_rows,
        "frontier_payload": {
            "points": frontier_points,
            "model_class": model_class,
            "complexity_label": complexity_label,
        },
        "species": SPECIES_ORDER,
        "cf_target_choices": cf_target_choices,
        # Region A rendering: tree plot for trees; coefficient table for logreg
        "tree_png": tree_to_png_base64(selected.pipeline) if model_class == "tree" else None,
        "coef_table": coefficient_table(selected.pipeline) if model_class == "logreg" else None,
        # Region B: counterfactuals
        "input_features": INPUT_FEATURES,
        "n_rows": n_rows,
        "cf_row": cf_row,
        "cf_target": cf_target,
        "cf_k": cf_k,
        "x_true_class": x_true_class,
        "x_pred_class": x_pred_class,
        "original_cells": original_cells,
        "cf_table": cf_table,
        # Region C: PDP / ALE
        "biometric_features": BIOMETRIC_FEATURES,
        "fe_feature": _fe_feature(request.GET.get("fe_feature")),
        "fe_n_grid": _parse_int(request.GET.get("fe_n_grid"), default=25, lo=5, hi=50),
        "fe_n_bins": _parse_int(request.GET.get("fe_n_bins"), default=20, lo=4, hi=40),
    }
    return render(request, "project2/dashboard.html", context)


def _fe_feature(value):
    """Validate the PDP/ALE feature, defaulting to the first biometric one."""
    return value if value in BIOMETRIC_FEATURES else BIOMETRIC_FEATURES[0]


def _parse_int(value, default, lo, hi):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _selected_from_request(request):
    """Rebuild the active model from the request's query params — the shared
    consistency entry point used by the PDP/ALE endpoints."""
    model_class = normalize_model_class(request.GET.get("model"))
    lam = clamp_lambda(request.GET.get("lambda", LAMBDA_DEFAULT))
    return get_selected_model(model_class, lam, DEFAULT_SEED), model_class, lam


def pdp_data(request):
    """JSON endpoint: PDP curves for the selected model + feature."""
    feature = request.GET.get("feature", "")
    if feature not in BIOMETRIC_FEATURES:
        return JsonResponse({"error": "Invalid feature."}, status=400)
    n_grid = _parse_int(request.GET.get("n_grid"), default=25, lo=5, hi=50)

    selected, _model_class, _lam = _selected_from_request(request)
    data = get_penguin_data(DEFAULT_SEED)
    return JsonResponse(compute_pdp(selected.pipeline, data.X_all, feature, n_grid))


def ale_data(request):
    """JSON endpoint: ALE curves for the selected model + feature."""
    feature = request.GET.get("feature", "")
    if feature not in BIOMETRIC_FEATURES:
        return JsonResponse({"error": "Invalid feature."}, status=400)
    n_bins = _parse_int(request.GET.get("n_bins"), default=20, lo=4, hi=40)

    selected, _model_class, _lam = _selected_from_request(request)
    data = get_penguin_data(DEFAULT_SEED)
    return JsonResponse(compute_ale(selected.pipeline, data.X_all, feature, n_bins))
