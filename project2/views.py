from django.shortcuts import render

from .services.data import (
    get_penguin_data, SPECIES_ORDER, INPUT_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
)
from .services.grids import unconstrained_tree_entry, get_grid
from .services.treeviz import tree_to_png_base64, tree_to_text
from .services.selection import (
    get_selected_model, selection_score, clamp_lambda, normalize_model_class,
    LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP, LAMBDA_DEFAULT, MODEL_CLASSES,
)

DEFAULT_SEED = 42


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
        # Region A rendering: tree plot for trees; logreg coef table comes in P2-4
        "tree_png": tree_to_png_base64(selected.pipeline) if model_class == "tree" else None,
    }
    return render(request, "project2/dashboard.html", context)
