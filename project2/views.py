from django.shortcuts import render

from .services.data import (
    get_penguin_data, SPECIES_ORDER, INPUT_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
)
from .services.grids import unconstrained_tree_entry
from .services.treeviz import tree_to_png_base64, tree_to_text

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
