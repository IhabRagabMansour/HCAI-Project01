from django.shortcuts import render

from .services.data import (
    get_penguin_data, SPECIES_ORDER, INPUT_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BIOMETRIC_FEATURES,
)


def index(request):
    data = get_penguin_data()
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
