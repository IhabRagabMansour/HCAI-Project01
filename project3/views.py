from django.shortcuts import render

from .services.data import get_agnews, class_distribution, CLASS_NAMES
from .services.baseline import get_baseline_eval, MODEL_DESCRIPTION


def index(request):
    data = get_agnews()
    train_dist = class_distribution(data.y_train)
    test_dist = class_distribution(data.y_test)
    dist_rows = [
        {"cls": cls, "train": train_dist[cls], "test": test_dist[cls]}
        for cls in CLASS_NAMES
    ]
    context = {
        "title": "Project 3 — Active Learning for Learning-to-Defer",
        "class_names": CLASS_NAMES,
        "n_train": data.n_train,
        "n_test": data.n_test,
        "n_classes": data.n_classes,
        "dist_rows": dist_rows,
    }
    return render(request, "project3/index.html", context)


def baseline(request):
    """Task 1: baseline classifier — model, config, test accuracy, confusion matrix."""
    ev = get_baseline_eval()
    class_names = ev["class_names"]
    cm = ev["confusion_matrix"]

    # Build confusion-matrix rows (true class -> predicted counts) for the table,
    # flagging the diagonal and tracking each row's max for heatmap shading.
    cm_rows = []
    for i, row in enumerate(cm):
        row_total = sum(row)
        cells = [
            {
                "value": v,
                "is_diag": (i == j),
                "frac": (v / row_total if row_total else 0.0),
            }
            for j, v in enumerate(row)
        ]
        cm_rows.append({"true_class": class_names[i], "cells": cells})

    per_class_rows = [
        {"cls": class_names[i], "accuracy": ev["per_class_accuracy"][i]}
        for i in range(len(class_names))
    ]

    context = {
        "title": "Task 1 — Baseline Classifier",
        "model_description": MODEL_DESCRIPTION,
        "accuracy": ev["accuracy"],
        "n_train": ev["n_train"],
        "n_test": ev["n_test"],
        "class_names": class_names,
        "cm_rows": cm_rows,
        "per_class_rows": per_class_rows,
        "per_class_payload": {
            "labels": class_names,
            "accuracy": ev["per_class_accuracy"],
        },
    }
    return render(request, "project3/baseline.html", context)
