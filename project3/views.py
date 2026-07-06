import numpy as np
from django.shortcuts import render

from .services.data import get_agnews, class_distribution, CLASS_NAMES
from .services.baseline import get_baseline_eval, MODEL_DESCRIPTION
from .services.expert import get_expert_eval, get_expert_test_predictions


def _truncate(text, n=160):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


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


def expert(request):
    """Task 2: simulated expert — description, accuracy, class-wise strengths."""
    ev = get_expert_eval()
    data = get_agnews()

    # A few illustrative test articles: where deferring to the expert helps
    # (expert right, classifier wrong) and where it hurts (classifier right,
    # expert wrong).
    clf_pred = np.asarray(get_baseline_eval()["y_pred"])
    exp_pred = np.asarray(get_expert_test_predictions())
    y_test = np.asarray(data.y_test)

    exp_correct = exp_pred == y_test
    clf_correct = clf_pred == y_test
    good = np.where(exp_correct & ~clf_correct)[0][:3]
    bad = np.where(clf_correct & ~exp_correct)[0][:3]

    def _examples(indices):
        rows = []
        for i in indices:
            rows.append({
                "text": _truncate(data.X_test[i]),
                "true": CLASS_NAMES[int(y_test[i])],
                "expert": CLASS_NAMES[int(exp_pred[i])],
                "classifier": CLASS_NAMES[int(clf_pred[i])],
            })
        return rows

    context = {
        "title": "Task 2 — Simulated Expert",
        "description": ev["description"],
        "competence_topics": ev["competence_topics"],
        "overall_accuracy": ev["overall_accuracy"],
        "p_high": ev["p_high"],
        "p_low": ev["p_low"],
        "comparison": ev["comparison"],
        "good_examples": _examples(good),
        "bad_examples": _examples(bad),
        "compare_payload": {
            "labels": ev["class_names"],
            "expert": ev["per_class_accuracy"],
            "classifier": ev["classifier_per_class"],
        },
    }
    return render(request, "project3/expert.html", context)
