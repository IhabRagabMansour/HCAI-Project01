import numpy as np
from django.shortcuts import render

from .services.data import get_agnews, class_distribution, CLASS_NAMES
from .services.baseline import get_baseline_eval, MODEL_DESCRIPTION
from .services.expert import get_expert_eval, get_expert_test_predictions
from .services.defer import get_deferral
from .services.active import get_active


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


# Deferral-metric rows for the side-by-side table (key -> display label).
_DEFER_METRIC_ROWS = [
    ("team_accuracy", "Team accuracy"),
    ("deferral_rate", "Deferral rate"),
    ("accuracy_deferred", "Accuracy on deferred"),
    ("accuracy_kept", "Accuracy on kept"),
    ("useful_deferral_frac", "Useful deferrals"),
    ("harmful_deferral_frac", "Harmful deferrals"),
    ("expert_correct_when_deferred", "Expert correct when deferred"),
    ("classifier_correct_when_kept", "Classifier correct when kept"),
]


def defer(request):
    """Task 3: learning to defer — strategies, team metrics, oracle, examples."""
    r = get_deferral()
    adv = r["advantage"]
    conf = r["confidence"]

    metric_rows = [
        {"label": label, "advantage": adv[key], "confidence": conf[key]}
        for key, label in _DEFER_METRIC_ROWS
    ]

    def _examples(items):
        return [
            {
                "text": _truncate(ex["text"]),
                "true": ex["true"], "expert": ex["expert"], "classifier": ex["classifier"],
            }
            for ex in items
        ]

    context = {
        "title": "Task 3 — Learning to Defer",
        "classifier_accuracy": adv["classifier_accuracy"],
        "expert_accuracy": adv["expert_accuracy"],
        "oracle_accuracy": adv["oracle_accuracy"],
        "advantage_team": adv["team_accuracy"],
        "confidence_team": conf["team_accuracy"],
        "confidence_threshold": r["confidence_threshold"],
        "metric_rows": metric_rows,
        "deferred_examples": _examples(r["deferred_examples"]),
        "kept_examples": _examples(r["kept_examples"]),
        "summary_payload": {
            "labels": ["Classifier only", "Expert only", "Confidence team",
                       "Advantage team", "Oracle"],
            "values": [
                adv["classifier_accuracy"], adv["expert_accuracy"],
                conf["team_accuracy"], adv["team_accuracy"], adv["oracle_accuracy"],
            ],
        },
    }
    return render(request, "project3/defer.html", context)


def active(request):
    """Task 4: active learning — query the expert efficiently to learn deferral."""
    r = get_active()
    checkpoints = r["checkpoints"]

    curve_rows = [
        {
            "n": checkpoints[i],
            "uncertainty": r["uncertainty_curve"][i]["team_accuracy"],
            "random": r["random_curve"][i]["team_accuracy"],
        }
        for i in range(len(checkpoints))
    ]

    context = {
        "title": "Task 4 — Active Learning",
        "pool_size": r["pool_size"],
        "budget": r["budget"],
        "n_random_runs": r["n_random_runs"],
        "classifier_only": r["classifier_only"],
        "full_supervision": r["full_supervision"],
        "target_accuracy": r["target_accuracy"],
        "unc_to_target": r["uncertainty_queries_to_target"],
        "rnd_to_target": r["random_queries_to_target"],
        "curve_rows": curve_rows,
        "curve_payload": {
            "checkpoints": checkpoints,
            "uncertainty": [p["team_accuracy"] for p in r["uncertainty_curve"]],
            "random": [p["team_accuracy"] for p in r["random_curve"]],
            "classifier_only": r["classifier_only"],
            "full_supervision": r["full_supervision"],
        },
    }
    return render(request, "project3/active.html", context)
