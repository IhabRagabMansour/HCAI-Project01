from django.shortcuts import render

from .services.data import get_agnews, class_distribution, CLASS_NAMES


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
