"""Decision-tree visualization helpers (Task 1).

Renders a fitted tree to a base64-encoded PNG (embedded directly in the page,
no media file needed) and to a plain-text representation. Uses the matplotlib
'Agg' backend so it works headless inside Django.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
from sklearn.tree import export_text, plot_tree

from .pipeline import feature_names_out


def tree_to_png_base64(pipeline) -> str:
    """Render the fitted DecisionTreeClassifier to a base64 PNG string."""
    estimator = pipeline.named_steps["estimator"]
    names = feature_names_out(pipeline)

    n_leaves = estimator.get_n_leaves()
    depth = max(estimator.get_depth(), 1)

    # Scale the canvas with the tree so nodes stay legible.
    width = max(10.0, n_leaves * 2.2)
    height = max(5.0, depth * 2.0)

    fig, ax = plt.subplots(figsize=(width, height))
    plot_tree(
        estimator,
        feature_names=names,
        class_names=list(estimator.classes_),
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=10,
        ax=ax,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def tree_to_text(pipeline) -> str:
    """Plain-text rendering of the tree (export_text)."""
    estimator = pipeline.named_steps["estimator"]
    names = feature_names_out(pipeline)
    return export_text(estimator, feature_names=names)
