"""Lambda-based model selection — the interpretability tradeoff (Task 2/3).

The interface lambda slider selects, *post-hoc*, among the already-trained grid
models. This is the single source of truth that every dashboard panel (model
display, counterfactuals, PDP, ALE) derives from, which is what guarantees the
"everything updates together when model class or lambda changes" requirement.

On the selection score
----------------------
The criterion is the **maximizer** of

    score = acc_test - lambda * Omega              # maximize

At ``lambda = 0`` this picks the most accurate model (ties broken toward the
simpler one); as ``lambda`` grows, complexity is penalized and progressively
simpler models win.

Note this is exactly equivalent to minimizing ``(1 - acc_test) + lambda*Omega``
(the two differ only by the additive constant 1), so the selected model is the
same either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grids import ModelEntry, get_grid

# Slider configuration. The penguins complexities span ~2..17, errors ~0..0.45,
# so this range traverses the full frontier from "most accurate" to "simplest".
LAMBDA_MIN = 0.0
LAMBDA_MAX = 0.05
LAMBDA_STEP = 0.001
LAMBDA_DEFAULT = 0.0

MODEL_CLASSES = [("tree", "Decision Tree"), ("logreg", "Logistic Regression")]


def selection_score(test_acc: float, complexity: int, lam: float) -> float:
    """Accuracy/complexity tradeoff score: acc_test - lambda*Omega.

    Higher values are better; the interface displays the maximizing model.
    """
    return test_acc - lam * complexity


def select_best(grid, lam: float) -> ModelEntry:
    """Return the grid entry MAXIMIZING the selection score. Ties are broken
    toward the *simpler* model, then the more accurate one."""
    return max(
        grid,
        key=lambda e: (
            selection_score(e.test_accuracy, e.complexity, lam),
            -e.complexity,         # prefer simpler on ties
            e.test_accuracy,       # then more accurate
        ),
    )


@dataclass
class SelectedModel:
    entry: ModelEntry
    lam: float
    model_class: str
    seed: int

    @property
    def pipeline(self):
        return self.entry.pipeline

    @property
    def test_accuracy(self) -> float:
        return self.entry.test_accuracy

    @property
    def complexity(self) -> int:
        return self.entry.complexity

    @property
    def param_value(self):
        return self.entry.param_value

    @property
    def param_name(self) -> str:
        return self.entry.param_name


def get_selected_model(model_class: str, lam: float, seed: int = 42) -> SelectedModel:
    """THE shared selection function. Every dashboard panel and AJAX endpoint
    rebuilds the active model from (model_class, lam, seed) via this function,
    guaranteeing consistency across the whole interface."""
    grid = get_grid(model_class, seed)
    entry = select_best(grid, lam)
    return SelectedModel(entry=entry, lam=lam, model_class=model_class, seed=seed)


def clamp_lambda(value) -> float:
    """Parse + clamp a lambda value from user input to [LAMBDA_MIN, LAMBDA_MAX]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return LAMBDA_DEFAULT
    return max(LAMBDA_MIN, min(LAMBDA_MAX, v))


def normalize_model_class(value) -> str:
    valid = {k for k, _ in MODEL_CLASSES}
    return value if value in valid else "tree"
