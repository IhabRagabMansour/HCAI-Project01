"""sklearn Pipeline construction helpers.

Builds a single object that bundles preprocessing (per-column transformations)
with the final estimator. Saving/loading the whole Pipeline means single-row
prediction at evaluation time becomes a one-liner: ``pipeline.predict(new_row)``.

Used by preprocessing, training, evaluation, and prediction services.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler,
)


# ── Per-branch step builders ────────────────────────────────────────────────

def _numeric_steps(config) -> list:
    """Return [(name, transformer), ...] for the numeric branch of a ColumnTransformer."""
    steps: list = []

    # Imputation. "drop" handled upstream of the pipeline (rows are removed
    # before fitting), so no imputer is needed in that case.
    if config.missing_strategy == "mean_mode":
        steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif config.missing_strategy == "zero_empty":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))

    # Scaling
    if config.scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif config.scaling == "minmax":
        steps.append(("scaler", MinMaxScaler()))

    if not steps:
        # Pipeline must contain at least one step; passthrough is a no-op.
        steps.append(("passthrough", "passthrough"))
    return steps


def _categorical_steps(config) -> list:
    """Return [(name, transformer), ...] for the categorical branch."""
    steps: list = []

    if config.missing_strategy == "mean_mode":
        steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    elif config.missing_strategy == "zero_empty":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="")))

    if config.categorical_encoding == "onehot":
        # handle_unknown="ignore" — at predict time, unseen categories produce
        # all-zero rows instead of crashing. sparse_output=False keeps the
        # output as a dense numpy array (simpler downstream).
        steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif config.categorical_encoding == "label":
        # OrdinalEncoder is sklearn's per-feature label encoder.
        steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

    if not steps:
        steps.append(("passthrough", "passthrough"))
    return steps


# ── Public construction API ─────────────────────────────────────────────────

def build_preprocessing(config, num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build an un-fit ColumnTransformer matching the experiment config.

    Args:
        config: ExperimentConfig (uses missing_strategy, categorical_encoding, scaling)
        num_cols: names of numeric columns to transform
        cat_cols: names of categorical columns to transform

    Returns:
        ColumnTransformer that, once fit, transforms a DataFrame to a numeric matrix.
        Columns not listed in either group are dropped.
    """
    transformers: list = []

    if num_cols:
        transformers.append(("num", Pipeline(_numeric_steps(config)), list(num_cols)))

    # When the encoding strategy is "drop", categorical columns are simply
    # excluded from the ColumnTransformer.
    if cat_cols and config.categorical_encoding != "drop":
        transformers.append(("cat", Pipeline(_categorical_steps(config)), list(cat_cols)))

    if not transformers:
        # Edge case: nothing to transform (e.g. all categorical columns + drop).
        # Returning an empty CT with passthrough remainder is a safe no-op.
        return ColumnTransformer(
            [], remainder="passthrough", verbose_feature_names_out=False,
        )

    # verbose_feature_names_out=False keeps original column names instead of
    # prefixing with the transformer name (e.g. "x1" not "num__x1").
    return ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False,
    )


def build_sampler(oversampling: str, random_seed: int = 42):
    """Return an imblearn sampler (or None when oversampling is disabled).

    - 'smote'        → SMOTE synthetic minority oversampling
    - 'random_over'  → simple random oversampling with replacement
    - 'none'         → None (no sampling step)
    """
    if oversampling == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=random_seed)
    if oversampling == "random_over":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(random_state=random_seed)
    return None


def build_full_pipeline(preprocessing, estimator, sampler=None):
    """Wrap (preprocessor, [sampler], estimator) into a single Pipeline.

    Without a sampler returns sklearn's Pipeline. With a sampler returns
    imblearn's Pipeline (a drop-in extension) so the sampler runs only at
    fit time, not at predict time.

    The returned pipeline can be fit on raw DataFrame input and persisted with
    joblib as a single artifact.
    """
    if sampler is None:
        return Pipeline([
            ("preprocessor", preprocessing),
            ("estimator", estimator),
        ])

    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([
        ("preprocessor", preprocessing),
        ("sampler", sampler),
        ("estimator", estimator),
    ])
