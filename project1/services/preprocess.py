from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


# ── Config dataclass ────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    missing_strategy: str = "mean_mode"    # drop | mean_mode | zero_empty
    categorical_encoding: str = "onehot"   # onehot | label | drop
    scaling: str = "standard"              # none | standard | minmax
    test_size: float = 0.2
    random_seed: int = 42
    stratify: bool = True


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_map: dict | None     # class name → int for string classification targets
    n_features_before: int
    n_features_after: int
    n_train: int
    n_test: int
    stratify_used: bool


# ── Step 1: missing values ──────────────────────────────────────────────────

def handle_missing(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "drop":
        return X.dropna().reset_index(drop=True)

    if strategy == "mean_mode":
        result = X.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
            else:
                mode = result[col].mode()
                result[col] = result[col].fillna(mode.iloc[0] if not mode.empty else "")
        return result

    if strategy == "zero_empty":
        result = X.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna("")
        return result

    raise ValueError(f"Unknown missing strategy: {strategy!r}")


# ── Step 2: categorical encoding ────────────────────────────────────────────

def encode_categorical(
    X: pd.DataFrame, strategy: str
) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    if strategy == "onehot":
        if cat_cols:
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
        else:
            X_enc = X.copy()
        return X_enc, list(X_enc.columns)

    if strategy == "label":
        X_enc = X.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        return X_enc, list(X_enc.columns)

    if strategy == "drop":
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X_enc = X[num_cols].copy()
        return X_enc, list(X_enc.columns)

    raise ValueError(f"Unknown encoding strategy: {strategy!r}")


# ── Step 3: target encoding ─────────────────────────────────────────────────

def encode_target(
    y: pd.Series, problem_type: str
) -> tuple[np.ndarray, dict | None]:
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        label_map = {cls: int(i) for i, cls in enumerate(le.classes_)}
        return y_enc, label_map
    return y.to_numpy(dtype=float), None


# ── Step 4: train/test split ─────────────────────────────────────────────────

def split_train_test(
    X: np.ndarray, y: np.ndarray,
    test_size: float, seed: int, stratify_flag: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if stratify_flag else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


# ── Step 5: feature scaling ──────────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray, X_test: np.ndarray, strategy: str
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "none":
        return X_train, X_test
    if strategy == "standard":
        scaler = StandardScaler()
    elif strategy == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling strategy: {strategy!r}")
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def prepare_experiment(
    df: pd.DataFrame,
    target_name: str,
    problem_type: str,
    config: ExperimentConfig,
) -> PreparedData:
    if target_name not in df.columns:
        raise ValueError(f"Target column {target_name!r} not found in DataFrame.")

    X = df.drop(columns=[target_name]).copy()
    y = df[target_name].copy()
    n_features_before = X.shape[1]

    # 1 — missing values (apply jointly so dropped rows stay aligned)
    combined = pd.concat([X, y.rename("__target__")], axis=1)
    combined = handle_missing(combined, config.missing_strategy)
    if combined.empty:
        raise ValueError("No rows remain after applying missing-value strategy.")
    y = combined.pop("__target__")
    X = combined

    # 2 — encode features
    X_enc, feature_names = encode_categorical(X, config.categorical_encoding)
    if X_enc.shape[1] == 0:
        raise ValueError("No features remain after encoding (all columns were categorical and strategy='drop').")
    n_features_after = X_enc.shape[1]

    # 3 — encode target
    y_enc, label_map = encode_target(y, problem_type)

    # 4 — train/test split (with stratify fallback)
    X_arr = X_enc.to_numpy(dtype=float)
    wants_stratify = config.stratify and problem_type == "classification"
    try:
        X_train, X_test, y_train, y_test = split_train_test(
            X_arr, y_enc, config.test_size, config.random_seed, stratify_flag=wants_stratify
        )
        stratify_used = wants_stratify
    except ValueError:
        X_train, X_test, y_train, y_test = split_train_test(
            X_arr, y_enc, config.test_size, config.random_seed, stratify_flag=False
        )
        stratify_used = False

    # 5 — scale (fit on train only to prevent data leakage)
    X_train, X_test = scale_features(X_train, X_test, config.scaling)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        label_map=label_map,
        n_features_before=n_features_before,
        n_features_after=n_features_after,
        n_train=len(y_train),
        n_test=len(y_test),
        stratify_used=stratify_used,
    )
