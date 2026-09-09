"""AG News dataset loading for Project 3 (Learning-to-Defer).

120,000 training and 7,600 test news articles in four classes (World, Sports,
Business, Sci/Tech). The dataset ships as ``data/agnews.csv.gz``, so a fresh
clone needs no network. Loaded from, in order: that file, a local joblib cache
if an earlier run left one, then the HuggingFace Hub.

The test split is used only for evaluation, never for model selection or
active-learning query choice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from django.conf import settings

# Canonical class order (matches the HF ClassLabel feature).
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
N_CLASSES = len(CLASS_NAMES)

DATA_FILE = os.path.join(settings.BASE_DIR, "data", "agnews.csv.gz")

# Cache left by earlier runs; still read if present.
CACHE_DIR = os.path.join(settings.BASE_DIR, "project3", "data_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "agnews.joblib")


@dataclass
class AGNewsData:
    X_train: list      # list[str] of article texts
    y_train: np.ndarray
    X_test: list
    y_test: np.ndarray
    class_names: list

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _load_from_csv():
    """Read the shipped dataset."""
    import pandas as pd

    df = pd.read_csv(DATA_FILE)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    return (
        train["text"].astype(str).tolist(),
        train["label"].to_numpy(dtype=int),
        test["text"].astype(str).tolist(),
        test["label"].to_numpy(dtype=int),
    )


def _load_from_huggingface():
    """Download AG News from the HuggingFace Hub and extract plain arrays."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    X_train = list(ds["train"]["text"])
    y_train = np.asarray(ds["train"]["label"], dtype=int)
    X_test = list(ds["test"]["text"])
    y_test = np.asarray(ds["test"]["label"], dtype=int)
    return X_train, y_train, X_test, y_test


def _write_csv(X_train, y_train, X_test, y_test):
    """Persist a downloaded copy for later runs."""
    import pandas as pd

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    pd.DataFrame({
        "split": ["train"] * len(X_train) + ["test"] * len(X_test),
        "label": list(y_train) + list(y_test),
        "text": list(X_train) + list(X_test),
    }).to_csv(DATA_FILE, index=False, compression="gzip")


def _load_raw():
    """Return (X_train, y_train, X_test, y_test) from the first source available."""
    if os.path.exists(DATA_FILE):
        return _load_from_csv()

    if os.path.exists(CACHE_FILE):
        import joblib
        blob = joblib.load(CACHE_FILE)
        return blob["X_train"], blob["y_train"], blob["X_test"], blob["y_test"]

    X_train, y_train, X_test, y_test = _load_from_huggingface()
    _write_csv(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


@lru_cache(maxsize=1)
def get_agnews() -> AGNewsData:
    """Load AG News (cached in-memory and on disk). The full dataset."""
    X_train, y_train, X_test, y_test = _load_raw()
    return AGNewsData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=list(CLASS_NAMES),
    )


def class_distribution(y: np.ndarray) -> dict:
    """Count of examples per class label index."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=N_CLASSES)
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)}
