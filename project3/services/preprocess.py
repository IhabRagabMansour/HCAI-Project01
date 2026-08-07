"""Lightweight text preprocessing for Project 3.

The preprocessing is applied only inside the TF-IDF pipeline. Raw article text
remains untouched in the data service so it can still be displayed exactly as
originally loaded.
"""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

_TOKEN_RE = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def _stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "this", "these", "those",
        }


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


def _wordnet_pos(token: str) -> str:
    try:
        tag = wordnet.synsets(token)
    except LookupError:
        return "n"
    return "v" if tag and any(ss.pos() == "v" for ss in tag) else "n"


def preprocess_text(text: str) -> str:
    """Normalize text for TF-IDF: lowercase, stop-word removal, lemmatize,
    then stem. Returns a space-separated token string."""
    tokens = _TOKEN_RE.findall(text.lower())
    stops = _stop_words()
    lemmatizer = _lemmatizer()
    stemmer = _stemmer()

    processed = []
    for token in tokens:
        if token in stops:
            continue
        lemma = lemmatizer.lemmatize(token, pos=_wordnet_pos(token))
        stemmed = stemmer.stem(lemma)
        if stemmed:
            processed.append(stemmed)
    return " ".join(processed)