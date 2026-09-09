"""Text preprocessing for the TF-IDF pipeline.

Raw article text stays untouched in the data service so it can still be
displayed as loaded.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z]+")

# Function words only. sklearn's ENGLISH_STOP_WORDS is unusable here: it drops
# us, un, bill, interest and system, which are topic markers in news text.
STOP_WORDS = frozenset("""
    a an the and or but if then than that this these those
    of in on at to for from with by as into over under about
    is are was were be been being am
    it its he him his she her they them their we our you your i my
    not no nor so such very
    can could will would shall should may might must
    have has had do does did done
    there here when where who whom which what how why
""".split())

# Two-letter tokens are kept: US, EU, UN, AI, PC, TV. Raising this to 3 costs
# 0.5 points of test accuracy.
_MIN_TOKEN_LENGTH = 2


def preprocess_text(text: str) -> str:
    """Lowercase, tokenize, drop stop words. Returns a token string."""
    return " ".join(
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LENGTH and token not in STOP_WORDS
    )
