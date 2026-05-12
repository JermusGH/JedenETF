"""
Name Normaliser — canonical company name generation for deduplication.

Strips corporate suffixes and reduces company names to a two-word uppercase
form so that the same company appearing in different ETF providers can be
merged into a single holding.
"""

import re

import pandas as pd

_STRIP_SUFFIXES = re.compile(
    r"\b(?:INC|CORP|CORPORATION|LTD|LIMITED|PLC|COMPANY|AG|SA|NV|"
    r"GROUP|HOLDINGS|HOLDING|SE|CLASS\s+[A-C]|CL\s+[A-C]|CO(?!\w))\b"
)


def normalise_name(name: str) -> str:
    """
    Reduce a company name to a short canonical form for deduplication.

    Steps:
        1. Return empty string for null/empty input.
        2. Remove all non-alphanumeric characters except spaces.
        3. Convert to uppercase.
        4. Strip corporate suffixes (INC, CORP, LTD, PLC, AG, SA, etc.).
        5. Reduce to the first two words only.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9 ]", "", text.upper())
    text = _STRIP_SUFFIXES.sub("", text)
    words = text.split()
    return " ".join(words[:2])
