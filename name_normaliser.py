"""
Name Normaliser — canonical company name generation for deduplication.

Strips corporate suffixes and reduces company names to a short uppercase
form (up to 3 words) so that the same company appearing in different ETF
providers can be merged into a single holding.
"""

import re

import pandas as pd

_STRIP_SUFFIXES = re.compile(
    r"\b(?:INC|CORP|CORPORATION|LTD|LIMITED|PLC|COMPANY|AG|SA|NV|"
    r"GROUP|HOLDINGS|HOLDING|SE|CLASS\s+[A-C]|CL\s+[A-C]|CO(?!\w))\b"
)

# Maximum number of words retained after normalisation.
# Three words improves disambiguation (e.g. "TAIWAN SEMICONDUCTOR MANUFACTURING"
# vs "TAIWAN MOBILE") while still being short enough for reliable deduplication.
MAX_NORMALISED_WORDS = 3


def normalise_name(name: str) -> str:
    """
    Reduce a company name to a short canonical form for deduplication.

    Steps:
        1. Return empty string for null/empty input.
        2. Remove all non-alphanumeric characters except spaces.
        3. Convert to uppercase.
        4. Strip corporate suffixes (INC, CORP, LTD, PLC, AG, SA, etc.).
        5. Reduce to the first MAX_NORMALISED_WORDS words.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9 ]", "", text.upper())
    text = _STRIP_SUFFIXES.sub("", text)
    words = text.split()
    return " ".join(words[:MAX_NORMALISED_WORDS])
