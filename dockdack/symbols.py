"""Keep Kiwoom's case-sensitive share-class suffixes (for example BRKb)."""

import re


def normalize_symbol(value: str) -> str:
    value = value.strip()
    # Rank/master APIs return upper-case base + lower-case class suffix. Do not
    # turn BRKb into BRKB: Kiwoom's quote/chart endpoints treat them differently.
    if re.fullmatch(r"[A-Z]{1,10}[a-z]", value):
        return value
    return value.upper()
