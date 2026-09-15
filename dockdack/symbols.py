"""Keep Kiwoom's case-sensitive share-class suffixes (for example BRKb)."""

import re


def normalize_symbol(value: str) -> str:
    value = value.strip()
    # Rank/master APIs return upper-case base + lower-case class suffix. Do not
    # turn BRKb into BRKB: Kiwoom's quote/chart endpoints treat them differently.
    if re.fullmatch(r"[A-Z]{1,10}[a-z]", value):
        return value
    return value.upper()


def normalize_us_exchange(value: str) -> str:
    """Normalize only specific venue labels; country/unknown names stay unknown."""
    if not isinstance(value, str):
        return ""
    value = value.strip().upper()
    return {"NASDAQ": "ND", "NYSE": "NY", "AMEX": "NA",
            "나스닥": "ND", "뉴욕": "NY", "아멕스": "NA"}.get(value, value)
