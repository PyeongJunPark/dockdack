"""Pure, offline catalog checks for an annotated training candidate universe.

``eligible`` means an ordinary-company/equity *candidate*, not a verified
security-master classification or historical point-in-time membership. In
particular, a US non-ETF issuer name cannot prove that a share is common stock.
ADR/REIT candidates are retained with flags. Explicit products are excluded;
name-based exclusions always carry HEURISTIC reasons. Unknown identity or
classification metadata goes to review. Nothing is renamed, fetched or written.

The caller supplies the dataset's market and separately validates every bar's
currency against that market. Catalogs do not themselves have a currency column;
if one is supplied here, it must match exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any


_EXCHANGES = {"domestic": frozenset({"KRX"}), "us": frozenset({"NA", "ND", "NY"})}
_CURRENCIES = {"domestic": "KRW", "us": "USD"}
_US_NAMES = {"ND": "NASDAQ", "NY": "NYSE", "NA": "AMEX"}
_DOMESTIC_NAMES = {"0": {"거래소", "코스피", "유가증권", "KOSPI"}, "10": {"코스닥", "KOSDAQ"}}
_DOMESTIC_PRODUCTS = {
    "2": "FUND", "3": "ELW", "4": "FUND", "5": "WARRANT", "7": "RIGHT",
    "8": "ETF", "9": "FUND", "60": "ETN", "70": "ETN", "80": "GOLD", "90": "ETN",
}
_COMPANY_CLASSES = {"", "벤처기업", "신성장기업", "외국기업", "우량기업", "중견기업"}
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}\Z")
_NAME_EXCLUSIONS = (
    ("ETF", re.compile(r"\bETF\b|상장\s*지수\s*펀드", re.I)),
    ("ETN", re.compile(r"\bETN\b|상장\s*지수\s*증권", re.I)),
    ("ELW", re.compile(r"\bELW\b", re.I)),
    ("WARRANT", re.compile(r"\bWARRANTS?\b|\bC/WTS\b|\bWTS\b|워런트|신주\s*인수권|\b\d+WR\b", re.I)),
    ("RIGHT", re.compile(r"\bRIGHTS\b|신주\s*인수권\s*증서", re.I)),
    ("PREFERRED", re.compile(r"\b(?:PREFERRED|PREFERENCE|PREF|PFD)\b|우선주", re.I)),
    # Do not mistake an issuer such as UNIT CORP or UNITED for a unit security.
    ("UNIT", re.compile(r"\bUNITS\b|\bUNIT\s+(?:\d|EACH\b|CONSISTING\b|OF\b)|\bUNIT$|유닛", re.I)),
    ("FUND", re.compile(r"\bFUNDS?\b|펀드|수익\s*증권|폐쇄형", re.I)),
    ("SPAC", re.compile(r"\bSPACS?\b|\bACQUISITION\s+(?:CORP(?:ORATION)?|CO(?:MPANY)?)\b|"
                        r"BLANK\s+CHECK|SPECIAL\s+PURPOSE\s+ACQUISITION|스팩|기업\s*인수\s*목적", re.I)),
    ("DEBT", re.compile(r"\b(?:DEBENTURES?|NOTES?|BONDS?)\b|후순위\s*채권", re.I)),
)
_ADR = re.compile(r"\b(?:ADR|ADS|DEPOSITARY|DEPOSITORY)\b", re.I)
_REIT = re.compile(r"\bREITS?\b|REAL\s+ESTATE\s+INVESTMENT\s+TRUST|리츠", re.I)


def _raw_object(value: Any) -> dict:
    def unique(pairs):
        result = {}
        for key, entry in pairs:
            if key in result:
                raise ValueError("duplicate catalog field")
            result[key] = entry
        return result

    def invalid_constant(_):
        raise ValueError("non-finite catalog constant")

    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError("missing raw catalog metadata")
    result = json.loads(value, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("raw catalog metadata is not an object")
    return result


def _flag(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value in {"0", "N", "n", "false", "False"}:
            return False
        if value in {"1", "Y", "y", "true", "True"}:
            return True
    return None


def classify_instrument(row: Mapping[str, Any], market: str) -> tuple[str, tuple[str, ...]]:
    """Classify one exact catalog row, without consulting APIs or other rows."""
    if not isinstance(market, str) or market not in _EXCHANGES:
        return "review", ("UNSUPPORTED_MARKET",)
    if not isinstance(row, Mapping):
        return "review", ("CATALOG_ROW_INVALID",)
    symbol, exchange = row.get("symbol"), row.get("exchange")
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        return "review", ("SYMBOL_INVALID",)
    if not isinstance(exchange, str) or exchange not in _EXCHANGES[market]:
        return "review", ("EXCHANGE_UNSUPPORTED",)
    if "market" in row and row["market"] != market:
        return "review", ("MARKET_MISMATCH",)
    if "currency" in row and row["currency"] != _CURRENCIES[market]:
        return "review", ("CURRENCY_MISMATCH",)
    try:
        raw = _raw_object(row.get("raw_json"))
    except (TypeError, ValueError):
        return "review", ("RAW_CATALOG_INVALID",)
    raw_symbol = raw.get("code" if market == "domestic" else "stk_cd")
    if raw_symbol != symbol:
        return "review", ("RAW_SYMBOL_MISMATCH",)
    for key in ("currency", "crnc_code"):
        if key in raw and raw[key] != _CURRENCIES[market]:
            return "review", ("CURRENCY_MISMATCH",)
    if "market" in raw and raw["market"] != market:
        return "review", ("MARKET_MISMATCH",)
    if "exchange" in raw and raw["exchange"] != exchange:
        return "review", ("RAW_EXCHANGE_MISMATCH",)
    name = row.get("name")
    if not isinstance(name, str) or not name.strip():
        return "review", ("ISSUER_NAME_MISSING",)
    raw_name = raw.get("name" if market == "domestic" else "stk_nm")
    if raw_name != name:
        return "review", ("RAW_NAME_MISMATCH",)
    english = row.get("english_name")
    if english is not None and not isinstance(english, str):
        return "review", ("ISSUER_NAME_INVALID",)
    if market == "us" and english != raw.get("stk_enm"):
        return "review", ("RAW_ENGLISH_NAME_MISMATCH",)
    flag = _flag(row.get("is_etf"))
    raw_flag = _flag(raw.get("isEtf"))
    if "isEtf" in raw and raw_flag is None:
        return "review", ("ETF_FLAG_UNKNOWN",)
    if row.get("is_etf") is not None and flag is None:
        return "review", ("ETF_FLAG_UNKNOWN",)
    if flag is not None and raw_flag is not None and flag != raw_flag:
        return "review", ("ETF_FLAG_CONFLICT",)

    category = row.get("catalog_market_code")
    if market == "domestic":
        if not isinstance(category, str) or category != raw.get("marketCode"):
            return "review", ("DOMESTIC_CATEGORY_MISMATCH",)
        if category in _DOMESTIC_PRODUCTS:
            return "excluded", ("CATALOG_CATEGORY_" + _DOMESTIC_PRODUCTS[category],)
    else:
        if raw.get("stex_tp") != exchange or category != exchange or row.get("listing_market") != exchange:
            return "review", ("RAW_EXCHANGE_MISMATCH",)
        if raw.get("mkgb") not in (None, "", _US_NAMES[exchange]):
            return "review", ("EXCHANGE_LABEL_CONFLICT",)
    if flag is True or raw_flag is True:
        return "excluded", ("EXPLICIT_ETF_FLAG",)

    names = [name.strip(), (english or "").strip()]
    if market == "domestic" and re.search(r"(?:\d*우(?:[A-Z]|선)?|우선주)$", names[0], re.I):
        return "excluded", ("HEURISTIC_NAME_PREFERRED", "NAME_SUBTYPE_NOT_MASTER_VERIFIED")
    reasons = ["HEURISTIC_NAME_" + subtype for subtype, pattern in _NAME_EXCLUSIONS
               if any(pattern.search(value) for value in names)]
    if reasons:
        return "excluded", tuple(reasons + ["NAME_SUBTYPE_NOT_MASTER_VERIFIED"])

    if market == "domestic":
        if category not in _DOMESTIC_NAMES:
            return "review", ("DOMESTIC_CATEGORY_OUTSIDE_CANDIDATE_SCOPE",)
        if (not isinstance(raw.get("marketName"), str) or raw["marketName"] not in _DOMESTIC_NAMES[category]
                or row.get("listing_market") != raw["marketName"]):
            return "review", ("DOMESTIC_MARKET_LABEL_CONFLICT",)
        if raw.get("kind") != "A":
            return "review", ("DOMESTIC_SECURITY_KIND_UNVERIFIED",)
        if not isinstance(raw.get("companyClassName"), str) or raw["companyClassName"] not in _COMPANY_CLASSES:
            return "review", ("DOMESTIC_COMPANY_CLASS_UNVERIFIED",)
        if not isinstance(raw.get("upName"), str) or not raw["upName"].strip():
            return "review", ("DOMESTIC_INDUSTRY_MISSING",)
        if re.fullmatch(r"[0-9A-Z]{5}0", symbol) is None:
            return "review", ("DOMESTIC_COMMON_CODE_UNVERIFIED",)
        return "eligible", ("DOMESTIC_COMPANY_CANDIDATE", "CURRENT_CATALOG_NOT_POINT_IN_TIME")

    if flag is not False or raw_flag is not False:
        return "review", ("ETF_FLAG_UNKNOWN",)
    reasons = ["NON_ETF_EQUITY_CANDIDATE", "SHARE_SUBTYPE_NOT_VERIFIED", "CURRENT_CATALOG_NOT_POINT_IN_TIME"]
    labels = " ".join(names + [str(raw.get("upgb") or "")])
    if _ADR.search(labels):
        reasons.append("ADR_CANDIDATE_RETAINED")
    if _REIT.search(labels):
        reasons.append("REIT_CANDIDATE_RETAINED")
    if not isinstance(raw.get("upgb"), str) or not raw["upgb"].strip():
        reasons.append("INDUSTRY_UNAVAILABLE")
    return "eligible", tuple(reasons)


def resolve_catalog_key(bar_symbol: str, exchange: str, catalog: Mapping) -> tuple[str, tuple, tuple[str, ...]]:
    """Return (exact/review/missing, candidate (exchange,symbol) keys, reasons).

    The mapping must be from one market. Only ``exact`` is a usable join.
    Case-insensitive or other-exchange matches are suggestions for a human/
    authoritative-master review, never permission to rewrite or merge bars.
    """
    if (not isinstance(bar_symbol, str) or _SYMBOL.fullmatch(bar_symbol) is None
            or not isinstance(exchange, str) or exchange not in {"KRX", "NA", "ND", "NY"}):
        return "review", (), ("BAR_IDENTITY_INVALID",)
    exact = (exchange, bar_symbol)
    if exact in catalog:
        row = catalog[exact]
        if not isinstance(row, Mapping) or row.get("symbol") != bar_symbol or row.get("exchange") != exchange:
            return "review", (exact,), ("CATALOG_KEY_ROW_CONFLICT",)
        return "exact", (exact,), ("EXACT_CATALOG_KEY",)
    candidates = tuple(sorted(key for key in catalog
                              if isinstance(key, tuple) and len(key) == 2 and key[0] == exchange
                              and isinstance(key[1], str) and key[1].casefold() == bar_symbol.casefold()))
    if candidates:
        return "review", candidates, ("CASE_ONLY_CANDIDATE_REQUIRES_VERIFICATION",)
    other = tuple(sorted(key for key in catalog
                         if isinstance(key, tuple) and len(key) == 2 and key[0] != exchange
                         and isinstance(key[0], str) and key[1] == bar_symbol))
    if other:
        return "review", other, ("OTHER_EXCHANGE_CANDIDATE_REQUIRES_VERIFICATION",)
    return "missing", (), ("CATALOG_KEY_NOT_FOUND",)
