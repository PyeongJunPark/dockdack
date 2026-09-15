"""Conservative Korean ordinary-company universe from Kiwoom's stock master.

The ka10099 category codes and row fields are documented by Kiwoom here:
https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/examples/국내주식/종목정보/list_domestic_stocks.py

Exchange membership is necessary, not sufficient: ETF/REIT/etc. category
membership wins even when the same security also appears in the KOSPI list.
Common-share code shape and issuer/industry metadata provide additional guards;
names are used only as an extra exclusion, never as positive evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from threading import RLock
from typing import Any
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo

from dockdack.exceptions import BrokerAPIError
from dockdack.http import KiwoomHTTPClient


EQUITY_MARKETS = ("0", "10")
_EQUITY_MARKET_NAMES = {
    "0": frozenset({"거래소", "코스피", "유가증권", "KOSPI"}),
    "10": frozenset({"코스닥", "KOSDAQ"}),
}
# Observed ka10099 issuer classes (demo master, 2026-09-15). Unknown new classes
# are not silently treated as ordinary businesses; refresh this set explicitly.
_COMPANY_CLASSES = frozenset({"", "벤처기업", "신성장기업", "외국기업", "우량기업", "중견기업"})
# Kiwoom's documented product categories, including rights and gold as well as
# funds. Query every category so an overlapping KOSPI row cannot admit a product.
EXCLUDED_MARKETS = ("2", "3", "4", "5", "6", "7", "8", "9", "60", "70", "80", "90")
_MASTER_KEYS = (
    "code", "name", "listCount", "auditInfo", "regDay", "lastPrice", "state",
    "marketCode", "marketName", "upName", "upSizeName", "orderWarning",
    "companyClassName", "nxtEnable",
)
_REQUIRED_METADATA = ("name", "marketCode", "marketName", "upName", "companyClassName")
_EXCLUDED_LABEL = re.compile(
    r"ETF|ETN|ELW|REIT|SPAC|펀드|(?<!메)리츠|스팩|기업\s*인수\s*목적|"
    r"부동산\s*투자\s*회사|투자\s*회사|투융자|신주\s*인수권|우선주|"
    r"상장\s*지수|수익\s*증권|예탁\s*증권|금현물",
    re.IGNORECASE,
)
_PREFERRED_NAME = re.compile(r"(?:\d*우(?:[A-Z]|선)?|우선주)$", re.IGNORECASE)
_CACHE: WeakKeyDictionary[KiwoomHTTPClient, tuple[date, frozenset[str]]] = WeakKeyDictionary()
_CACHE_LOCK = RLock()


def _symbol(value: Any) -> str:
    if not isinstance(value, str):
        raise BrokerAPIError("국내 종목 분류 응답의 종목코드를 확인할 수 없습니다.")
    symbol = value.strip().upper()
    if len(symbol) == 7 and symbol[0] in {"A", "Q", "J"}:
        symbol = symbol[1:]
    if re.fullmatch(r"[0-9A-Z]{6}", symbol) is None:
        raise BrokerAPIError("국내 종목 분류 응답의 종목코드 형식이 잘못되었습니다.")
    return symbol


def is_domestic_common_row(row: Mapping[str, Any], *, excluded_symbols: frozenset[str] = frozenset()) -> bool:
    """Accept only a classified KOSPI/KOSDAQ ordinary-company master row.

    Missing/malformed classification data is ineligible. A present but blank
    companyClassName is not treated as a positive ordinary-company flag.
    A six-character common-share code ending in 0 is an additional conservative
    restriction, not a replacement for the complete product-category exclusion.
    """
    if not isinstance(row, Mapping):
        return False
    try:
        symbol = _symbol(row.get("code"))
    except BrokerAPIError:
        return False
    if symbol in excluded_symbols or re.fullmatch(r"[0-9A-Z]{5}0", symbol) is None:
        return False
    if any(not isinstance(row.get(key), str) for key in _REQUIRED_METADATA):
        return False
    market_code = row["marketCode"].strip()
    if market_code not in EQUITY_MARKETS:
        return False
    if row["marketName"].strip().upper() not in _EQUITY_MARKET_NAMES[market_code]:
        return False
    if row["companyClassName"].strip() not in _COMPANY_CLASSES:
        return False
    # Current responses also expose kind=A for stock certificates. It is not in
    # the older official row schema, so treat a supplied non-A value as a veto.
    if "kind" in row and row["kind"] != "A":
        return False
    if any(not row[key].strip() for key in ("name", "marketName", "upName")):
        return False
    # ka10099 does not guarantee an isEtf field. If supplied, only an explicit
    # false is acceptable; an unknown value must not silently become false.
    if "isEtf" in row:
        flag = row["isEtf"]
        false_flag = flag is False or (type(flag) is int and flag == 0) or (
            isinstance(flag, str) and flag in ("0", "N", "n", "false", "False")
        )
        if not false_flag:
            return False
    for key in ("name", "upName", "companyClassName", "marketName", "upSizeName", "state"):
        value = row.get(key, "")
        if not isinstance(value, str) or _EXCLUDED_LABEL.search(value):
            return False
    if _PREFERRED_NAME.search(row["name"].strip()):
        return False
    return True


def _master_rows(http: KiwoomHTTPClient, market_code: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    complete = False
    for page in http.iter_pages(
        api_id="ka10099", path="/api/dostk/stkinfo", body={"mrkt_tp": market_code}, max_pages=20,
    ):
        data = page.body.get("list")
        # Kiwoom returns an explicit successful null list for an empty rights
        # category. Missing fields, incomplete pages, and empty equity lists
        # still fail closed; do not normalize all malformed data to [].
        if (market_code in EXCLUDED_MARKETS and "list" in page.body and data is None
                and page.body.get("return_code") in (0, "0") and not page.has_next):
            data = []
        if not isinstance(data, list):
            raise BrokerAPIError("국내 종목 분류 목록이 누락되었거나 형식이 잘못되었습니다.")
        for value in data:
            if isinstance(value, dict):
                row = dict(value)
            elif isinstance(value, (list, tuple)) and len(value) == len(_MASTER_KEYS):
                row = dict(zip(_MASTER_KEYS, value, strict=True))
            else:
                raise BrokerAPIError("국내 종목 분류 행의 형식을 확인할 수 없습니다.")
            try:
                row["code"] = _symbol(row.get("code"))
            except BrokerAPIError:
                raw_code = row.get("code")
                # Rights (0036221D), funds (7010003), and gold (M04020000)
                # have longer identities. These cannot equal a six-character
                # stock candidate; never truncate them into a different stock.
                if (market_code in EXCLUDED_MARKETS and isinstance(raw_code, str)
                        and re.fullmatch(r"[0-9A-Z]{7,12}", raw_code.strip().upper())):
                    continue
                raise
            rows.append(row)
        complete = not page.has_next
    if not complete or (market_code in EQUITY_MARKETS and not rows):
        raise BrokerAPIError("국내 종목 분류 목록의 전체 조회를 확인하지 못했습니다.")
    return tuple(rows)


def domestic_common_symbols(
    http: KiwoomHTTPClient, *, as_of: date | None = None, force: bool = False,
) -> frozenset[str]:
    """Get a complete, per-client/per-Seoul-date cached common-company set.

    New days and forced refreshes never fall back to an old or partial set on
    errors. This performs master-data reads only; no quotes/accounts/orders.
    ``as_of`` identifies cache freshness, not a historical master-data request.
    """
    selected_day = as_of or datetime.now(ZoneInfo("Asia/Seoul")).date()
    if type(selected_day) is not date:
        raise ValueError("종목 분류 기준일은 date여야 합니다.")
    with _CACHE_LOCK:
        cached = _CACHE.get(http)
        if not force and cached is not None and cached[0] == selected_day:
            return cached[1]
        # A failed forced refresh must also invalidate today's earlier cache.
        _CACHE.pop(http, None)
        candidates: dict[str, dict[str, Any]] = {}
        ambiguous: set[str] = set()
        for market_code in EQUITY_MARKETS:
            for row in _master_rows(http, market_code):
                symbol = row["code"]
                if row.get("marketCode") != market_code:
                    ambiguous.add(symbol)
                previous = candidates.get(symbol)
                if previous is not None and previous != row:
                    ambiguous.add(symbol)
                candidates[symbol] = row
        excluded = set(ambiguous)
        for market_code in EXCLUDED_MARKETS:
            excluded.update(row["code"] for row in _master_rows(http, market_code))
        denied = frozenset(excluded)
        result = frozenset(symbol for symbol, row in candidates.items()
                           if is_domestic_common_row(row, excluded_symbols=denied))
        if not result:
            raise BrokerAPIError("분류가 확인된 국내 일반기업 보통주가 없습니다. 관심종목 선정을 중단합니다.")
        _CACHE[http] = (selected_day, result)
        return result
