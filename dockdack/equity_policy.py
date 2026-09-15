"""Common-company eligibility shared by rankings and the final automatic-order gate."""

from datetime import datetime
from threading import RLock
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo

from dockdack.equity_universe import domestic_common_symbols
from dockdack.exceptions import BrokerAPIError
from dockdack.models import Market
from dockdack.symbols import normalize_symbol


_US_CACHE = WeakKeyDictionary()
_LOCK = RLock()


def _us_broker_common_candidates(http):
    day = datetime.now(ZoneInfo("America/New_York")).date()
    with _LOCK:
        cached = _US_CACHE.get(http)
        if cached and cached[0] == day:
            return cached[1]
        _US_CACHE.pop(http, None)
        candidates, excluded = set(), set()
        for api in ("usa10099", "usa10104"):
            complete = False
            count = 0
            for page in http.iter_pages(api_id=api, path="/api/us/stkinfo", body={"stex_tp": "%"}, max_pages=20):
                rows = page.body.get("list")
                if not isinstance(rows, list):
                    raise BrokerAPIError("미국 종목/ETF 분류 목록을 확인할 수 없습니다.")
                for row in rows:
                    if not isinstance(row, dict):
                        raise BrokerAPIError("미국 종목 분류 행이 잘못되었습니다.")
                    raw = row.get("stk_cd") or row.get("code")
                    if row.get("stk_cd") and row.get("code") and row["stk_cd"] != row["code"]:
                        raise BrokerAPIError("미국 종목 분류 코드가 서로 다릅니다.")
                    if not isinstance(raw, str) or not raw.strip():
                        raise BrokerAPIError("미국 종목 분류 코드가 없습니다.")
                    exchange = row.get("stex_tp")
                    if exchange not in {"ND", "NY", "NA"}:
                        continue
                    key = (normalize_symbol(raw), exchange)
                    if api == "usa10099" and row.get("isEtf") == "N":
                        candidates.add(key)
                    else:
                        excluded.add(key)
                    count += 1
                complete = not page.has_next
            if not complete or count == 0:
                raise BrokerAPIError("미국 종목 분류 전체 조회를 확인하지 못했습니다.")
        result = frozenset(candidates - excluded)
        if not result:
            raise BrokerAPIError("미국 주식 후보 분류가 비어 있습니다.")
        _US_CACHE[http] = (day, result)
        return result


def common_equities(http, market: Market, candidates):
    """Return eligible (broker symbol, exchange) pairs; unknowns are excluded."""
    candidates = tuple(candidates)
    if market is Market.DOMESTIC:
        common = domestic_common_symbols(http)
        return frozenset((symbol, exchange) for symbol, exchange in candidates
                         if exchange == "KRX" and symbol in common)
    from dockdack.us_equity_universe import us_common_symbols
    broker_common = _us_broker_common_candidates(http)
    eligible = tuple(key for key in candidates if key in broker_common)
    return us_common_symbols(http, eligible) if eligible else frozenset()
