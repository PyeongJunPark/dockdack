"""Conservative public-source eligibility for US operating-company equities.

Kiwoom's ``isEtf=N`` is not a common-share classification. Combine KIS's
public security/DR master with Nasdaq's directory and industry metadata.
Unknown or conflicting metadata is excluded. This deliberately loses some
valid companies and is not a guarantee that upstream classifications are
perfect (a SPAC can, for example, have an incorrect industry assignment).

Sources/definitions:
https://github.com/koreainvestment/open-trading-api/blob/main/stocks_info/overseas_stock_code.py
https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs
https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true

No brokerage credentials, account endpoints, or orders are used here.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .exceptions import BrokerAPIError


CACHE_PATH = Path(".dockdack/us_equity_universe.json")
KIS_URL = "https://new.real.download.dws.co.kr/common/master/{exchange}mst.cod.zip"
DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
_VERSION = 1
_EXCHANGES = {"nas": "ND", "nys": "NY", "ams": "NA"}
_DIRECTORY_EXCHANGES = {"Q": "ND", "N": "NY", "A": "NA"}
_MAX_BYTES = 20_000_000
_LOCK = threading.RLock()
_MEMORY: tuple[str, str, frozenset[tuple[str, str]]] | None = None
_FAILURE: tuple[str, str, float] | None = None

_COMMON = re.compile(r"\b(?:common\s+(?:stocks?|shares?)|ordinary\s+shares?|capital\s+stock)\b", re.I)
_EXCLUDED_NAME = re.compile(
    r"\b(?:ETF|ETN|ETP|REITs?|SPACs?|funds?|trusts?|warrants?|rights?|units?|"
    r"preferred|preference|depositary|depository|debentures?|notes?|bonds?|"
    r"acquisition|blank\s+checks?|special\s+purpose|beneficial\s+interest)\b", re.I)
_EXCLUDED_INDUSTRY = re.compile(
    r"blank\s*checks?|trust|REIT|real\s+estate\s+investment|closed[ -]?end|"
    r"investment\s+(?:fund|compan|vehicle)|exchange[ -]?traded|"
    # Includes BDCs such as ARCC; also excludes some genuine consumer lenders.
    r"finance\s*:\s*consumer\s+services", re.I)


def _public_symbol(symbol: str) -> str:
    """Join explicit class notation only; never conflate BRKb with BRKB."""
    symbol = symbol.strip()
    if re.fullmatch(r"[A-Z]{1,10}[a-z]", symbol):
        return symbol[:-1] + "." + symbol[-1].upper()
    symbol = symbol.upper()
    if re.fullmatch(r"[A-Z]{1,10}/[A-Z]", symbol):
        symbol = symbol.replace("/", ".")
    return symbol


def _put_unique(target: dict, key, value) -> None:
    if key in target and target[key] != value:
        raise ValueError("미국 종목 분류 자료에 상충하는 중복 종목이 있습니다.")
    target[key] = value


def _download(url: str) -> bytes:
    with requests.get(url, timeout=(5, 20), stream=True,
                      headers={"User-Agent": "dockdack-paper-equity-classifier/1.0",
                               "Accept": "application/json,text/plain,application/zip,*/*"}) as response:
        response.raise_for_status()
        parts, size = [], 0
        for part in response.iter_content(64 * 1024):
            size += len(part)
            if size > _MAX_BYTES:
                raise ValueError("미국 분류 자료의 크기 제한을 초과했습니다.")
            parts.append(part)
        if not size:
            raise ValueError("미국 분류 자료가 비어 있습니다.")
        return b"".join(parts)


def _kis_rows(data: bytes, exchange: str) -> dict[tuple[str, str], dict[str, str]]:
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(files) != 1 or not files[0].filename.lower().endswith("mst.cod"):
            raise ValueError("KIS 미국 종목 마스터 압축 형식이 변경되었습니다.")
        if files[0].file_size > _MAX_BYTES:
            raise ValueError("KIS 미국 종목 마스터의 크기 제한을 초과했습니다.")
        content = archive.read(files[0]).decode("cp949")
    for row in csv.reader(io.StringIO(content), delimiter="\t"):
        if not row:
            continue
        if len(row) != 24:
            raise ValueError("KIS 미국 종목 마스터 열 수가 변경되었습니다.")
        row = [value.strip() for value in row]
        symbol = _public_symbol(row[4])
        if not symbol or row[2].upper() != exchange.upper():
            raise ValueError("KIS 종목 또는 거래소를 확인할 수 없습니다.")
        _put_unique(result, (symbol, _EXCHANGES[exchange]), {
            "name": row[7], "type": row[8], "dr": row[17],
            "industry": row[19], "etp": row[22],
        })
    if not result:
        raise ValueError("KIS 미국 종목 마스터가 비어 있습니다.")
    return result


def _directory_rows(data: bytes) -> dict[tuple[str, str], dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter="|")
    required = {"Symbol", "Security Name", "Listing Exchange", "ETF", "Test Issue", "NextShares"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("Nasdaq 종목 디렉터리 형식이 변경되었습니다.")
    result, trailer = {}, False
    for row in reader:
        if any(str(value).startswith("File Creation Time:") for value in row.values()):
            trailer = True
            continue
        if trailer or None in row or any(row.get(field) is None for field in required):
            raise ValueError("Nasdaq 종목 디렉터리가 불완전합니다.")
        exchange = _DIRECTORY_EXCHANGES.get(row["Listing Exchange"])
        if exchange is None:  # Kiwoom's three requested listing markets only.
            continue
        symbol = _public_symbol(row["Symbol"])
        if not symbol:
            raise ValueError("Nasdaq 종목코드가 비어 있습니다.")
        _put_unique(result, (symbol, exchange), {
            "name": row["Security Name"].strip(), "etf": row["ETF"].strip(),
            "test": row["Test Issue"].strip(), "next": row["NextShares"].strip(),
        })
    if not trailer or not result:
        raise ValueError("Nasdaq 종목 디렉터리의 완료 표식을 확인할 수 없습니다.")
    return result


def _screener_rows(data: bytes) -> dict[str, dict[str, str]]:
    document = json.loads(data)
    if not isinstance(document, dict) or document.get("status", {}).get("rCode") != 200:
        raise ValueError("Nasdaq 업종정보 조회가 실패했습니다.")
    body = document.get("data")
    if not isinstance(body, dict):
        raise ValueError("Nasdaq 업종정보 응답을 확인할 수 없습니다.")
    rows = body.get("rows", body.get("table", {}).get("rows") if isinstance(body.get("table"), dict) else None)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nasdaq 업종정보가 비어 있습니다.")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("symbol"), str):
            raise ValueError("Nasdaq 업종정보의 종목코드를 확인할 수 없습니다.")
        symbol = _public_symbol(row["symbol"])
        if not symbol:
            raise ValueError("Nasdaq 업종정보의 종목코드가 비어 있습니다.")
        fields = {name: row.get(name, "") for name in ("name", "industry", "sector")}
        if not all(isinstance(value, str) for value in fields.values()):
            raise ValueError("Nasdaq 업종정보의 자료형을 확인할 수 없습니다.")
        _put_unique(result, symbol, {key: value.strip() for key, value in fields.items()})
    return result


def _eligible(kis: dict, directory: dict, screen: dict) -> bool:
    if not kis or not directory or not screen:
        return False
    if kis.get("type") != "2" or kis.get("dr") != "N" or kis.get("etp") != "":
        return False
    industry_code = kis.get("industry", "")
    if not industry_code.isdigit() or int(industry_code) <= 0:
        return False
    if any(directory.get(flag) != "N" for flag in ("etf", "test", "next")):
        return False
    if not screen.get("industry") or not screen.get("sector"):
        return False
    if _EXCLUDED_INDUSTRY.search(screen["industry"]):
        return False
    names = [kis.get("name", ""), directory.get("name", ""), screen.get("name", "")]
    if any(not name or _EXCLUDED_NAME.search(name) for name in names):
        return False
    # Both independent Nasdaq representations must positively identify shares.
    return bool(_COMMON.search(names[1]) and _COMMON.search(names[2]))


def _fetch_universe() -> frozenset[tuple[str, str]]:
    kis = {}
    for exchange in _EXCHANGES:
        kis.update(_kis_rows(_download(KIS_URL.format(exchange=exchange)), exchange))
    directory = _directory_rows(_download(DIRECTORY_URL))
    screener = _screener_rows(_download(SCREENER_URL))
    eligible = frozenset(key for key, row in kis.items()
                         if _eligible(row, directory.get(key, {}), screener.get(key[0], {})))
    if not eligible:
        raise ValueError("교차 검증된 미국 일반주 종목이 없습니다.")
    return eligible


def _today() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _read_cache(path: Path, day: str) -> frozenset[tuple[str, str]] | None:
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != _VERSION or document.get("day") != day:
            return None
        rows = document["eligible"]
        if not isinstance(rows, list) or not rows:
            return None
        if any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
               or not row[0] or row[0] != _public_symbol(row[0])
               or row[1] not in _EXCHANGES.values() for row in rows):
            return None
        result = frozenset(tuple(row) for row in rows)
        return result if len(result) == len(rows) else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _write_cache(path: Path, day: str, eligible: frozenset[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"version": _VERSION, "day": day, "eligible": sorted(eligible)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _universe() -> frozenset[tuple[str, str]]:
    global _MEMORY, _FAILURE
    day, path = _today(), CACHE_PATH.resolve()
    identity = str(path)
    with _LOCK:
        if _MEMORY is not None and _MEMORY[:2] == (identity, day):
            return _MEMORY[2]
        cached = _read_cache(path, day)
        if cached is not None:
            _MEMORY = (identity, day, cached)
            return cached
        if _FAILURE is not None and _FAILURE[:2] == (identity, day) and time.monotonic() - _FAILURE[2] < 60:
            raise BrokerAPIError("미국 일반주 분류 자료를 확인하지 못했습니다. 60초 후 다시 시도하세요.")
        try:
            eligible = _fetch_universe()
            _write_cache(path, day, eligible)
        except Exception as exc:
            _FAILURE = (identity, day, time.monotonic())
            raise BrokerAPIError("미국 일반주 분류 자료 조회/저장 실패로 종목 선정을 차단했습니다.") from exc
        _FAILURE = None
        _MEMORY = (identity, day, eligible)
        return eligible


def us_common_symbols(http, candidates: tuple[tuple[str, str], ...]) -> frozenset[tuple[str, str]]:
    """Return original broker identifiers passing today's conservative filter.

    ``http`` is accepted for rank-service compatibility but never used: broker
    credentials must not be sent to these independent public data providers.
    A failed provider raises BrokerAPIError, not an empty successful universe.
    """
    del http
    if not candidates:
        return frozenset()
    universe = _universe()
    return frozenset((symbol, exchange) for symbol, exchange in candidates
                     if exchange in _EXCHANGES.values() and (_public_symbol(symbol), exchange) in universe)


def is_us_common(http, symbol: str, exchange: str) -> bool:
    return (symbol, exchange) in us_common_symbols(http, ((symbol, exchange),))
