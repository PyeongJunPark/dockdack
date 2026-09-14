"""Kiwoom saved-condition lookup over WebSocket."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

from dockdack.config import KiwoomConfig
from dockdack.exceptions import BrokerAPIError, OptionalDependencyError
from dockdack.models import ConditionMatch, Market, SavedCondition


ConnectFactory = Callable[[str, float], Awaitable[Any]]


class KiwoomConditionClient:
    """Runs conditions saved in 영웅문4 or 영웅문Global."""

    def __init__(
        self,
        config: KiwoomConfig,
        token_provider: Callable[[], str],
        *,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        self.config = config
        self._token_provider = token_provider
        self._connect_factory = connect_factory or _default_connect

    async def list_conditions(self, market: Market | str) -> tuple[SavedCondition, ...]:
        selected_market = _as_market(market)
        trnm = "CNSRLST" if selected_market is Market.DOMESTIC else "GCNSRLST"
        response = await self._request_once(selected_market, {"trnm": trnm})
        result: list[SavedCondition] = []
        for row in _records(response.get("data", []), ("seq", "name")):
            result.append(
                SavedCondition(
                    sequence=str(row.get("seq", "")),
                    name=str(row.get("name", "")),
                    market=selected_market,
                )
            )
        return tuple(result)

    async def run_condition(
        self,
        market: Market | str,
        sequence: str,
        *,
        domestic_exchange: str = "K",
        max_pages: int = 10,
    ) -> tuple[ConditionMatch, ...]:
        selected_market = _as_market(market)
        if not str(sequence).strip():
            raise ValueError("sequence가 필요합니다.")
        if max_pages <= 0:
            raise ValueError("max_pages는 1 이상이어야 합니다.")

        path = _path(selected_market)
        websocket = await self._connect_factory(
            f"{self.config.websocket_base_url}{path}",
            self.config.timeout_seconds,
        )
        try:
            await self._login(websocket)
            trnm = "CNSRREQ" if selected_market is Market.DOMESTIC else "GCNSRREQ"
            body: dict[str, Any] = {
                "trnm": trnm,
                "seq": str(sequence),
                "search_type": "0",
                "cont_yn": "N",
                "next_key": "",
            }
            if selected_market is Market.DOMESTIC:
                body["stex_tp"] = domestic_exchange

            matches: list[ConditionMatch] = []
            for _ in range(max_pages):
                await websocket.send(json.dumps(body, ensure_ascii=False))
                response = await self._receive(websocket)
                _raise_ws_error(response)
                for row in _records(
                    response.get("data", []),
                    ("9001", "302", "10", "25", "11", "12", "13", "16", "17", "18"),
                ):
                    matches.append(_condition_match(selected_market, row))
                if str(response.get("cont_yn", "N")) != "Y":
                    break
                body["cont_yn"] = "Y"
                body["next_key"] = str(response.get("next_key", ""))
            return tuple(matches)
        finally:
            await websocket.close()

    async def _request_once(self, market: Market, body: dict[str, Any]) -> dict[str, Any]:
        websocket = await self._connect_factory(
            f"{self.config.websocket_base_url}{_path(market)}",
            self.config.timeout_seconds,
        )
        try:
            await self._login(websocket)
            await websocket.send(json.dumps(body, ensure_ascii=False))
            response = await self._receive(websocket)
            _raise_ws_error(response)
            return response
        finally:
            await websocket.close()

    async def _login(self, websocket: Any) -> None:
        await websocket.send(
            json.dumps(
                {"trnm": "LOGIN", "token": self._token_provider()},
                ensure_ascii=False,
            )
        )
        response = await self._receive(websocket, include_login=True)
        if str(response.get("trnm", "")).upper() != "LOGIN":
            raise BrokerAPIError("키움 WebSocket LOGIN 응답을 받지 못했습니다.")
        _raise_ws_error(response)

    async def _receive(self, websocket: Any, *, include_login: bool = False) -> dict[str, Any]:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=self.config.timeout_seconds)
            except TimeoutError as exc:
                raise BrokerAPIError("키움 WebSocket 응답 시간이 초과되었습니다.") from exc
            parsed = _parse_message(raw)
            if _is_ping(parsed):
                await websocket.send(_serialize_message(parsed))
                continue
            if not isinstance(parsed, dict):
                raise BrokerAPIError("키움 WebSocket 응답 형식이 올바르지 않습니다.")
            if not include_login and str(parsed.get("trnm", "")).upper() == "LOGIN":
                continue
            return parsed


async def _default_connect(uri: str, timeout_seconds: float) -> Any:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise OptionalDependencyError(
            '조건검색에는 websockets가 필요합니다. pip install -e ".[conditions]"를 실행하세요.'
        ) from exc
    return await connect(uri, open_timeout=timeout_seconds, ping_interval=None)


def _parse_message(message: Any) -> Any:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if isinstance(message, str):
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return message
    return message


def _serialize_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    return json.dumps(message, ensure_ascii=False)


def _raise_ws_error(response: dict[str, Any]) -> None:
    code = response.get("return_code")
    try:
        normalized = int(code) if code not in (None, "") else 0
    except (TypeError, ValueError):
        normalized = code
    if normalized != 0:
        raise BrokerAPIError(
            str(response.get("return_msg") or "키움 WebSocket 요청에 실패했습니다."),
            return_code=code,
        )


def _is_ping(message: Any) -> bool:
    if isinstance(message, str):
        return message.strip().upper() == "PING"
    return isinstance(message, dict) and str(message.get("trnm", "")).upper() == "PING"


def _path(market: Market) -> str:
    return "/api/dostk/websocket" if market is Market.DOMESTIC else "/api/us/websocket"


def _as_market(value: Market | str) -> Market:
    if isinstance(value, Market):
        return value
    try:
        return Market(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError("market은 'domestic' 또는 'us'여야 합니다.") from exc


def _records(value: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return result
    for row in value:
        if isinstance(row, dict):
            result.append(row)
        elif isinstance(row, (list, tuple)):
            result.append(dict(zip(keys, row)))
    return result


def _decimal(value: Any, *, absolute: bool = False) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except InvalidOperation:
        return None
    return abs(number) if absolute else number


def _condition_match(market: Market, row: dict[str, Any]) -> ConditionMatch:
    exchange = str(row.get("stex_tp") or ("KRX" if market is Market.DOMESTIC else ""))
    symbol = str(row.get("9001", "")).strip().upper()
    if market is Market.DOMESTIC and symbol.startswith("A") and symbol[1:].isdigit():
        symbol = symbol[1:]
    return ConditionMatch(
        market=market,
        symbol=symbol,
        name=str(row.get("302", "")),
        exchange=exchange,
        price=_decimal(row.get("10"), absolute=True),
        change_rate=_decimal(row.get("12")),
        volume=_decimal(row.get("13"), absolute=True),
        raw=dict(row),
    )
