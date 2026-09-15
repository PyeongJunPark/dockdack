"""Small, testable HTTP layer for the Kiwoom REST API."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Protocol
from weakref import WeakValueDictionary

from dockdack.config import KiwoomConfig
from dockdack.exceptions import BrokerAPIError
from dockdack.models import TradingMode


# POST is also used for reads. Never infer retry safety from an API prefix or
# endpoint alone: an unknown/new API must remain at-most-once until reviewed.
_READ_ONLY_APIS = frozenset({
    ("ka10001", "/api/dostk/stkinfo"),
    ("ka10099", "/api/dostk/stkinfo"),
    ("ka10081", "/api/dostk/chart"),
    ("ka10032", "/api/dostk/rkinfo"),
    ("kt00018", "/api/dostk/acnt"),
    ("kt00001", "/api/dostk/acnt"),
    ("ka10075", "/api/dostk/acnt"),
    ("ka10076", "/api/dostk/acnt"),
    ("kt00007", "/api/dostk/acnt"),
    ("usa20100", "/api/us/mrkcond"),
    ("usa10098", "/api/us/stkinfo"),
    ("usa10099", "/api/us/stkinfo"),
    ("usa10104", "/api/us/stkinfo"),
    ("usa06012", "/api/us/chart"),
    ("usa20540", "/api/us/rkinfo"),
    ("ust21070", "/api/us/acnt"),
    ("ust21110", "/api/us/acnt"),
    ("ust21050", "/api/us/acnt"),
    ("ust21510", "/api/us/acnt"),
    ("ust21150", "/api/us/acnt"),
})
_MAX_READ_RATE_LIMIT_RETRIES = 2
# Verified 2026-09-15: https://openapi.kiwoom.com/intro?dummyVal=0
# Demo: 1 request/TR/second. usa10099 additionally: 5 requests/minute.
# These finish-to-start gaps are deliberately more conservative than the limits.
_DEMO_INTERVAL = 1.25
_REAL_INTERVAL = 0.4  # Also below the US peak-hour read limit of 3/second.
_US_CATALOG_INTERVAL = 12.2
_ADAPTIVE_SECONDS = 60.0
_ORDER_SEND_GUARD: ContextVar[Callable[[], None] | None] = ContextVar("order_send_guard", default=None)


@contextmanager
def order_send_guard(callback: Callable[[], None]) -> Iterator[None]:
    """Revalidate a scoped order after pacing, immediately before its only send.

    The callback may inspect local state/files but must not perform network
    requests. It raises to deny transmission; its error passes through unchanged
    (not an ambiguous transport failure). ContextVar keeps independent worker
    threads/tasks and nested calls isolated.
    """
    inherited = _ORDER_SEND_GUARD.get()

    def combined():
        if inherited is not None:
            inherited()
        callback()

    token = _ORDER_SEND_GUARD.set(combined)
    try:
        yield
    finally:
        _ORDER_SEND_GUARD.reset(token)


@dataclass
class _RequestGate:
    lock: Any = field(default_factory=RLock)
    state_lock: Any = field(default_factory=RLock)
    last_request_at: float | None = None
    last_completed_at: float | None = None
    catalog_completed_at: float | None = None
    last_interval: float = 0.0
    waiting_until: float | None = None
    in_flight: bool = False
    active_api_id: str | None = None
    request_count: int = 0
    rate_limit_count: int = 0
    retry_count: int = 0
    last_rate_limit_at: float | None = None
    last_rate_limit_api_id: str | None = None
    penalty_level: int = 0
    penalty_until: float = 0.0
    cooldown_until: float = 0.0


# Coordinate concurrent GUI/broker instances in this process without retaining
# credentials or dead clients. Separate test clocks must never share timestamps.
_REQUEST_GATES: WeakValueDictionary[tuple[str, bytes, int], _RequestGate] = WeakValueDictionary()
_REQUEST_GATES_LOCK = RLock()


def _request_gate(config: KiwoomConfig, monotonic: Callable[[], float]) -> _RequestGate:
    key = (config.base_url, sha256(config.app_key.encode("utf-8")).digest(), id(monotonic))
    with _REQUEST_GATES_LOCK:
        gate = _REQUEST_GATES.get(key)
        if gate is None:
            gate = _RequestGate()
            _REQUEST_GATES[key] = gate
        return gate


class ResponseLike(Protocol):
    status_code: int
    text: str
    headers: Mapping[str, str]

    def json(self) -> Any: ...


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> ResponseLike: ...


class RequestsTransport:
    """Default transport backed by a persistent requests session."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: float,
    ) -> ResponseLike:
        return self._session.request(
            method=method,
            url=url,
            headers=dict(headers),
            json=dict(json),
            timeout=timeout,
            allow_redirects=False,
        )


@dataclass(frozen=True, slots=True)
class APIPage:
    body: dict[str, Any]
    has_next: bool
    next_key: str | None
    cont_yn: str | None


@dataclass(slots=True)
class _AccessToken:
    value: str
    expires_at: datetime

    def is_valid(self) -> bool:
        return datetime.now() + timedelta(seconds=60) < self.expires_at


class KiwoomHTTPClient:
    """Handles OAuth, throttling, error normalization, and pagination."""

    def __init__(
        self,
        config: KiwoomConfig,
        *,
        transport: HttpTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or RequestsTransport()
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._request_gate = _request_gate(config, monotonic)
        self._token_lock = RLock()
        self._token: _AccessToken | None = None

    @property
    def _base_interval(self) -> float:
        # Explicit interval overrides remain useful for offline fake transports.
        # The production GUI does not set an override.
        if self.config.min_request_interval_seconds is not None:
            return self.config.request_interval_seconds
        floor = _DEMO_INTERVAL if self.config.mode is TradingMode.DEMO else _REAL_INTERVAL
        return max(floor, self.config.request_interval_seconds)

    @property
    def _catalog_interval(self) -> float:
        return _US_CATALOG_INTERVAL if self.config.min_request_interval_seconds is None else 0.0

    def _effective_interval(self, now: float, *, read_only: bool = True) -> float:
        gate = self._request_gate
        base = self._base_interval
        if read_only and now < gate.penalty_until:
            return max(base, min(5.0, max(base, _DEMO_INTERVAL) * 1.5 ** gate.penalty_level))
        return base

    def rate_status(self) -> dict[str, Any]:
        """Cheap in-memory snapshot; never takes the network/sleep lock or calls API.

        Timestamps use the injected monotonic clock, not Unix time. Clients in
        the same process/credential scope share counters and ``scope_id``.
        """
        gate = self._request_gate
        with gate.state_lock:
            now = self._monotonic()
            interval = self._effective_interval(now)
            next_allowed = max(now, gate.cooldown_until, gate.waiting_until or 0.0)
            if gate.last_completed_at is not None:
                next_allowed = max(next_allowed, gate.last_completed_at + max(interval, gate.last_interval))
            catalog_next = (gate.catalog_completed_at + self._catalog_interval
                            if gate.catalog_completed_at is not None else now)
            return {
                "scope_id": id(gate),
                "configured_interval_seconds": self.config.request_interval_seconds,
                "effective_interval_seconds": interval,
                "catalog_interval_seconds": self._catalog_interval,
                "next_allowed_monotonic": next_allowed,
                "next_allowed_in_seconds": max(0.0, next_allowed - now),
                "catalog_next_in_seconds": max(0.0, catalog_next - now),
                "waiting_until_monotonic": gate.waiting_until,
                "wait_remaining_seconds": max(0.0, (gate.waiting_until or now) - now),
                "cooldown_remaining_seconds": max(0.0, gate.cooldown_until - now),
                "adaptive_remaining_seconds": max(0.0, gate.penalty_until - now),
                "in_flight": gate.in_flight,
                "active_api_id": gate.active_api_id,
                "request_count": gate.request_count,
                "rate_limit_count": gate.rate_limit_count,
                "retry_count": gate.retry_count,
                "last_rate_limit_api_id": gate.last_rate_limit_api_id,
                "last_rate_limit_monotonic": gate.last_rate_limit_at,
            }

    def get_access_token(self) -> str:
        with self._token_lock:
            if self._token is None or not self._token.is_valid():
                self._token = self._issue_token()
            return self._token.value

    def request(
        self,
        *,
        api_id: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        cont_yn: str | None = None,
        next_key: str | None = None,
        retry_auth: bool = True,
    ) -> APIPage:
        if not path.startswith("/"):
            raise ValueError("path는 '/'로 시작해야 합니다.")
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.get_access_token()}",
            "api-id": api_id,
            "User-Agent": "DockDack/0.1",
        }
        if cont_yn is not None:
            headers["cont-yn"] = cont_yn
        if next_key is not None:
            headers["next-key"] = next_key

        read_only = (api_id, path) in _READ_ONLY_APIS
        request_body = dict(body or {})
        rate_limit_retries = 0
        auth_retry_available = retry_auth and read_only
        data: dict[str, Any] | None = None

        def inspect_response(response: ResponseLike) -> None:
            nonlocal data
            if response.status_code == 401 and auth_retry_available:
                return
            data = self._decode_json(response)
            # Observe the rejection before the shared send lock is released, so
            # another API/client cannot slip through before the cooldown starts.
            if _is_rate_limit(response.status_code, data):
                self._record_rate_limit(api_id)

        while True:
            # Transport failures are ambiguous, even for reads: only a decoded,
            # explicit Kiwoom rate-limit rejection below is retried.
            response = self._send(
                "POST",
                f"{self.config.base_url}{path}",
                headers=headers,
                body=request_body,
                inspect_response=inspect_response,
            )
            if response.status_code == 401 and auth_retry_available:
                auth_retry_available = False
                with self._token_lock:
                    self._token = None
                    headers["authorization"] = f"Bearer {self.get_access_token()}"
                continue

            assert data is not None
            if (read_only and _is_rate_limit(response.status_code, data)
                    and rate_limit_retries < _MAX_READ_RATE_LIMIT_RETRIES):
                rate_limit_retries += 1
                with self._request_gate.state_lock:
                    self._request_gate.retry_count += 1
                continue
            self._raise_for_error(response.status_code, data)
            break
        response_cont_yn = _header(response.headers, "cont-yn")
        response_next_key = _header(response.headers, "next-key")
        return APIPage(
            body=data,
            has_next=response_cont_yn == "Y",
            next_key=response_next_key,
            cont_yn=response_cont_yn,
        )

    def iter_pages(
        self,
        *,
        api_id: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        max_pages: int = 10,
    ) -> Iterator[APIPage]:
        if max_pages <= 0:
            raise ValueError("max_pages는 1 이상이어야 합니다.")
        cont_yn: str | None = None
        next_key: str | None = None
        for _ in range(max_pages):
            page = self.request(
                api_id=api_id,
                path=path,
                body=body,
                cont_yn=cont_yn,
                next_key=next_key,
            )
            yield page
            if not page.has_next:
                return
            if not page.next_key or page.next_key == next_key:
                raise BrokerAPIError("연속조회 키가 없거나 반복됩니다. 조회 결과를 완전한 목록으로 사용할 수 없습니다.")
            cont_yn = page.cont_yn or "Y"
            next_key = page.next_key
        raise BrokerAPIError("최대 조회 페이지를 초과했습니다. 일부 결과만으로 주문을 판단하지 마세요.")

    def _issue_token(self) -> _AccessToken:
        response = self._send(
            "POST",
            f"{self.config.base_url}/oauth2/token",
            headers={
                "Content-Type": "application/json;charset=UTF-8",
                "User-Agent": "DockDack/0.1",
            },
            body={
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "secretkey": self.config.secret_key,
            },
        )
        data = self._decode_json(response)
        self._raise_for_error(response.status_code, data)
        token = str(data.get("token", "")).strip()
        if not token:
            raise BrokerAPIError("키움 토큰 응답에 token 값이 없습니다.")
        expires_at = _parse_expiry(data.get("expires_dt"))
        return _AccessToken(token, expires_at)

    def _send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        inspect_response: Callable[[ResponseLike], None] | None = None,
    ) -> ResponseLike:
        # Keep reservation and send together. Otherwise a descheduled thread can
        # send its reserved request alongside a later client's request.
        gate = self._request_gate
        api_id = str(headers.get("api-id", "oauth2/token"))
        path = url.removeprefix(self.config.base_url)
        read_only = (api_id, path) in _READ_ONLY_APIS
        with gate.lock:
            try:
                interval = self._throttle(api_id, read_only=read_only)
                guard = _ORDER_SEND_GUARD.get()
                if guard is not None and not read_only and path != "/oauth2/token":
                    # An OFF/expired decision here means no order was sent, not
                    # an unknown outcome. Do not catch it as a transport error.
                    guard()
                with gate.state_lock:
                    gate.last_request_at = self._monotonic()
                    gate.last_interval = interval
                    gate.in_flight = True
                    gate.request_count += 1
                try:
                    response = self.transport.request(
                        method,
                        url,
                        headers=headers,
                        json=body,
                        timeout=self.config.timeout_seconds,
                    )
                    if inspect_response is not None:
                        inspect_response(response)
                    return response
                except BrokerAPIError:
                    raise
                except Exception as exc:
                    raise BrokerAPIError(f"키움 API 연결에 실패했습니다: {exc}") from exc
            finally:
                with gate.state_lock:
                    if gate.in_flight:
                        gate.last_completed_at = self._monotonic()
                        if api_id == "usa10099":
                            gate.catalog_completed_at = gate.last_completed_at
                    gate.in_flight = False
                    gate.waiting_until = None
                    gate.active_api_id = None

    def _throttle(self, api_id: str, *, read_only: bool) -> float:
        """Caller holds the shared request gate through the following send."""
        gate = self._request_gate
        while True:
            with gate.state_lock:
                now = self._monotonic()
                interval = self._effective_interval(now, read_only=read_only)
                due = now
                if gate.last_completed_at is not None:
                    due = max(due, gate.last_completed_at + max(interval, gate.last_interval))
                if read_only:
                    due = max(due, gate.cooldown_until)
                if api_id == "usa10099" and gate.catalog_completed_at is not None:
                    due = max(due, gate.catalog_completed_at + self._catalog_interval)
                wait = due - now
                gate.active_api_id = api_id
                gate.waiting_until = due if wait > 1e-9 else None
                if wait <= 1e-9:
                    return interval
            # Recalculate after sleeping: a late wakeup must not accrue credits
            # or send several overdue requests in a catch-up burst.
            self._sleeper(wait)

    def _record_rate_limit(self, api_id: str) -> None:
        gate = self._request_gate
        with gate.state_lock:
            now = self._monotonic()
            if now >= gate.penalty_until:
                gate.penalty_level = 0
            gate.penalty_level = min(4, gate.penalty_level + 1)
            gate.penalty_until = now + _ADAPTIVE_SECONDS
            cooldown = min(20.0, 2.5 * 2 ** (gate.penalty_level - 1))
            gate.cooldown_until = max(gate.cooldown_until, now + cooldown)
            gate.rate_limit_count += 1
            gate.last_rate_limit_at = now
            gate.last_rate_limit_api_id = api_id

    @staticmethod
    def _decode_json(response: ResponseLike) -> dict[str, Any]:
        try:
            data = response.json()
        except Exception as exc:
            raise BrokerAPIError(
                "키움 API가 JSON이 아닌 응답을 반환했습니다.",
                status_code=response.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise BrokerAPIError(
                "키움 API 응답의 최상위 값이 객체가 아닙니다.",
                status_code=response.status_code,
            )
        return data

    @staticmethod
    def _raise_for_error(status_code: int, data: Mapping[str, Any]) -> None:
        return_code = data.get("return_code")
        normalized = _normalize_return_code(return_code)
        if 200 <= status_code < 300 and normalized in (None, 0):
            return
        message = str(data.get("return_msg") or f"HTTP {status_code} 요청 실패")
        raise BrokerAPIError(
            message,
            return_code=return_code,
            status_code=status_code,
        )


def _parse_expiry(value: Any) -> datetime:
    if value:
        try:
            return datetime.strptime(str(value), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    # 형식이 바뀌어도 토큰을 무기한 캐시하지 않는다.
    return datetime.now() + timedelta(hours=23)


def _normalize_return_code(value: Any) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _is_rate_limit(status_code: int, data: Mapping[str, Any]) -> bool:
    # Kiwoom can wrap its specific 1700 code inside a broader return_code.
    # Do not treat an arbitrary message mentioning "1700" (or HTTP 429 alone)
    # as evidence, and never retry auth/redirect/server failures here.
    if not (200 <= status_code < 300 or status_code == 429):
        return False
    raw_code = data.get("return_code")
    # JSON booleans/floats are not valid broker codes. Do not coerce a malformed
    # fractional code such as 1700.5 into a confirmed 1700 rejection.
    if type(raw_code) not in (int, str):
        return False
    code = _normalize_return_code(raw_code)
    if code == 1700:
        return True
    return (code not in (None, 0)
            and "[1700:허용된 API 요청 개수를 초과하였습니다" in str(data.get("return_msg", "")))


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return str(value)
    return None
