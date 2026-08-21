"""Small, testable HTTP layer for the Kiwoom REST API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Mapping, Protocol

from dockdack.config import KiwoomConfig
from dockdack.exceptions import BrokerAPIError


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
        self._last_request_at: float | None = None
        self._token: _AccessToken | None = None

    def get_access_token(self) -> str:
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

        response = self._send(
            "POST",
            f"{self.config.base_url}{path}",
            headers=headers,
            body=body or {},
        )
        if response.status_code == 401 and retry_auth:
            self._token = None
            return self.request(
                api_id=api_id,
                path=path,
                body=body,
                cont_yn=cont_yn,
                next_key=next_key,
                retry_auth=False,
            )

        data = self._decode_json(response)
        self._raise_for_error(response.status_code, data)
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
            cont_yn = page.cont_yn or "Y"
            next_key = page.next_key or ""

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
    ) -> ResponseLike:
        self._throttle()
        try:
            return self.transport.request(
                method,
                url,
                headers=headers,
                json=body,
                timeout=self.config.timeout_seconds,
            )
        except BrokerAPIError:
            raise
        except Exception as exc:
            raise BrokerAPIError(f"키움 API 연결에 실패했습니다: {exc}") from exc

    def _throttle(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            wait = self.config.request_interval_seconds - (now - self._last_request_at)
            if wait > 0:
                self._sleeper(wait)
                now = self._monotonic()
        self._last_request_at = now

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
        if status_code < 400 and normalized in (None, 0):
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


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return str(value)
    return None
