"""Configuration loading for Kiwoom demo and real environments."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from dockdack.exceptions import ConfigurationError
from dockdack.models import Market, TradingMode


def _as_mode(value: TradingMode | str) -> TradingMode:
    if isinstance(value, TradingMode):
        return value
    try:
        return TradingMode(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError("거래 환경은 'demo' 또는 'real'이어야 합니다.") from exc


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"참/거짓 설정값을 해석할 수 없습니다: {value!r}")


def _as_market(value: Market | str) -> Market:
    if isinstance(value, Market):
        return value
    try:
        return Market(str(value).strip().lower())
    except ValueError as exc:
        raise ConfigurationError("시장은 'domestic' 또는 'us'여야 합니다.") from exc


@dataclass(frozen=True, slots=True)
class KiwoomConfig:
    app_key: str
    secret_key: str
    mode: TradingMode = TradingMode.DEMO
    timeout_seconds: float = 15.0
    min_request_interval_seconds: float | None = None
    allow_live_orders: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _as_mode(self.mode))
        if not self.app_key.strip():
            raise ConfigurationError("선택한 환경의 키움 App Key가 필요합니다.")
        if not self.secret_key.strip():
            raise ConfigurationError("선택한 환경의 키움 Secret Key가 필요합니다.")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds는 0보다 커야 합니다.")
        if self.min_request_interval_seconds is not None and self.min_request_interval_seconds < 0:
            raise ConfigurationError("min_request_interval_seconds는 0 이상이어야 합니다.")

    @property
    def base_url(self) -> str:
        if self.mode is TradingMode.REAL:
            return "https://api.kiwoom.com"
        return "https://mockapi.kiwoom.com"

    @property
    def websocket_base_url(self) -> str:
        if self.mode is TradingMode.REAL:
            return "wss://api.kiwoom.com:10000"
        return "wss://mockapi.kiwoom.com:10000"

    @property
    def request_interval_seconds(self) -> float:
        if self.min_request_interval_seconds is not None:
            return self.min_request_interval_seconds
        # 모의투자는 TR별 초당 1회 제한이다. 실전도 조회 제한보다 보수적으로 둔다.
        return 1.05 if self.mode is TradingMode.DEMO else 0.21

    @classmethod
    def from_env(
        cls,
        mode: TradingMode | str | None = None,
        *,
        market: Market | str | None = None,
    ) -> "KiwoomConfig":
        # .env는 Git에서 제외된 로컬 비밀정보 파일이다. 이미 설정된 환경변수는 덮어쓰지 않는다.
        load_dotenv(override=False)
        selected_mode = _as_mode(
            mode
            or os.getenv("DOCKDACK_TRADING_MODE")
            or os.getenv("KIWOOM_MODE")
            or TradingMode.DEMO
        )
        prefix = "REAL" if selected_mode is TradingMode.REAL else "DEMO"
        app_key = ""
        secret_key = ""
        if market is not None:
            selected_market = _as_market(market)
            market_prefix = "DOMESTIC" if selected_market is Market.DOMESTIC else "US"
            app_key = os.getenv(
                f"DOCKDACK_KIWOOM_{prefix}_{market_prefix}_APP_KEY",
                "",
            )
            secret_key = os.getenv(
                f"DOCKDACK_KIWOOM_{prefix}_{market_prefix}_SECRET_KEY",
                "",
            )

        # 국내·미국이 같은 계좌 키를 쓰는 경우와 기존 설정을 위한 공통 키 fallback.
        app_key = app_key or os.getenv(f"DOCKDACK_KIWOOM_{prefix}_APP_KEY", "")
        secret_key = secret_key or os.getenv(f"DOCKDACK_KIWOOM_{prefix}_SECRET_KEY", "")
        try:
            timeout_seconds = float(os.getenv("DOCKDACK_HTTP_TIMEOUT_SECONDS", "15"))
        except ValueError as exc:
            raise ConfigurationError("DOCKDACK_HTTP_TIMEOUT_SECONDS는 숫자여야 합니다.") from exc
        return cls(
            app_key=app_key,
            secret_key=secret_key,
            mode=selected_mode,
            timeout_seconds=timeout_seconds,
            allow_live_orders=_as_bool(os.getenv("DOCKDACK_ALLOW_LIVE_ORDERS")),
        )
