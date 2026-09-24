"""Isolated mark1 prototype signals and an unconditional no-order backend."""
from __future__ import annotations

from decimal import Decimal
from threading import Event

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import OrderNotSent
from dockdack.mark1_adapter import Mark1SignalProducer
from dockdack.models import TradingMode

SOURCE_ID = "mark1-prototype-daily-barrier"
STRATEGY_ID = "mark1-prototype"
ORDER_BLOCK_REASON = "mark1 prototype은 관찰·연구 신호 전용입니다. 모의·실전 주문 전송을 허용하지 않습니다."
STRATEGY_NOTICE = (
    "연구 신호: 보정 성공확률 50% 초과 (> 50%, 50%는 대기) · "
    "보유 기준 +1% 익절 / -0.9% 손절 · 모든 주문 전송 차단"
)
RISK_NOTICE = (
    "연구 검증 미통과 · 기업행사/주식분할 가격단위 데이터 문제 확인 · "
    "미국 선택 모델은 과거 검증에서 매수 신호 0건 (성공을 의미하지 않음)"
)


def _blocked(*args, **kwargs):
    raise OrderNotSent(ORDER_BLOCK_REASON)


class _ReadOnlyBroker:
    """Expose only the broker reads needed by the monitoring dashboard."""
    _READS = frozenset({
        "get_quote", "daily_history", "account_domestic", "account_us",
        "list_open_orders", "list_order_executions", "list_execution_history",
        "resolve_us_exchange", "top_turnover", "top_volume", "common_equities",
    })
    _WRITES = frozenset({
        "build_order", "build_order_at_current_price", "place_order", "buy", "sell",
        "buy_at_current_price", "sell_at_current_price", "cancel_order", "modify_order", "amend_order",
    })

    def __init__(self, broker):
        self.__broker = broker

    @property
    def mode(self):
        if self.__broker.mode is not TradingMode.DEMO:
            raise ValueError("prototype 읽기 전용 브로커의 모의 환경이 변경되었습니다.")
        return TradingMode.DEMO

    def __getattr__(self, name):
        if name in self._WRITES:
            return _blocked
        if name in self._READS:
            method = getattr(self.__broker, name)
            def read(*args, **kwargs):
                self.mode
                return method(*args, **kwargs)
            return read
        raise AttributeError(name)


class PrototypeReadOnlyService:
    """Fail-closed facade, including direct service and broker order bypasses.

    This is application-level isolation, not a sandbox against arbitrary Python
    introspection. Existing TradingService instances are not mutated.
    """
    _READS = frozenset({
        "resolve", "quote", "history", "top_turnover", "top_volume", "common_equities", "ensure_common_equity",
        "ensure_demo", "ensure_environment", "account", "orders", "executions", "execution_history",
        "safety_orders", "safety_executions", "safety_account", "protected_symbols",
    })
    _WRITES = frozenset({
        "prepare", "submit", "ensure_order_permission", "acknowledge_live_risk",
        "place_order", "buy", "sell", "cancel_order", "modify_order", "amend_order",
    })

    def __init__(self, service):
        if getattr(service, "mode", None) is not TradingMode.DEMO:
            raise ValueError("mark1 prototype은 모의 환경의 읽기 전용 서비스만 사용합니다.")
        self.__service = service

    @property
    def mode(self):
        if self.__service.mode is not TradingMode.DEMO:
            raise ValueError("prototype 읽기 전용 서비스의 모의 환경이 변경되었습니다.")
        return TradingMode.DEMO

    @property
    def storage_scope(self):
        return getattr(self.__service, "storage_scope", "demo")

    @property
    def live_risk_acknowledged(self):
        return False

    @property
    def brokers(self):
        return {market: _ReadOnlyBroker(broker) for market, broker in getattr(self.__service, "brokers", {}).items()}

    def broker(self, market):
        self.mode
        return _ReadOnlyBroker(self.__service.broker(market))

    def __getattr__(self, name):
        if name in self._WRITES:
            return _blocked
        if name in self._READS:
            method = getattr(self.__service, name)
            def read(*args, **kwargs):
                self.mode
                return method(*args, **kwargs)
            return read
        raise AttributeError(name)


class PrototypeAutoTrader(AutoTrader):
    """Reuse signal ingestion/polling, but no reachable supported order path."""
    def __init__(self, service, store, *, items, clock):
        if not isinstance(service, PrototypeReadOnlyService):
            service = PrototypeReadOnlyService(service)
        super().__init__(service, store, clock=clock)
        self.allowed_items = {item.id for item in items}
        self.initial_attempts = {row["rule_id"] for row in store.attempts()}
        self.safety_reason = "PROTOTYPE_OBSERVATION_ONLY"
        self.external_stop = Event()
        self.universe = None
        self.close_liquidator = None
        self.predictors = {}
        self.enable_holdings_exits = False
        self.equity_buy_percent = None
        self.us_retry_attempts = 1

    def configure_external_sources(self, sources):
        sources = tuple(sources)
        if any(policy.source_id != SOURCE_ID for policy, _ in sources):
            raise ValueError("prototype 전용 신호 출처 외에는 연결할 수 없습니다.")
        return super().configure_external_sources(sources)

    def holding_exit_targets(self, position):
        """Display-only prototype cost boundaries, never the shared -.8% fallback."""
        from dockdack.gui_service import Instrument
        from dockdack.watchlist import instrument_key
        key = instrument_key(Instrument(position.market, position.symbol, position.exchange))
        average = position.average_price
        valid = isinstance(average, Decimal) and average.is_finite() and average > 0
        return {"take_profit_price": average * Decimal("1.01") if valid else None,
                "stop_loss_price": average * Decimal("0.991") if valid else None,
                "source": "평균매입가 +1% / -0.9% · 연구 표시만 / 주문 차단" if valid else "매입가 확인 필요 · 주문 차단",
                "watch_id": key, "research_only": True, "orders_permitted": False}

    @property
    def orders_enabled(self):
        # Even directly setting the parent's Event cannot grant permission.
        return False

    def enable_orders(self, confirmation):
        self.disarm()
        raise ValueError(ORDER_BLOCK_REASON)

    def _ensure_environment(self, instrument=None, *, orders=False):
        if orders:
            self.disarm()
            raise OrderNotSent(ORDER_BLOCK_REASON)
        if self.external_stop.is_set():
            self.disarm()
            raise ValueError("외부 중지 요청으로 mark1 prototype 감시가 중지되었습니다.")
        if self.service.mode is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO:
            self.disarm()
            raise ValueError("mark1 prototype은 모의 환경 조회만 허용합니다.")
        if self.universe is not None:
            self.universe.validate_active()
        elif {item.id for item in self.store.items()} != self.allowed_items:
            raise ValueError("mark1 prototype의 승인된 관심종목이 변경되었습니다.")
        return super()._ensure_environment(instrument, orders=False)

    def _critical_attempts(self):
        return [row for row in self.store.attempts() if row["status"] in {"unknown", "submitting"}]

    def _preflight(self, *args, **kwargs):
        self.disarm()
        raise OrderNotSent(ORDER_BLOCK_REASON)

    def _execute(self, *args, **kwargs):
        self.disarm()
        raise OrderNotSent(ORDER_BLOCK_REASON)

    def _execute_once(self, *args, **kwargs):
        self.disarm()
        raise OrderNotSent(ORDER_BLOCK_REASON)

    def _before_order_send(self, *args, **kwargs):
        self.disarm()
        raise OrderNotSent(ORDER_BLOCK_REASON)


class PrototypeSignalProducer(Mark1SignalProducer):
    """Same validated daily-barrier wire, isolated source and research metadata."""
    source_id = SOURCE_ID

    def __init__(self, *args, **kwargs):
        if kwargs.get("trading_mode", "demo") != "demo":
            raise ValueError("prototype은 모의 환경 연구 신호만 생성합니다.")
        super().__init__(*args, **kwargs)
        if any(row["payload"].get("trading_mode") != "demo" for row in self.state.values()):
            raise ValueError("prototype 전용 모의 결정 상태가 필요합니다.")

    def model_prediction(self, predictor, bars, price):
        prediction, detail = super().model_prediction(predictor, bars, price)
        # This version is intentionally fixed to the saved diagnostic policies.
        # Do not silently reinterpret a future selective policy as p > 0.5.
        if prediction.get("policy_threshold", .5) != .5 or prediction.get("stop_probability_cap", 1.) != 1.:
            raise ValueError("이 prototype은 저장된 > 50% / stop cap 1 정책만 지원합니다.")
        detail.update(strategy_notice=STRATEGY_NOTICE, input_tokens=None, input_features=184,
                      input_description="완료 30봉 + 현재가로 계산한 184개 인과적 표 특성")
        return prediction, detail

    def _decision(self, stock, charts, now):
        decision, detail = super()._decision(stock, charts, now)
        predictor = self.predictors.get(stock.get("market"))
        metadata = getattr(predictor, "metadata", {})
        detail.update(
            strategy=STRATEGY_ID, title="mark1 prototype", research_only=True,
            research_qualified=False, deployment_allowed=False, orders_permitted=False,
            known_data_quality_issues=True, intraday_path_verified=False,
            model_name=metadata.get("model_name", metadata.get("architecture", "모델 없음")),
            risk_notice=RISK_NOTICE, order_block_reason=ORDER_BLOCK_REASON,
        )
        return decision, detail
