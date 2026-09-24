"""Saved mark1 prototype as a normal, manually armed DEMO signal source.

This bridge runs on the existing desktop worker.  It never starts monitoring,
arms the engine, changes account limits, or sends orders.  The frozen bundle's
research failure flags remain unchanged; DEMO experimentation is a separate
GUI permission and is not model qualification.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import threading
from uuid import NAMESPACE_URL, uuid5

from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.lstm30_adapter import instrument, number
from dockdack.mark1_adapter import Mark1SignalProducer, consecutive_completed_bars, decide_position
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.portfolio import PortfolioCache
from dockdack.runtime_paths import model_bundle
from dockdack.signal_bridge import atomic_json


SOURCE_ID = "mark1-prototype-demo-trigger"
TITLE = "mark1 prototype"
STRATEGY_NOTICE = "매수 확률 > 50% · 평균매수가 +1% 익절 / -0.9% 손절 · 모의 전용"
RISK_NOTICE = (
    "연구 검증 미통과 · 미국 과거 검증 매수 신호 0건 · "
    "주식분할 가격단위 데이터 문제 확인 · 장중 선후관계 미검증"
)


def _price_text(value):
    return format(value.normalize(), "f")


class SharedPrototypeAccounts:
    """Short-lived read-only SIGNAL snapshots shared by this GUI's model feeds.

    This object is not an order preflight cache. Changing service, ledger,
    mode or credential scope invalidates it, even during an in-flight read.
    """

    def __init__(self, window):
        self.window = window
        self.service, self.store = window.service, window.store
        self.scope = getattr(self.service, "storage_scope", "demo")
        self.accounts = {}
        self._lock = threading.Lock()

    def _check(self, window):
        if (window.service is not self.service or window.store is not self.store
                or self.window.service is not self.service or self.window.store is not self.store
                or self.service.mode is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO
                or getattr(self.service, "storage_scope", "demo") != self.scope
                or getattr(self.store, "storage_scope", "demo") != self.scope):
            self.accounts.clear()
            raise ValueError("공유 신호 잔고의 계좌·환경이 변경되어 다시 연결해야 합니다.")

    def get(self, stock, *, window):
        with self._lock:
            self._check(window)
            market = Market(stock["market"])
            now = self.window.engine.clock()
            cached = self.accounts.get(market)
            if cached is None or not 0 <= (now - cached[1]).total_seconds() < 10:
                account = self.service.safety_account(Instrument(market, stock["symbol"], stock["exchange"]))
                self._check(window)
                PortfolioCache._validate(account, market)
                cached = (account, self.window.engine.clock())
                self.accounts[market] = cached
            return cached


class Mark1DemoSignalProducer(Mark1SignalProducer):
    """Distinct from the saved observation-only producer and decision state."""

    source_id = SOURCE_ID
    strategy_id = "mark1-prototype"
    title = TITLE
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    take_profit_pct = 1.
    stop_loss_pct = .9
    take_multiplier = Decimal("1.01")
    stop_multiplier = Decimal("0.991")
    strict_model_identity = False  # Keep the original prototype wire compatible.

    def __call__(self, charts, *, now=None):
        payload, diagnostics = super().__call__(charts, now=now)
        # Deterministic origin marker survives an edited source_id and lets the
        # common engine reject this research strategy in REAL independently.
        for row in payload["signals"]:
            row["signal_id"] = self.strategy_id + ":" + row["signal_id"]
        return payload, diagnostics

    def __init__(self, *args, **kwargs):
        if kwargs.get("trading_mode", "demo") != "demo":
            raise ValueError(f"{self.title} 트리거는 모의 환경만 지원합니다.")
        super().__init__(*args, **kwargs)
        if any(row["payload"].get("trading_mode") != "demo" for row in self.state.values()):
            raise ValueError("mark1 prototype 모의 전용 결정 기록이 필요합니다.")

    @classmethod
    def validate_prediction(cls, prediction):
        if (prediction.get("policy_threshold", .5) != .5
                or prediction.get("stop_probability_cap", 1.) != 1.
                or prediction.get("take_profit_pct", None if cls.strict_model_identity else cls.take_profit_pct) != cls.take_profit_pct
                or prediction.get("stop_loss_pct", None if cls.strict_model_identity else cls.stop_loss_pct) != cls.stop_loss_pct
                or (cls.strict_model_identity and prediction.get("strategy_id") != cls.strategy_id)):
            raise ValueError(f"저장된 {cls.title} 확률/익절/손절/모델 정책과 다릅니다.")

    def model_prediction(self, predictor, bars, price):
        prediction, detail = super().model_prediction(predictor, bars, price)
        self.validate_prediction(prediction)
        detail.update(input_tokens=None, input_features=184, strategy_notice=self.strategy_notice,
                      target_basis=f"daily high >= entry * {self.take_multiplier} AND daily low > entry * {self.stop_multiplier}; intraday ordering unverified",
                      research_only=True, research_qualified=False, deployment_allowed=False,
                      known_data_quality_issues=True, risk_notice=self.risk_notice,
                      execution_scope="demo_manual_gui_permission_only")
        return prediction, detail

    def _decision(self, stock, charts, now):
        decision, detail = super()._decision(stock, charts, now)
        if decision["action"] == "sell":
            # One execution owner: the GUI's holdings exit pass uses the actual
            # broker average cost, even after rank removal or missing history.
            detail["model_position_reason"] = decision["reason"]
            decision = {"action": "hold", "reason": "POSITION_EXIT_MANAGED_BY_GUI"}
        elif decision["action"] == "buy":
            price = number(stock["price"], "current price")
            decision = {**decision,
                        "take_profit_price": _price_text(price * self.take_multiplier),
                        "stop_loss_price": _price_text(price * self.stop_multiplier),
                        "strategy_id": self.strategy_id, "model_title": self.title}
            prediction = detail.get("prediction", {})
            metadata = getattr(self.predictors.get(stock.get("market")), "metadata", {})
            for field, value in (("model_version", prediction.get("version", metadata.get("version"))),
                                 ("model_manifest_sha256", prediction.get("bundle_manifest_sha256", metadata.get("bundle_manifest_sha256")))):
                if isinstance(value, str) and value:
                    decision[field] = value
        return decision, {**detail, "title": self.title, "strategy_id": self.strategy_id, "strategy_notice": self.strategy_notice,
                          "risk_notice": self.risk_notice, "research_only": True,
                          "research_qualified": False, "deployment_allowed": False,
                          "intraday_path_verified": False, "known_data_quality_issues": True}


class Mark1TriggerBridge:
    """``publish(chart)`` plugs into the normal desktop's existing worker hook."""

    source_id = SOURCE_ID
    title = TITLE
    strategy_id = "mark1-prototype"
    strategy_notice = STRATEGY_NOTICE
    risk_notice = RISK_NOTICE
    producer_type = Mark1DemoSignalProducer
    bundle_directory = "models/mark1_prototype"
    state_filename = "exchange/mark1-trigger-decisions-v1.json"

    def __init__(self, window, *, predictors=None, bundle_root=None, account_snapshots=None):
        self.window = window
        self.bundle_root = Path(bundle_root) if bundle_root is not None else model_bundle(Path(self.bundle_directory).name)
        self.predictors = dict(predictors or {})
        if set(self.predictors) - {"domestic", "us"}:
            raise ValueError("지원하지 않는 mark1 시장 모델입니다.")
        self._injected = predictors is not None
        self._market_errors = {}
        self._load_error = ""
        self.producer = None
        self.accounts = {}
        self.account_snapshots = account_snapshots
        self.diagnostics = {}
        self.status = f"{self.title} · 첫 장중 조회 시 해당 시장 모델 확인\n{self.strategy_notice}\n{self.risk_notice}"

    def _ensure_demo(self):
        if (getattr(self.window.service, "mode", None) is not TradingMode.DEMO
                or getattr(self.window.store, "mode", None) is not TradingMode.DEMO):
            self.accounts.clear()
            raise ValueError(f"{self.title} 트리거는 실전에서 사용할 수 없습니다.")

    def _position(self, stock):
        self._ensure_demo()
        market = Market(stock["market"])
        now = self.window.engine.clock()
        cached = (self.account_snapshots.get(stock, window=self.window)
                  if self.account_snapshots is not None else self.accounts.get(market))
        if self.account_snapshots is None and (cached is None or not 0 <= (now - cached[1]).total_seconds() < 10):
            account = self.window.service.safety_account(Instrument(market, stock["symbol"], stock["exchange"]))
            self._ensure_demo()
            PortfolioCache._validate(account, market)
            cached = (account, self.window.engine.clock())
            self.accounts[market] = cached
        account, fetched = cached
        positions = [p for p in account.positions if p.symbol == stock["symbol"]]
        if any(p.exchange != stock["exchange"] for p in positions):
            raise ValueError("보유종목 거래소를 확인할 수 없어 매수를 차단합니다.")
        quantity = sum((p.quantity for p in positions), Decimal(0))
        sellable = sum((p.sellable_quantity for p in positions), Decimal(0))
        cost = sum((p.quantity * p.average_price for p in positions), Decimal(0))
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None,
                "fetched_at": fetched.isoformat()}

    def _load_market(self, market):
        if market in self.predictors or market in self._market_errors:
            return
        try:
            if self._injected:
                raise ValueError(f"{market} 모델이 연결되지 않았습니다.")
            # Native CatBoost import/load stays on the existing worker and is
            # performed only for a requested market, once per bridge session.
            self.predictors[market] = self._new_predictor(market)
        except Exception as exc:
            self._market_errors[market] = str(exc)[:500] or type(exc).__name__
        self._load_error = " / ".join(f"{key}: {value}" for key, value in self._market_errors.items())

    def _new_predictor(self, market):
        from dockdack.mark1_prototype_inference import PrototypePredictor
        return PrototypePredictor(self.bundle_root, market)

    def _publish_unavailable(self, chart, reason):
        now = self.window.engine.clock()
        rows = []
        export_id = chart.get("export_id") if isinstance(chart, dict) else None
        stocks = chart.get("stocks", []) if isinstance(chart, dict) else []
        if isinstance(export_id, str) and isinstance(stocks, list):
            from dockdack.lstm30_adapter import IDENTIFIER
            if IDENTIFIER.fullmatch(export_id):
                for stock in stocks[:500]:
                    try:
                        key = instrument(stock)
                        if stock.get("status") != "ok":
                            continue
                    except (KeyError, TypeError, ValueError, AttributeError):
                        continue
                    rows.append({**{key: stock[key] for key in ("market", "symbol", "exchange")},
                                 "signal_id": self.strategy_id + ":" + uuid5(NAMESPACE_URL, f"{self.source_id}:{export_id}:{key}").hex,
                                 "export_id": export_id, "action": "hold", "generated_at": now.isoformat(),
                                 "expires_at": (now + timedelta(seconds=120)).isoformat()})
                    self.diagnostics[key] = {"watch_id": key, "reason": "MARK1_UNAVAILABLE",
                                             "error": str(reason), "emitted": True}
        atomic_json(self.window.engine.external_reader.path,
                    {"schema_version": 1, "source_id": self.source_id, "trading_mode": "demo", "signals": rows})
        self.status = f"{self.title} 실행 불가 · 이 신호는 HOLD\n{str(reason)[:500]}\n{self.risk_notice}"

    def publish(self, chart):
        if self.window.engine._stop.is_set():
            return
        try:
            self._ensure_demo()
            if self.window.engine.external_policy.source_id != self.source_id:
                raise ValueError(f"{self.title} 전용 신호 출처가 연결되지 않았습니다.")
            if (not isinstance(chart, dict) or chart.get("trading_mode") != "demo"
                    or chart.get("source") != "kiwoom_demo" or not isinstance(chart.get("stocks"), list)):
                raise ValueError("모의 키움 일봉 내보내기만 사용할 수 있습니다.")
            for market in {row.get("market") for row in chart["stocks"] if isinstance(row, dict)}:
                if market in {"domestic", "us"}:
                    self._load_market(market)
            if self.producer is None:
                policy = self.window.engine.external_policy
                self.producer = self.producer_type(
                    self.predictors, position_provider=self._position, quantity=1,
                    max_krw=policy.max_krw, max_usd=policy.max_usd, clock=self.window.engine.clock,
                    state_path=self.window.store.path.parent / self.state_filename)
            self.producer.predictors = dict(self.predictors)
            payload, diagnostics = self.producer(chart)
            self._ensure_demo()
        except Exception as exc:
            self._publish_unavailable(chart, exc)
            return
        for row in diagnostics:
            self.diagnostics[row["watch_id"]] = row
        while len(self.diagnostics) > 500:
            self.diagnostics.pop(next(iter(self.diagnostics)))
        atomic_json(self.window.engine.external_reader.path, payload)
        self.status = f"{self.title} 연결됨 · 완료 30봉 + 현재가마다 재추론\n{self.strategy_notice}\n{self.risk_notice}"
        if self._load_error:
            self.status += "\n모델 없음: 해당 시장 HOLD · " + self._load_error

    def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price, *, stage="preflight"):
        """Local-only recheck for the normal engine's preflight/final-send hook.

        It grants no permission and performs no broker queries or lazy loads.
        An accepted old file cannot replace inference at the fresh quote and
        actual rounded limit price. The generic engine owns all other guards.
        """
        self._ensure_demo()
        if stage not in {"preflight", "final_send"}:
            raise ValueError("알 수 없는 mark1 주문 검증 단계입니다.")
        if rule.side is not OrderSide.BUY:
            raise ValueError("mark1 외부 트리거는 매수만 생성합니다. 보유 매도는 독립 감시가 담당합니다.")
        engine = self.window.engine
        # Real engine execution always has a persisted external signal. Bind
        # its family to this callback so a wrongly registered old/new bridge
        # cannot validate the other model's order. A standalone model check
        # without a stored rule grants no execution authority.
        record = self.window.store.external_for_rule(rule.id)
        if record is not None:
            from dockdack.signal_bridge import prototype_record_family
            family = prototype_record_family(record, watch_id=item.id, action="buy")
            if (record["source_id"] != self.source_id or family is None
                    or family.id != self.strategy_id):
                raise ValueError("매수 신호 모델과 주문 재검증 모델이 일치하지 않습니다.")
        engine._validate_snapshot(item, fresh_snapshot)
        now = engine.clock()
        if not 0 <= (now - fresh_snapshot.fetched_at).total_seconds() <= 15:
            raise ValueError("mark1 재검증 시세가 오래되었거나 미래 시각입니다.")
        market = item.instrument.market.value
        predictor = self.predictors.get(market)
        if predictor is None or getattr(predictor, "metadata", {}).get("market") != market:
            raise ValueError("mark1 주문을 재검증할 시장 모델이 없습니다.")
        today = market_time(item.instrument.market, now).date()
        history = fresh_snapshot.history.bars[-item.days:]
        stock = {"market": market, "complete": len(history) >= 30, "available_days": len(history),
                 "bars": [{"date": bar.day.isoformat(), "is_current_day": bar.day == today,
                           **{key: str(getattr(bar, key)) for key in ("open", "high", "low", "close", "volume")}}
                          for bar in history]}
        bars = consecutive_completed_bars(stock, now)
        if actual_limit_price is None:
            raise ValueError("mark1 prototype 매수는 재검증한 지정가 주문만 지원합니다.")
        prices = [number(fresh_snapshot.quote.price, "current price")]
        limit_price = number(actual_limit_price, "limit price")
        if limit_price != prices[0]:
            prices.append(limit_price)
        for price in prices:
            prediction = predictor.predict(bars, current_price=price)
            self.producer_type.validate_prediction(prediction)
            decision = decide_position(current_price=price, quantity=0, sellable_quantity=0, prediction=prediction)
            if decision["action"] != "buy":
                raise ValueError("mark1 현재가/실제 지정가의 성공확률이 50%를 초과하지 않습니다.")
        self._ensure_demo()
        if not 0 <= (engine.clock() - fresh_snapshot.fetched_at).total_seconds() <= 15:
            raise ValueError("mark1 추론 중 시세가 만료되었습니다.")
