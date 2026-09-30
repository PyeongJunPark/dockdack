"""DEMO-only order bridge for daily-trained, actual-five-minute prototypes.

The broker feed supplies completed five-minute bars. This adapter neither
trains on them nor places orders: it publishes data-only BUY/HOLD signals for
the existing, manually armed engine. Every BUY is rechecked from the exact
in-memory minute snapshot and fresh broker quote at preflight/final send.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from dockdack.gui_service import Instrument
from dockdack.minute_feed import DomesticMinuteFeed
from dockdack.minute_model_catalog import MINUTE_RESEARCH_BY_ID, MINUTE_TRANSFER_IDS
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.research.daily_proxy_minute import (
    DAILY_PROXY_CONFIGS, infer_daily_proxy, load_configured_daily_proxy, sha256_file,
)
from dockdack.signal_bridge import atomic_json, prototype_record_family


KST = ZoneInfo("Asia/Seoul")
MODELS_ROOT = Path(__file__).resolve().parents[2] / "models" / "mark1_minute"


class SharedDomesticMinuteFeed:
    """Create one DEMO broker feed lazily on the normal polling worker."""

    def __init__(self, window):
        self.window = window
        self.service, self.store = window.service, window.store
        self.scope = getattr(window.service, "storage_scope", None)
        self._lock = Lock()
        self._feed = None

    def get_complete_bars(self, symbol, exchange, *, now, count):
        self._check_scope()
        with self._lock:
            if self._feed is None:
                broker = self.service.broker(Market.DOMESTIC)
                self._feed = DomesticMinuteFeed(
                    broker, self.store.path.parent / "exchange" / "minute-cache",
                    clock=self.window.engine.clock)
            feed = self._feed
        result = feed.get_complete_bars(symbol, exchange, now=now, count=count)
        self._check_scope()
        return result

    def _check_scope(self):
        if (self.window.service is not self.service or self.window.store is not self.store
                or self.service.mode is not TradingMode.DEMO
                or self.store.mode is not TradingMode.DEMO
                or getattr(self.service, "storage_scope", None) != self.scope):
            raise ValueError("분봉 피드의 모의계정 범위가 바뀌어 재연결해야 합니다.")


class DailyProxyMinuteFeed:
    """One selected model, one source file, shared minute/account readers."""

    def __init__(self, window, model_id, policy, output_path, *, minute_feed=None,
                 account_snapshots=None, models_root=None):
        if model_id not in MINUTE_TRANSFER_IDS:
            raise ValueError("일봉 학습·5분봉 추론 모델 ID가 아닙니다.")
        spec = MINUTE_RESEARCH_BY_ID[model_id]
        if policy.source_id != spec.source_id:
            raise ValueError("분봉 모델과 주문 신호 출처가 다릅니다.")
        self.window, self.model_id, self.policy = window, model_id, policy
        self.service, self.store = window.service, window.store
        self.scope = getattr(window.service, "storage_scope", None)
        self.source_id, self.strategy_id, self.title = spec.source_id, model_id, spec.title
        self.output_path = Path(output_path)
        self.models_root = Path(models_root) if models_root is not None else MODELS_ROOT
        self.minute_feed = minute_feed
        self.account_snapshots = account_snapshots
        self.artifact = None
        self.manifest = None
        self.manifest_sha256 = None
        self.diagnostics = {}
        self.status = f"{self.title} · 완료 일봉 학습 / 실제 5분봉 추론 대기 · 모의 전용"
        self._decisions = {}
        self._ready = False
        self._closed = False

    def _ensure_demo(self):
        if (self._closed or self.window.service is not self.service
                or self.window.store is not self.store
                or self.service.mode is not TradingMode.DEMO
                or self.store.mode is not TradingMode.DEMO
                or getattr(self.service, "storage_scope", None) != self.scope
                or self.window.engine._stop.is_set()):
            self._ready = False
            raise ValueError("분봉 prototype은 모의투자 감시 중에만 사용할 수 있습니다.")

    def _model(self):
        if self.artifact is None:
            artifact, manifest = load_configured_daily_proxy(self.models_root, self.model_id)
            config = DAILY_PROXY_CONFIGS[self.model_id]
            manifest_path = self.models_root / config.bundle_name / "manifest.json"
            self.artifact, self.manifest = artifact, manifest
            self.manifest_sha256 = sha256_file(manifest_path)
        self._check_bundle()
        return self.artifact

    def _check_bundle(self):
        config = DAILY_PROXY_CONFIGS[self.model_id]
        folder = self.models_root / config.bundle_name
        if (self.manifest_sha256 is None
                or sha256_file(folder / "manifest.json") != self.manifest_sha256
                or sha256_file(folder / f"{config.architecture}.pt")
                    != self.manifest["models"][config.architecture]["state_sha256"]):
            raise ValueError("분봉 prototype 학습 번들이 변경되어 주문을 중단합니다.")

    def _feed(self):
        if self.minute_feed is None:
            broker = self.window.service.broker(Market.DOMESTIC)
            self.minute_feed = DomesticMinuteFeed(
                broker, self.window.store.path.parent / "exchange" / "minute-cache",
                clock=self.window.engine.clock)
        return self.minute_feed

    def _position_allows_buy(self, stock):
        """Unknown account/lot state is HOLD, never assumed flat."""
        if self.policy.max_krw <= 0:
            raise ValueError("국내 주문금액 상한이 0이어서 매수를 보류합니다.")
        if self.account_snapshots is None:
            raise ValueError("공유 모의계좌 스냅샷이 없어 매수를 보류합니다.")
        account, _ = self.account_snapshots.get(stock, window=self.window)
        positions = [position for position in account.positions if position.symbol == stock["symbol"]]
        if any(position.exchange != stock["exchange"] for position in positions):
            raise ValueError("보유종목 거래소가 조회 종목과 다릅니다.")
        quantity = sum((position.quantity for position in positions), Decimal(0))
        sellable = sum((position.sellable_quantity for position in positions), Decimal(0))
        inventory = self.window.store.prototype_inventory(
            stock["watch_id"], broker_quantity=quantity, broker_sellable=sellable)
        if not inventory["reconciled"]:
            raise ValueError("모델별 확정 체결과 증권사 보유수량을 대조할 수 없습니다.")
        if self.window.store.prototype_pending_buys(stock["watch_id"], strategy_id=self.strategy_id):
            raise ValueError("같은 모델의 미확정 매수가 있습니다.")
        if any(lot["strategy_id"] == self.strategy_id and lot["quantity_remaining"] > 0
               for lot in inventory["lots"]):
            raise ValueError("같은 모델의 보유분이 있어 추가 매수하지 않습니다.")

    def _signal(self, stock, export_id, now, action):
        watch_id = stock["watch_id"]
        signal_id = (self.model_id + ":" + uuid5(
            NAMESPACE_URL, f"{self.source_id}:{export_id}:{watch_id}").hex)
        row = {key: stock[key] for key in ("market", "symbol", "exchange")}
        row.update(signal_id=signal_id, export_id=export_id, action=action,
                   generated_at=now.isoformat(),
                   expires_at=(now + timedelta(seconds=90)).isoformat())
        if action == "buy":
            row.update(quantity=1, max_notional=str(self.policy.max_krw),
                       strategy_id=self.model_id, model_title=self.title,
                       model_version=DAILY_PROXY_CONFIGS[self.model_id].bundle_name,
                       model_manifest_sha256=self.manifest_sha256)
        return row

    @staticmethod
    def _stock_quote(stock, now):
        if (stock.get("market") != "domestic" or stock.get("exchange") != "KRX"
                or not isinstance(stock.get("symbol"), str)
                or stock.get("watch_id") != f"domestic:KRX:{stock.get('symbol')}"):
            raise ValueError("국내 KRX 종목만 5분봉 신호로 판단합니다.")
        try:
            fetched = datetime.fromisoformat(stock["quote_fetched_at"])
            price = Decimal(str(stock["price"]))
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise ValueError("현재가 조회값을 확인할 수 없습니다.") from exc
        if (fetched.tzinfo is None or not 0 <= (now - fetched).total_seconds() <= 15
                or not price.is_finite() or price <= 0):
            raise ValueError("현재가 시각·가격이 유효하지 않습니다.")
        return price

    def publish(self, chart):
        if self._closed or self.window.engine._stop.is_set():
            return
        now = self.window.engine.clock()
        self._ready = False
        rows = []
        pending = {}
        self.diagnostics = {}
        try:
            self._ensure_demo()
            if (not isinstance(chart, dict) or chart.get("trading_mode") != "demo"
                    or chart.get("source") != "kiwoom_demo"
                    or not isinstance(chart.get("stocks"), list)
                    or not isinstance(chart.get("export_id"), str)):
                raise ValueError("검증된 키움 모의 차트 내보내기만 분봉 모델에 사용할 수 있습니다.")
            artifact = self._model()
            config = DAILY_PROXY_CONFIGS[self.model_id]
            for stock in chart["stocks"]:
                if not isinstance(stock, dict) or stock.get("status") != "ok":
                    continue
                watch_id = stock.get("watch_id")
                if not isinstance(watch_id, str):
                    continue
                action = "hold"
                diagnostic = {"watch_id": watch_id, "reason": "MINUTE_HOLD",
                              "score_unit": "일봉 학습 점수 · 분봉 성능 미검증"}
                try:
                    quote = self._stock_quote(stock, now)
                    if stock["symbol"] not in artifact["training_symbols"]:
                        # This frozen bundle cannot authorize another symbol;
                        # do not spend a chart request on an inevitable HOLD.
                        diagnostic["reason"] = "SYMBOL_OUTSIDE_DAILY_TRAINING"
                        self.diagnostics[watch_id] = diagnostic
                        rows.append(self._signal(stock, chart["export_id"], now, action))
                        continue
                    snapshot = self._feed().get_complete_bars(
                        stock["symbol"], stock["exchange"], now=now,
                        count=config.lookback + 2)
                    result = infer_daily_proxy(
                        artifact, snapshot.bars, architecture=config.architecture,
                        as_of=now)
                    diagnostic.update(score=result["probability_proxy"],
                                      threshold=result["validation_threshold"],
                                      signal_bar_label=result["signal_bar_label"],
                                      minute_bar_label=snapshot.last_bar_label.isoformat(),
                                      minute_receipt_sha256=snapshot.sha256)
                    if result["candidate"]:
                        # The model uses no current quote, but a grossly
                        # divergent price signals stale data/market disruption.
                        close = snapshot.bars[-1].close
                        if abs(quote / close - 1) > Decimal(".05"):
                            raise ValueError("현재가와 최신 완료 5분봉 가격 차이가 5%를 넘습니다.")
                        self._position_allows_buy(stock)
                        action = "buy"
                        diagnostic["reason"] = "MINUTE_BUY_CANDIDATE"
                        pending[watch_id] = {"snapshot": snapshot, "decision": result,
                                             "slot": DomesticMinuteFeed._complete_slot(now.astimezone(KST)),
                                             "manifest_sha256": self.manifest_sha256}
                    elif not result["symbol_in_daily_training_universe"]:
                        diagnostic["reason"] = "SYMBOL_OUTSIDE_DAILY_TRAINING"
                    elif not result["lifecycle_fits_session"]:
                        diagnostic["reason"] = "MINUTE_EXIT_TIME_UNAVAILABLE"
                    else:
                        diagnostic["reason"] = "MINUTE_THRESHOLD_NOT_MET"
                except Exception as exc:
                    diagnostic["reason"] = "MINUTE_INPUT_UNAVAILABLE"
                    diagnostic["error"] = str(exc)[:300]
                self.diagnostics[watch_id] = diagnostic
                row = self._signal(stock, chart["export_id"], now, action)
                if action == "buy":
                    pending[watch_id]["signal_id"] = row["signal_id"]
                rows.append(row)
            self._ensure_demo()
            atomic_json(self.output_path, {"schema_version": 1, "source_id": self.source_id,
                                           "trading_mode": "demo", "signals": rows})
            self._decisions = pending
            self._ready = True
            buys = sum(row["action"] == "buy" for row in rows)
            self.status = (f"{self.title} · 실제 완료 5분봉 {config.lookback}+지연 2봉"
                           f" · 매수 후보 {buys}개 / 조회 {len(rows)}개 · 모의 전용")
        except Exception as exc:
            # Do not leave a previous BUY authorized if an entire publication
            # fails. If even HOLD cannot be written, the callback stays closed.
            self._decisions = {}
            try:
                holds = [self._signal(stock, chart["export_id"], now, "hold")
                         for stock in chart.get("stocks", [])
                         if isinstance(stock, dict) and stock.get("status") == "ok"
                         and isinstance(stock.get("watch_id"), str)] if isinstance(chart, dict) else []
                atomic_json(self.output_path, {"schema_version": 1, "source_id": self.source_id,
                                               "trading_mode": "demo", "signals": holds})
            except Exception:
                self.close()
            self.status = f"{self.title} · 분봉 연결 불가, 주문 보류: {str(exc)[:200]}"

    def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price, *, stage="preflight"):
        """Local-only repeat inference; broker quote and lot checks belong to engine."""
        self._ensure_demo()
        if not self._ready or stage not in {"preflight", "final_send"}:
            raise ValueError("분봉 모델 검증 연결이 없거나 주문 단계가 올바르지 않습니다.")
        if rule.side is not OrderSide.BUY or actual_limit_price is None:
            raise ValueError("분봉 prototype은 현재가 지정가 매수만 생성합니다.")
        record = self.window.store.external_for_rule(rule.id)
        family = prototype_record_family(record, watch_id=item.id, action="buy")
        if family is None or family.id != self.model_id or record["source_id"] != self.source_id:
            raise ValueError("매수 신호와 분봉 모델의 원본 ID가 다릅니다.")
        metadata = json.loads(record["payload"])
        state = self._decisions.get(item.id)
        if (state is None or record["signal_id"] != state.get("signal_id")
                or metadata.get("model_manifest_sha256") != self.manifest_sha256
                or metadata.get("model_version") != DAILY_PROXY_CONFIGS[self.model_id].bundle_name):
            raise ValueError("현재 분봉 판단과 주문 신호의 번들·신호 ID가 다릅니다.")
        self._check_bundle()
        self.window.engine._validate_snapshot(item, fresh_snapshot)
        now = self.window.engine.clock()
        if (not 0 <= (now - fresh_snapshot.fetched_at).total_seconds() <= 15
                or item.instrument.market is not Market.DOMESTIC
                or item.instrument.exchange != "KRX"):
            raise ValueError("주문 직전 국내 시세·시간을 확인할 수 없습니다.")
        snapshot = state["snapshot"]
        if (snapshot.symbol != item.instrument.symbol or snapshot.exchange != item.instrument.exchange
                or snapshot.session_date != now.astimezone(KST).date()
                or state["slot"] != DomesticMinuteFeed._complete_slot(now.astimezone(KST))
                or not 0 <= (now - snapshot.fetched_at_utc).total_seconds() < 300):
            raise ValueError("새 완료 5분봉 구간으로 바뀌었거나 분봉이 오래됐습니다.")
        result = infer_daily_proxy(
            self.artifact, snapshot.bars,
            architecture=DAILY_PROXY_CONFIGS[self.model_id].architecture, as_of=now)
        if not result["candidate"] or result["signal_bar_label"] != state["decision"]["signal_bar_label"]:
            raise ValueError("주문 직전 분봉 신호가 유지되지 않습니다.")
        price = fresh_snapshot.quote.price
        limit = Decimal(str(actual_limit_price))
        if (not limit.is_finite() or limit <= 0 or not price.is_finite() or price <= 0
                or abs(price / snapshot.bars[-1].close - 1) > Decimal(".05")
                or abs(limit / price - 1) > Decimal(".01")):
            raise ValueError("현재가·지정가가 분봉 신호의 가격 범위를 벗어났습니다.")
        self._ensure_demo()
        if (self.window.engine.clock() - fresh_snapshot.fetched_at).total_seconds() > 15:
            raise ValueError("분봉 재검증 중 현재가가 만료됐습니다.")

    def close(self):
        self._closed, self._ready = True, False
        self._decisions.clear()


class MinuteResearchHoldFeed:
    """Visible, strictly non-order adapter for unqualified paired-hedge ideas."""

    def __init__(self, window, model_id, policy, output_path):
        spec = MINUTE_RESEARCH_BY_ID[model_id]
        if spec.kind != "hedge" or policy.source_id != spec.source_id:
            raise ValueError("인버스 연구 출처가 일치하지 않습니다.")
        self.window, self.model_id, self.policy = window, model_id, policy
        self.source_id, self.title = spec.source_id, spec.title
        self.output_path = Path(output_path)
        self.diagnostics = {}
        self.status = f"{self.title} · 인버스 두 다리 연구 · 주문 보류"
        self._ready = True

    def publish(self, chart):
        self.diagnostics = {}
        if (not isinstance(chart, dict) or chart.get("source") != "kiwoom_demo"
                or chart.get("trading_mode") != "demo" or not isinstance(chart.get("stocks"), list)
                or not isinstance(chart.get("export_id"), str)):
            self._ready = False
            raise ValueError("인버스 연구는 모의 차트 식별이 필요합니다.")
        now = self.window.engine.clock()
        rows = []
        for stock in chart["stocks"]:
            if not isinstance(stock, dict) or stock.get("status") != "ok":
                continue
            watch_id = stock.get("watch_id")
            if not isinstance(watch_id, str):
                continue
            self.diagnostics[watch_id] = {"watch_id": watch_id,
                                          "reason": "HEDGE_RESEARCH_ORDER_HOLD",
                                          "error": "시장 인버스 ETF 동시 체결·헤지 비율 검증 전까지 주문 보류"}
            rows.append({"signal_id": self.model_id + ":" + uuid5(
                NAMESPACE_URL, f"{self.source_id}:{chart['export_id']}:{watch_id}").hex,
                         "export_id": chart["export_id"],
                         "market": stock["market"], "symbol": stock["symbol"],
                         "exchange": stock["exchange"], "action": "hold",
                         "generated_at": now.isoformat(),
                         "expires_at": (now + timedelta(seconds=90)).isoformat()})
        atomic_json(self.output_path, {"schema_version": 1, "source_id": self.source_id,
                                       "trading_mode": "demo", "signals": rows})
        self._ready = True

    def validate_execution(self, item, rule, fresh_snapshot, actual_limit_price, *, stage="preflight"):
        raise ValueError("인버스 두 다리 연구 모델은 주문이 허용되지 않습니다.")

    def close(self):
        self._ready = False
