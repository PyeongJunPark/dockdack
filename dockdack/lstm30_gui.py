"""The existing order dashboard, bound exclusively to trained LSTM30 DEMO signals.

This is a GUI owner of the same dedicated ledger as the headless runtime, not a
second trading engine. The shared OS lock requires the headless owner to exit
before this window opens. Ordinary launch leaves automatic orders OFF.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import signal
import sys
from threading import Event, Thread

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from dockdack.autotrade import AutoTrader
from dockdack.demo_session import SessionController
from dockdack.exceptions import OrderNotSent
from dockdack.gui_service import Instrument, TradingService
from dockdack.history import market_time
from dockdack.lstm30_adapter import LSTM30SignalProducer, SOURCE_ID, demo_position_provider
from dockdack.lstm30_close import CloseLiquidator
from dockdack.lstm30_runtime import DEFAULT_ITEMS, DEFAULT_RUNTIME_DIR, SessionLock, _BAD_DIAGNOSTICS
from dockdack.lstm30_universe import LSTM30Universe, ScopedRankingScheduler
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.signal_bridge import ExternalPolicy, atomic_json
from dockdack.watch_gui import WatchlistDialog
from dockdack.watchlist import WatchItem, WatchStore, utc_now


class _LSTM30AutoTrader(AutoTrader):
    """Preserve main's checks while making quote/rejection failures sticky OFF."""

    def __init__(self, service, store, *, items, clock):
        super().__init__(service, store, clock=clock)
        self.allowed_items = {item.id for item in items}
        self.initial_attempts = {row["rule_id"] for row in store.attempts()}
        self.safety_reason = ""
        self.external_stop = Event()
        self.universe = None
        self.close_liquidator = None

    def _ensure_environment(self, instrument=None, *, orders=False):
        if self.external_stop.is_set():
            self.disarm()
            raise ValueError("외부 중지 요청을 처리했습니다. 다시 실행하려면 LSTM30 창을 새로 여세요.")
        if self.service.mode is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO:
            self.disarm()
            raise ValueError("LSTM30 GUI는 모의투자 전용입니다. 실전 전환은 허용하지 않습니다.")
        if self.universe is not None:
            try:
                self.universe.validate_active()
            except Exception:
                self.disarm()
                raise
        elif {item.id for item in self.store.items()} != self.allowed_items:
            self.disarm()
            raise ValueError("승인한 LSTM30 관심종목이 변경되었습니다. 자동주문을 차단합니다.")
        super()._ensure_environment(instrument, orders=orders)

    def _critical_attempts(self):
        return [row for row in self.store.attempts()
                if row["rule_id"] not in self.initial_attempts
                and row["status"] in {"rejected", "unknown", "submitting"}]

    def enable_orders(self, confirmation):
        if confirmation != "DEMO_AUTOTRADE":
            self.disarm()
            raise ValueError("DEMO_AUTOTRADE 확인만 허용합니다.")
        if self._critical_attempts():
            self.disarm()
            raise ValueError("이 세션의 거절/미확정 주문을 확인한 후 실행기를 다시 시작하세요.")
        if self.universe is not None and not self.universe.initialized:
            self.disarm()
            raise ValueError("승인된 시장의 보통주 TOP100 확정·전체 조회가 필요합니다.")
        super().enable_orders(confirmation)
        self.safety_reason = ""

    def poll(self, progress=None, checkpoint=None, on_snapshot=None):
        def guarded_progress(update):
            if isinstance(update[1], Exception):
                self.safety_reason = "QUOTE_OR_SNAPSHOT_FAILED"
                self.disarm()
            if self._critical_attempts():
                self.safety_reason = "ORDER_REJECTED_OR_OUTCOME_UNKNOWN"
                self.disarm()
            if progress is not None:
                progress(update)
        def guarded_checkpoint():
            # The same broker worker owns both normal and closing orders.
            # Never wait for an entire 100-stock sweep before checking the close.
            if self.close_liquidator is not None:
                try:
                    self.close_liquidator.tick()
                except Exception:
                    self.safety_reason = "CLOSE_LIQUIDATION_CHECK_FAILED"
                    self.disarm()
                    raise
            return checkpoint() if checkpoint is not None else False
        return super().poll(progress=guarded_progress, checkpoint=guarded_checkpoint, on_snapshot=on_snapshot)

    def _execute(self, item, rule, snapshot):
        if (rule.side is OrderSide.BUY and self.close_liquidator is not None
                and self.close_liquidator.buy_blocked(item.instrument.market)):
            self.store.pause_rule(rule.id)
            self.store.event(item.id, "마감 청산 구간 · 신규 매수 차단", category="signal")
            return False
        return super()._execute(item, rule, snapshot)

    def _before_order_send(self, item, rule, fresh):
        def reject_closing_buy():
            if (rule.side is OrderSide.BUY and self.close_liquidator is not None
                    and self.close_liquidator.buy_blocked(item.instrument.market)):
                raise OrderNotSent("마감 청산 구간에 진입하여 신규 매수 전송을 차단했습니다.")
        reject_closing_buy()
        super()._before_order_send(item, rule, fresh)
        reject_closing_buy()
        # The closing calendar check follows the parent's final guard. Preserve
        # its freshness/stop guarantees if that additional local work took time.
        if not 0 <= (self.clock()-fresh.fetched_at).total_seconds() <= 15:
            raise OrderNotSent("마감 확인 후 시세가 15초를 초과했거나 미래 시각이므로 전송하지 않습니다.")
        if self._stop.is_set() or self.external_stop.is_set() or not self.orders_enabled:
            raise OrderNotSent("최종 마감 확인 중 자동주문이 OFF 또는 중지되었습니다.")

class LSTM30GUIBridge:
    """No arming or order submission; produces signals for the actual GUI engine."""

    def __init__(self, window, predictors, position_provider):
        self.window, self.diagnostics = window, {}
        self.producer = LSTM30SignalProducer(
            predictors, position_provider=position_provider, quantity=window.lstm_policy.max_quantity,
            max_krw=window.lstm_policy.max_krw, max_usd=window.lstm_policy.max_usd,
            state_path=window.runtime_dir / "exchange/decisions.json", clock=window.engine.clock)

    def begin_sweep(self):
        self.diagnostics = {}

    def _buy_attempted_today(self, item):
        today = market_time(item.instrument.market, self.window.engine.clock()).date()
        with self.window.store.connection() as db:
            rows = db.execute("""SELECT a.started_at FROM attempts a JOIN rules r ON r.id=a.rule_id
                                 WHERE a.watch_id=? AND r.side=?""", (item.id, OrderSide.BUY.value)).fetchall()
        return any(market_time(item.instrument.market, datetime.fromisoformat(row["started_at"])).date() == today
                   for row in rows)

    def publish(self, chart):
        window = self.window
        # This runs on the GUI's worker. Never read Qt widgets here.
        window.engine._ensure_environment()
        if (not window.engine.external_only or window.engine.external_policy != window.lstm_policy
                or window.engine.external_reader is None
                or window.engine.external_reader.path != window.runtime_dir / "exchange/signals.json"):
            window.engine.disarm()
            raise ValueError("LSTM30 신호 실행 정책이 승인된 설정과 다릅니다.")
        if window.engine._stop.is_set():
            return
        allowed = {item.id: item for item in window.lstm_items}
        if any(row.get("watch_id") not in allowed for row in chart.get("stocks", [])):
            raise ValueError("LSTM30 설정 밖의 종목 차트는 주문에 연결할 수 없습니다.")
        payload, diagnostics = self.producer(chart)
        atomic_json(window.runtime_dir / "exchange/signals.json", payload)
        window.engine.external_reader()
        actions = {f"{row['market']}:{row['exchange']}:{row['symbol']}": row["action"] for row in payload["signals"]}
        updated = dict(self.diagnostics)
        for row in diagnostics:
            detail = {key: value for key, value in row.items() if key not in {"error", "watch_id"}}
            detail["action"] = actions.get(row["watch_id"], "none")
            item = allowed[row["watch_id"]]
            if self._buy_attempted_today(item):
                for rule in window.store.rules(item.id):
                    if rule.status == "ready" and rule.side is OrderSide.BUY:
                        window.store.pause_rule(rule.id)
                detail["execution_gate"] = "DAILY_BUY_ATTEMPT_LIMIT"
            updated[item.id] = detail
        self.diagnostics = updated
        atomic_json(window.runtime_dir / "exchange/diagnostics.json", {
            "updated_at": window.engine.clock().isoformat(), "strategy": "lstm30", "diagnostics": updated})
        bad = [row["reason"] for row in diagnostics if row["reason"] in _BAD_DIAGNOSTICS]
        if bad:
            window.engine.safety_reason = "LSTM30_INPUT_UNAVAILABLE"
            window.engine.disarm()
            raise ValueError("LSTM30 입력 검증 실패 · 자동주문 OFF: " + ", ".join(sorted(set(bad))))


class LSTM30SessionController(SessionController):
    """Main's one-shot warmup approval plus an immediate non-Qt stop watchdog."""

    stop_requested = Signal()

    def __init__(self, window, folder, *, auto_arm=False):
        super().__init__(window, folder, auto_arm=auto_arm)
        self.stop_requested.connect(window.stop_monitoring)
        self._watcher_done = Event()
        self._stop_reported = Event()
        self._watcher = Thread(target=self._watch_stop, name="lstm30-gui-stop", daemon=True)
        self._watcher.start()

    def _watch_stop(self):
        while not self._watcher_done.wait(0.5):
            if self._stop_reported.is_set():
                continue
            for name in ("stop.json", "demo-session-stop.json"):
                try:
                    request = json.loads((self.folder / name).read_text(encoding="utf-8"))
                except FileNotFoundError:
                    continue
                except (OSError, ValueError):
                    request = None
                invalid = not isinstance(request, dict)
                current = (not invalid and request.get("session_id") == self.session_id
                           and (name == "stop.json" or request.get("action") == "stop"))
                if invalid or current:
                    self._stop_reported.set()
                    # This event is consulted by main's final paced-send guard.
                    # Qt widgets are touched only via the queued signal.
                    self.window.engine.external_stop.set()
                    self.window.engine.stop()
                    self.stop_requested.emit()
                    break

    def close(self):
        self.cancel_arm()
        self.timer.stop()
        self._watcher_done.set()
        self._watcher.join(timeout=1)

    def report(self):
        super().report()
        status_path = self.folder / "demo-session-status.json"
        state = json.loads(status_path.read_text(encoding="utf-8"))
        window = self.window
        state.update(
            strategy="lstm30", source_id=SOURCE_ID, checkpoints=window.checkpoint_paths,
            buy_thresholds=window.lstm_buy_thresholds,
            checkpoint_buy_thresholds=window.lstm_checkpoint_buy_thresholds,
            close_liquidation=window.close_liquidator.status(),
            diagnostics=dict(window.lstm_bridge.diagnostics), safety_reason=window.engine.safety_reason,
            buy_attempts_per_symbol_local_day=1,
            store_path=str(window.store.path),
            universe=window.lstm_universe.status(),
        )
        state["dashboard"]["connection"]["strategy"] = "lstm30"
        atomic_json(status_path, state)
        # Keep the headless runtime's existing status/stop interface useful after
        # handoff, with this GUI's current session identity and actual permission.
        atomic_json(self.folder / "status.json", {
            **state, "phase": "running" if state["orders_enabled"] else state["state"],
            "heartbeat": state["reported_at"], "gui": True,
            "session_order_counts": state["session_order_status_counts"],
        })


class LSTM30WatchlistDialog(WatchlistDialog):
    """Trained LSTM signals inside the normal dashboard's real ON/OFF controls."""

    def __init__(self, service=None, *, runtime_dir=DEFAULT_RUNTIME_DIR, predictors,
                 quantity=1, max_krw="500000", max_usd="1000", items=None,
                 position_provider=None, clock=utc_now, checkpoint_paths=None, ranked_markets=(),
                 close_all_before_minutes=None, close_all_confirmation=None, parent=None):
        self._lstm_configured = False
        self._lstm_released = False
        self.session_controller = None
        self.close_all_before_minutes = close_all_before_minutes
        if (close_all_before_minutes is None) != (close_all_confirmation is None):
            raise ValueError("마감 전량 청산 시간과 DEMO_CLOSE_ALL_SELLABLE 승인을 함께 지정하세요.")
        self.runtime_dir = Path(runtime_dir).resolve()
        service = service if service is not None else TradingService(mode=TradingMode.DEMO)
        if getattr(service, "mode", None) is not TradingMode.DEMO:
            raise ValueError("LSTM30 GUI는 명시적인 모의투자 서비스만 허용합니다.")
        self.lstm_items = tuple(DEFAULT_ITEMS if items is None else items)
        self.ranked_markets = frozenset(Market(market) for market in ranked_markets)
        if (not self.lstm_items or len({item.id for item in self.lstm_items}) != len(self.lstm_items)
                or any(item.days < 31 for item in self.lstm_items)):
            raise ValueError("중복 없는 관심종목과 최소 31개 일봉이 필요합니다.")
        self.lstm_policy = ExternalPolicy(SOURCE_ID, quantity, Decimal(str(max_krw)), Decimal(str(max_usd)))
        for market in {item.instrument.market.value for item in self.lstm_items} | {market.value for market in self.ranked_markets}:
            if market not in predictors or getattr(predictors[market], "metadata", {}).get("market") != market:
                raise ValueError(f"시장에 맞는 {market} 학습 모델이 필요합니다.")
        self.checkpoint_paths = {key: str(Path(path).resolve()) for key, path in (checkpoint_paths or {}).items()}
        self.lstm_buy_thresholds = {
            market: getattr(predictor, "buy_threshold", predictor.metadata.get("buy_threshold"))
            for market, predictor in predictors.items()}
        self.lstm_checkpoint_buy_thresholds = {
            market: predictor.metadata.get("buy_threshold") for market, predictor in predictors.items()}
        self.session_lock = SessionLock(self.runtime_dir / "session.lock")
        self.session_lock.acquire()
        try:
            store = WatchStore(self.runtime_dir / "watchlist.sqlite3", mode=TradingMode.DEMO)
            existing = store.items()
            if existing:
                existing_fixed = {item.id for item in existing if item.instrument.market not in self.ranked_markets}
                requested_fixed = {item.id for item in self.lstm_items if item.instrument.market not in self.ranked_markets}
                if existing_fixed != requested_fixed:
                    raise ValueError("순위 갱신 대상 이외의 기존 관심종목이 다릅니다. 별도 실행 폴더를 사용하세요.")
            for item in self.lstm_items:
                store.save_item(item)
            self.lstm_items = store.items()
            with store.connection() as db:
                db.executemany("UPDATE watchlist SET days=31 WHERE id=? AND days<31", ((item.id,) for item in self.lstm_items))
            self.lstm_items = store.items()
            for rule in store.rules():
                if rule.status == "ready":
                    store.pause_rule(rule.id)
            atomic_json(self.runtime_dir / "exchange/signals.json", {
                "schema_version": 1, "source_id": SOURCE_ID, "trading_mode": "demo", "signals": []})
            super().__init__(service, store, parent)
            self.engine = _LSTM30AutoTrader(service, store, items=self.lstm_items, clock=clock)
            self.lstm_universe = LSTM30Universe(
                service, store, ranked_markets=self.ranked_markets, baseline_items=self.lstm_items,
                clock=clock, stopped=lambda: self.engine._stop.is_set(), on_change=self._universe_changed)
            self.engine.universe = self.lstm_universe
            self.close_liquidator = CloseLiquidator(
                service, store, self.engine, enabled=close_all_before_minutes is not None,
                minutes_before_close=close_all_before_minutes if close_all_before_minutes is not None else 5,
                confirmation=close_all_confirmation, clock=clock)
            self.engine.close_liquidator = self.close_liquidator
            self.close_timer = QTimer(self)
            self.close_timer.setInterval(1000)
            self.close_timer.timeout.connect(self._close_wakeup)
            self.close_timer.start()
            self.scheduler = ScopedRankingScheduler(
                service, store, universe=self.lstm_universe, clock=clock,
                stopped=lambda: self.engine._stop.is_set(), on_error=self._ranking_error)
            self.external_mode.setChecked(True)
            self.random_demo.setChecked(False)
            self.hourly_ranking.setChecked(bool(self.ranked_markets))
            if self.ranked_markets:
                labels = "/".join("한국" if market is Market.DOMESTIC else "미국" for market in sorted(self.ranked_markets, key=lambda value: value.value))
                self.hourly_ranking.setText(f"{labels} 보통주 TOP100 · 개장/매 정시 갱신 (31개 일봉)")
            self.external_source.setText(SOURCE_ID)
            self.external_quantity.setValue(quantity)
            self.external_krw.setValue(float(self.lstm_policy.max_krw))
            self.external_usd.setValue(float(self.lstm_policy.max_usd))
            self.random_us.setCurrentIndex(self.random_us.findData("limit"))
            self.signal_path.setText(str(self.runtime_dir / "exchange/signals.json"))
            self.chart_path.setText(str(self.runtime_dir / "exchange/charts.json"))
            self.days_input.setValue(31)
            self.interval.setValue(30)
            self.lstm_bridge = LSTM30GUIBridge(self, predictors, position_provider or self._demo_position)
            self._lstm_configured = True
            self.configure_external()
            self._sync_environment()
            self.update_controls()
            self._update_connection()
            self.message.setText("학습된 LSTM30 연결됨 · 자동주문 OFF · 전체 조회 검증 후 ON 가능 · 임의 테스트 주문 없음")
        except Exception:
            self.session_lock.release()
            self._lstm_released = True
            raise

    def _demo_position(self, stock):
        market = Market(stock["market"])
        return demo_position_provider({market.value: self.service.broker(market)}, clock=self.engine.clock)(stock)

    def _universe_changed(self, items):
        # Worker callback: data assignment only; the queued watchlist progress
        # update refreshes the actual tables on the Qt thread.
        self.lstm_items = tuple(items)

    def _ranking_error(self, market, exc):
        self.engine.safety_reason = "TOP100_REFRESH_FAILED"
        self.engine.disarm()

    def _close_wakeup(self):
        if self.monitoring and self.engine.orders_enabled and self.worker is None:
            try:
                if self.close_liquidator.closing_markets():
                    self.timer.stop()
                    self.refresh_all()
            except Exception as exc:
                self.engine.disarm()
                self.message.setText(f"마감 시간 확인 실패 · 자동주문 OFF: {exc}")

    def confirm_automation(self):
        if self.close_all_before_minutes is None:
            return super().confirm_automation()
        self._validate_fixed_configuration()
        return QMessageBox.question(
            self, "LSTM30 모의자동주문 · 계좌 전체 마감 청산 확인",
            "전체 조회 검증 후 모의자동주문을 켭니다. 실제 자금 주문은 하지 않습니다.\n\n"
            f"일반 주문: 최대 {self.lstm_policy.max_quantity}주 · "
            f"{self.lstm_policy.max_krw:,} KRW / {self.lstm_policy.max_usd:,} USD\n"
            f"마감 청산: 각 시장 정규장 종료 {self.close_all_before_minutes}분 전부터 새 매수 차단\n"
            "모의계좌의 모든 국내·미국 보유종목(기존 보유분 포함)을 청산합니다.\n"
            "마감 청산만 일반 수량·금액 한도를 적용하지 않고 확인된 매도가능 수량 전부를 지정가로 주문합니다.\n"
            "미체결·거래정지·API 지연 등으로 마감 전 전량 체결은 보장되지 않습니다.\n"
            "접수 불명확한 주문은 재전송하지 않습니다. OFF/감시 중지는 마감 청산도 중지합니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _run(self, operation, *, streaming=False, done=None, focus_result=False, job_kind="task"):
        if job_kind == "quotes" and streaming and not self.lstm_universe.initialized:
            original = operation
            def with_initial_rankings(progress):
                self.lstm_universe.bootstrap()
                self.scheduler.record_bootstrap()
                progress(("watchlist", None))
                return original(progress)
            operation = with_initial_rankings
        return super()._run(operation, streaming=streaming, done=done, focus_result=focus_result, job_kind=job_kind)

    def _validate_fixed_configuration(self):
        if self.service.mode is not TradingMode.DEMO or self.store.mode is not TradingMode.DEMO:
            self.engine.disarm()
            raise ValueError("LSTM30 GUI는 모의투자 전용입니다.")
        actual = (self.external_mode.isChecked(), self.random_demo.isChecked(), self.external_source.text().strip(),
                  self.external_quantity.value(), Decimal(str(self.external_krw.value())), Decimal(str(self.external_usd.value())),
                  self.hourly_ranking.isChecked(), Path(self.signal_path.text()).resolve(), Path(self.chart_path.text()).resolve())
        expected = (True, False, SOURCE_ID, self.lstm_policy.max_quantity, self.lstm_policy.max_krw, self.lstm_policy.max_usd,
                    bool(self.ranked_markets), self.runtime_dir / "exchange/signals.json", self.runtime_dir / "exchange/charts.json")
        if actual != expected:
            self.engine.disarm()
            raise ValueError("승인된 LSTM30 출처·종목·수량·상한 설정을 변경할 수 없습니다.")
        self.lstm_universe.validate_active()

    def configure_external(self):
        self._validate_fixed_configuration()
        result = super().configure_external()
        self.test_producer = self.lstm_bridge
        return result

    def refresh_all(self):
        if not self.worker:
            self._validate_fixed_configuration()
            self.lstm_bridge.begin_sweep()
        return super().refresh_all()

    def activation_failure(self, results, error, *, external_error_baseline):
        base = super().activation_failure(results, error, external_error_baseline=external_error_baseline)
        if base:
            return base
        if not self.lstm_universe.initialized:
            return "승인된 시장의 보통주 TOP100이 확정되지 않았습니다."
        diagnostics = self.lstm_bridge.diagnostics
        if {item.id for item in self.lstm_items} - set(diagnostics):
            return "전체 관심종목의 LSTM30 입력·잔고·모델 검증이 완료되지 않았습니다."
        if any(row["reason"] in _BAD_DIAGNOSTICS for row in diagnostics.values()):
            return "LSTM30 입력 검증 실패로 자동주문 OFF를 유지합니다."
        if self.engine._critical_attempts():
            return "이 세션의 거절/미확정 주문을 확인해야 합니다."
        return ""

    def _sync_environment(self):
        super()._sync_environment()
        self.setWindowTitle("DOCKDACK | LSTM30 모의 실행기 · 학습 모델 자동주문")
        self.environment_caption.setText("학습된 LSTM30 · 30일봉 · 모의투자 전용")
        thresholds = "/".join(f"{('국내' if market == 'domestic' else '미국')} {value:.0%} 이상"
                              for market, value in sorted(self.lstm_buy_thresholds.items()) if value is not None)
        self.environment_notice.setText("모의투자만 사용 · 실제 자금 주문 불가 · "
                                       + (f"매수 점수 {thresholds} · " if thresholds else "")
                                       + "+1% 익절 / -0.8% 손절 · 종목별 당일 매수 시도 1회"
                                       + (f" · 마감 {self.close_all_before_minutes}분 전 계좌 전체 청산"
                                          if self.close_all_before_minutes is not None else ""))
        self.environment_selector.setEnabled(False)
        self.random_demo.setEnabled(False)

    def request_environment(self, mode):
        if TradingMode(mode) is not TradingMode.DEMO:
            self.stop_monitoring()
            self.message.setText("이 LSTM30 실행기는 모의투자 전용입니다. 실전 전환을 차단했습니다.")

    def _random_toggled(self, enabled):
        if enabled:
            self.random_demo.blockSignals(True)
            self.random_demo.setChecked(False)
            self.random_demo.blockSignals(False)
            if getattr(self, "_lstm_configured", False):
                self.engine.disarm()
                self.message.setText("LSTM30 학습 모델만 사용합니다. 무작위 테스트 신호는 실행하지 않습니다.")

    def update_controls(self):
        super().update_controls()
        if not getattr(self, "_lstm_configured", False):
            return
        for widget in (self.environment_selector, self.edit_panel, self.rule_panel, self.ranking_button,
                       self.hourly_ranking, self.strategy_panel, self.external_panel, self.interval):
            widget.setEnabled(False)
        self.random_demo.setEnabled(False)

    def connection_status(self):
        status = super().connection_status()
        status["strategy"] = "lstm30"
        return status

    def _update_connection(self):
        super()._update_connection()
        if not getattr(self, "_lstm_configured", False):
            return
        self.signal_connection_panel.source_label.setText(f"학습된 LSTM30 · 같은 실행기에서 추론 · source_id: {SOURCE_ID}")
        state = "연결 활성" if self.monitoring and self.engine.external_reader else "연결 설정됨 · 감시 중지"
        self.connection_summary.setText(f"LSTM30 학습 모델 · {state} | {self._market_summary}")

    def start_session(self, confirmation=None):
        if confirmation not in (None, "DEMO_AUTOTRADE"):
            raise ValueError("DEMO_AUTOTRADE 확인만 허용합니다.")
        if self.session_controller is not None:
            raise RuntimeError("이 창의 시작 요청은 한 번만 사용할 수 있습니다. ON/OFF 버튼을 사용하세요.")
        self.session_controller = LSTM30SessionController(
            self, self.runtime_dir, auto_arm=confirmation == "DEMO_AUTOTRADE")
        self.session_controller.start()
        return self.session_controller

    def shutdown(self):
        self.stop_monitoring()
        if self.worker is not None or self._inspection_worker is not None:
            return False
        if self.session_controller is not None:
            self.session_controller.report()
            self.session_controller.close()
        for timer in (self.timer, self.schedule_timer, self.environment_timer, self.order_status_timer,
                      self.health_timer, self.close_timer):
            timer.stop()
        if not self._lstm_released:
            self.session_lock.release()
            self._lstm_released = True
        return True

    def closeEvent(self, event):
        if self.shutdown():
            event.accept()
        else:
            event.ignore()

    def reject(self):
        if self.shutdown():
            super().reject()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--arm", choices=["DEMO_AUTOTRADE"], help="One-time approval after the GUI completes a valid orders-OFF warmup")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-krw", default="500000")
    parser.add_argument("--max-usd", default="1000")
    parser.add_argument("--buy-threshold", type=float, default=0.4,
                        help="Inclusive model-score BUY threshold (default 0.4); does not change weights")
    parser.add_argument("--close-all-before-minutes", type=int,
                        help="Close every DEMO account holding this many minutes before each market close; SELL-only full sellable size")
    parser.add_argument("--confirm-close-all", choices=["DEMO_CLOSE_ALL_SELLABLE"],
                        help="Explicit consent for all existing holdings and closing-only quantity/notional limit exemption")
    parser.add_argument("--domestic-checkpoint", type=Path, default=Path("models/lstm30/domestic.pt"))
    parser.add_argument("--us-checkpoint", type=Path, default=Path("models/lstm30/us.pt"))
    parser.add_argument("--symbol", action="append", help="Repeat MARKET:EXCHANGE:SYMBOL; defaults Samsung and Apple")
    parser.add_argument("--top-us100", action="store_true", help="Select verified US common-stock turnover TOP100, then refresh only US at session open/hourly")
    parser.add_argument("--top-domestic100", action="store_true", help="Select verified Korean common-stock turnover TOP100, then refresh only Korea at session open/hourly")
    args = parser.parse_args(argv)
    items = None
    if args.symbol:
        try:
            items = [WatchItem(Instrument(Market(market), symbol, exchange), days=31)
                     for market, exchange, symbol in (value.split(":") for value in args.symbol)]
        except (ValueError, TypeError) as exc:
            parser.error(f"Invalid --symbol: {exc}")
    paths = {"domestic": args.domestic_checkpoint, "us": args.us_checkpoint}
    ranked_markets = ({Market.US} if args.top_us100 else set()) | ({Market.DOMESTIC} if args.top_domestic100 else set())
    markets = {item.instrument.market.value for item in (items or DEFAULT_ITEMS)} | {market.value for market in ranked_markets}
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        from dockdack.ml30 import Predictor
        predictors = {market: Predictor(paths[market], device="cpu", buy_threshold=args.buy_threshold)
                      for market in markets}
        window = LSTM30WatchlistDialog(
            runtime_dir=args.runtime_dir, predictors=predictors, quantity=args.quantity,
            max_krw=args.max_krw, max_usd=args.max_usd, items=items,
            checkpoint_paths={market: paths[market] for market in markets}, ranked_markets=ranked_markets,
            close_all_before_minutes=args.close_all_before_minutes, close_all_confirmation=args.confirm_close_all)
    except Exception as exc:
        QMessageBox.critical(None, "LSTM30 모의 실행기 시작 실패", f"자동주문은 시작되지 않았습니다.\n{exc}")
        return 1
    previous = {}
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, lambda *_: window.stop_monitoring())
        window.show()
        def start():
            try:
                window.start_session(args.arm)
            except Exception as exc:
                window.stop_monitoring()
                QMessageBox.critical(window, "LSTM30 모의 감시 시작 실패", f"자동주문 OFF를 유지합니다.\n{exc}")
        QTimer.singleShot(0, start)
        return app.exec()
    finally:
        # An ordinary close is accepted only after the GUI worker has drained.
        window.stop_monitoring()
        window.pool.waitForDone()
        window.inspection_pool.waitForDone()
        app.processEvents()
        window.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
