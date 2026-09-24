"""Mark1 inside the existing DEMO dashboard, initially stopped and orders OFF."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from dockdack.exceptions import OrderNotSent
from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.lstm30_adapter import atomic_json, previous_trading_day, read_json
from dockdack.lstm30_gui import LSTM30WatchlistDialog, _LSTM30AutoTrader
from dockdack.lstm30_runtime import DEFAULT_ITEMS
from dockdack.mark1_adapter import Mark1SignalProducer, SOURCE_ID, STRATEGY_NOTICE, TARGET_BASIS, consecutive_completed_bars, decide_position
from dockdack.market_schedule import calendar_for
from dockdack.models import Market, OrderSide
from dockdack.watchlist import WatchItem, utc_now


DEFAULT_RUNTIME_DIR = Path(".dockdack/mark1-demo")


class Mark1AutoTrader(_LSTM30AutoTrader):
    """Revalidate price-dependent entries at preflight and the final send guard.

    The final guard is local only: it cannot ask the broker for another price
    while inside the HTTP transport's pacing callback. It rechecks the last
    preflight quote (at most 15 seconds old), the actual limit price, and model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictors = {}
        self._mark1_entry = None

    def _check_entry(self, item, fresh, *, limit_price):
        self._validate_snapshot(item, fresh)
        now = self.clock()
        if not 0 <= (now - fresh.fetched_at).total_seconds() <= 15:
            raise ValueError("mark1 매수 재검증 시세가 오래되었거나 미래 시각입니다.")
        inst = item.instrument
        predictor = self.predictors.get(inst.market.value)
        if predictor is None or getattr(predictor, "metadata", {}).get("market") != inst.market.value:
            raise ValueError("mark1 매수 직전 재검증에 필요한 시장 모델이 없습니다.")
        today = market_time(inst.market, now).date()
        history = fresh.history.bars[-item.days:]
        stock = {"market": inst.market.value, "complete": len(history) >= 31,
                 "available_days": len(history), "bars": [
                     {"date": bar.day.isoformat(), "is_current_day": bar.day == today,
                      **{key: str(getattr(bar, key)) for key in ("open", "high", "low", "close", "volume")}}
                     for bar in history]}
        bars = consecutive_completed_bars(stock, now)
        prices = [fresh.quote.price]
        if limit_price is None:
            raise ValueError("mark1 매수는 검증한 지정가 주문만 허용합니다.")
        if limit_price != fresh.quote.price:
            prices.append(limit_price)
        for price in prices:
            prediction = predictor.predict(bars, current_price=price)
            decision = decide_position(current_price=price, quantity=0, sellable_quantity=0, prediction=prediction)
            if decision["action"] != "buy":
                raise ValueError("mark1 매수 직전 현재가/지정가의 보정 성공확률이 50%를 초과하지 않습니다.")
        if not 0 <= (self.clock() - fresh.fetched_at).total_seconds() <= 15:
            raise ValueError("mark1 추론 중 시세가 만료되었습니다.")

    def _preflight(self, item, rule, snapshot):
        if rule.side is OrderSide.BUY:
            self._mark1_entry = None
        request, fresh, effective_rule = super()._preflight(item, rule, snapshot)
        if rule.side is OrderSide.BUY:
            self._check_entry(item, fresh, limit_price=request.price)
            self._mark1_entry = (rule.id, item.id, fresh.fetched_at, fresh.quote.price, request.price)
        return request, fresh, effective_rule

    def _before_order_send(self, item, rule, fresh):
        super()._before_order_send(item, rule, fresh)
        if rule.side is not OrderSide.BUY:
            return
        try:
            entry = self._mark1_entry
            if entry is None or entry[:4] != (rule.id, item.id, fresh.fetched_at, fresh.quote.price):
                raise ValueError("mark1 현재가 재검증 기록이 주문과 일치하지 않습니다.")
            self._check_entry(item, fresh, limit_price=entry[4])
            if self._stop.is_set() or self.external_stop.is_set() or not self.orders_enabled:
                raise ValueError("mark1 최종 추론 중 자동주문 OFF 또는 중지가 요청되었습니다.")
            # Inference may take time: re-read superseding HOLD/SELL, the closing
            # boundary, OFF/stop and all original final-send guards afterward.
            super()._before_order_send(item, rule, fresh)
        except Exception as exc:
            raise OrderNotSent(str(exc) or type(exc).__name__) from exc


class Mark1WatchlistDialog(LSTM30WatchlistDialog):
    """Actual external-signal ingestion and engine, with mark0 isolation."""

    source_id = SOURCE_ID
    strategy_id = "mark1"
    producer_class = Mark1SignalProducer
    engine_class = Mark1AutoTrader

    def __init__(self, service=None, *, runtime_dir=DEFAULT_RUNTIME_DIR, **kwargs):
        root = Path(runtime_dir).resolve()
        marker = root / "strategy.json"
        if marker.exists():
            if read_json(marker).get("source_id") != SOURCE_ID:
                raise ValueError("다른 전략의 실행 폴더입니다. mark1 전용 폴더를 사용하세요.")
        elif (root / "watchlist.sqlite3").exists() or (root / "exchange/decisions.json").exists():
            raise ValueError("기존 실행 장부를 재사용할 수 없습니다. 비어 있는 mark1 전용 폴더를 사용하세요.")
        # The shared dashboard reports BOTH exchange sessions even with only
        # one watched market. Construct only their current-year calendars here,
        # before its worker and status reporter can race in pandas caches.
        # This is local calendar work, never a broker/account/network request.
        now = kwargs.get("clock", utc_now)()
        for market in Market:
            calendar_for(market, market_time(market, now).year)
            previous_trading_day(market.value, now)
        super().__init__(service, runtime_dir=root, **kwargs)
        self.engine.predictors = dict(kwargs["predictors"])
        atomic_json(marker, {"strategy": self.strategy_id, "source_id": self.source_id, "target_basis": TARGET_BASIS})
        page = QWidget()
        layout = QVBoxLayout(page)
        self.mark1_strategy = QLabel(STRATEGY_NOTICE)
        self.mark1_strategy.setWordWrap(True)
        layout.addWidget(self.mark1_strategy)
        descriptions = []
        for market, predictor in sorted(kwargs["predictors"].items()):
            metadata = predictor.metadata
            model_name = metadata.get("variant", metadata.get("model_name", metadata.get("architecture", "mark1")))
            descriptions.append(f"{'국내' if market == 'domestic' else '미국'}: {model_name}")
        self.mark1_model_summary = QLabel("연결된 학습 모델 · " + " / ".join(descriptions))
        self.mark1_model_summary.setWordWrap(True)
        layout.addWidget(self.mark1_model_summary)
        self.mark1_limitations = QLabel(
            "연구용 모델 · 50% 초과 신호의 성능·수익성 미입증. 표시 확률이 실제 성공률을 보장하지 않습니다.\n"
            "입력: 완료된 30일 OHLCV + 현재가 질문 토큰 1개 (31개). "
            "당일 최종 고가·저가·종가·거래량은 입력하지 않습니다.\n"
            "확률은 정제 일봉으로 학습·검증한 하루 전체 구간 사건의 추정입니다. "
            "장중 진입 이후의 익절/손절 선후관계나 실제 체결을 보장하지 않습니다. "
            "미래 당일 고가/저가가 둘 다 경계를 넘으면 보수적으로 실패로 학습합니다.\n"
            "거래비용 미반영 기준이며, 갭·슬리피지·조회 지연 때문에 실제 손익은 표시 경계를 넘을 수 있습니다. "
            "자동주문·감시는 시작 시 OFF이며 사용자의 별도 실행이 필요합니다."
        )
        self.mark1_limitations.setWordWrap(True)
        layout.addWidget(self.mark1_limitations)
        self.mark1_model_table = QTableWidget(0, 5)
        self.mark1_model_table.setHorizontalHeaderLabels(["종목", "모델", "모델 추정 확률 (미검증)", "전달 신호", "판단 사유"])
        self.mark1_model_table.horizontalHeader().setStretchLastSection(True)
        self.mark1_model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.mark1_model_table)
        self.workspace_tabs.addTab(page, "mark1 모델 · 매매 조건")
        self.mark1_refresh_timer = QTimer(self)
        self.mark1_refresh_timer.setInterval(500)
        self.mark1_refresh_timer.timeout.connect(lambda: self._update_mark1_table(visible_only=True))
        self.workspace_tabs.currentChanged.connect(lambda _: self._update_mark1_table(visible_only=True))
        self.mark1_refresh_timer.start()
        self._update_mark1_table()
        self.message.setText("mark1 모델 연결됨 · 감시 중지 · 자동주문 OFF · 시작은 사용자의 별도 조작 필요")

    def _sync_environment(self):
        super()._sync_environment()
        self.setWindowTitle("DOCKDACK | mark1 일봉 장벽 모델 · 모의 실행기")
        self.environment_caption.setText("mark1 · 완료 30봉 + 현재가 토큰 · 모의투자 전용")
        self.environment_notice.setText("연구용 · 성능 미입증 · 모의투자만 사용 · " + STRATEGY_NOTICE)

    def _update_connection(self):
        super()._update_connection()
        if not getattr(self, "_lstm_configured", False):
            return
        self.signal_connection_panel.source_label.setText(f"mark1 보수적 일봉 장벽 모델 · source_id: {self.source_id}")
        state = "연결 활성" if self.monitoring and self.engine.external_reader else "연결 설정됨 · 감시 중지"
        self.connection_summary.setText(f"mark1 · 모델 추정 확률 > 50% / +1% 익절 / -0.9% 손절 · {state} | {self._market_summary}")

    def _update_mark1_table(self, *, visible_only=False):
        if visible_only and not self.mark1_model_table.isVisible():
            return
        diagnostics = dict(self.lstm_bridge.diagnostics)
        rows = []
        for item in self.lstm_items:
            row = diagnostics.get(item.id, {})
            prediction = row.get("prediction", {})
            probability = prediction.get("probability_success")
            display = "조회 전 / 모델 판단 없음" if probability is None else f"{float(probability):.2%}"
            values = (item.instrument.symbol, row.get("model_name", "—"), display,
                      row.get("action", "미전달"), row.get("reason", "감시를 시작하지 않았습니다"))
            rows.append(values)
        signature = tuple(rows)
        if signature == getattr(self, '_mark1_table_signature', None):
            return
        self._mark1_table_signature = signature
        self.mark1_model_table.setRowCount(len(rows))
        for index, values in enumerate(rows):
            for column, value in enumerate(values):
                cell = self.mark1_model_table.item(index, column)
                if cell is None:
                    self.mark1_model_table.setItem(index, column, QTableWidgetItem(str(value)))
                elif cell.text() != str(value):
                    cell.setText(str(value))
        self.mark1_model_table.resizeColumnsToContents()

    def shutdown(self):
        result = super().shutdown()
        if result and hasattr(self, "mark1_refresh_timer"):
            self.mark1_refresh_timer.stop()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-krw", default="500000")
    parser.add_argument("--max-usd", default="1000")
    parser.add_argument("--domestic-checkpoint", type=Path, default=Path("models/mark1/domestic.pt"))
    parser.add_argument("--us-checkpoint", type=Path, default=Path("models/mark1/us.pt"))
    parser.add_argument("--symbol", action="append", help="Repeat MARKET:EXCHANGE:SYMBOL; defaults Samsung and Apple")
    parser.add_argument("--top-us100", action="store_true")
    parser.add_argument("--top-domestic100", action="store_true")
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
        from dockdack.mark1_inference import Predictor
        predictors = {market: Predictor(paths[market], device="cpu") for market in markets}
        window = Mark1WatchlistDialog(
            runtime_dir=args.runtime_dir, predictors=predictors, quantity=args.quantity,
            max_krw=args.max_krw, max_usd=args.max_usd, items=items,
            checkpoint_paths={market: paths[market] for market in markets}, ranked_markets=ranked_markets,
        )
    except Exception as exc:
        QMessageBox.critical(None, "mark1 시작 실패", f"자동주문은 시작되지 않았습니다.\n{exc}")
        return 1
    previous = {}
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, lambda *_: window.stop_monitoring())
        window.show()
        # No automatic session start or order-arming path in this launcher.
        return app.exec()
    finally:
        window._close_when_idle = True
        window._pending_environment = None
        window.stop_monitoring()
        window.pool.waitForDone()
        window.inspection_pool.waitForDone()
        window.activity_pool.waitForDone()
        app.processEvents()
        window.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
