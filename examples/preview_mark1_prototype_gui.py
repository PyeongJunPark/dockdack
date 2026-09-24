"""Offline visual/integration proof: actual saved models + read-only historical DB.

Quotes are replayed historical opens, positions are explicitly synthetic/flat.
No credentials, broker HTTP, existing runtime, live quotes, or orders are used.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from unittest.mock import patch

from dockdack.local_data_paths import default_clean_database_dir
from examples.run_mark1_prototype_gui import ROOT, configure_local_dependencies


def historical_input(path, symbol, exchange, day):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT trade_date,open,high,low,close,volume FROM daily_bars "
            "WHERE symbol=? AND exchange=? AND trade_date<? ORDER BY trade_date DESC LIMIT 30",
            (symbol, exchange, day.isoformat()),
        ).fetchall()[::-1]
        target = db.execute("SELECT open FROM daily_bars WHERE symbol=? AND exchange=? AND trade_date=?",
                            (symbol, exchange, day.isoformat())).fetchone()
    if len(rows) != 30 or target is None:
        raise ValueError(f"Missing complete historical replay window: {symbol}")
    return rows, Decimal(target[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/mark1_prototype")
    parser.add_argument("--database-dir", type=Path,
                        default=default_clean_database_dir(ROOT))
    parser.add_argument("--day", type=date.fromisoformat, default=date(2024, 7, 15))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mark1/prototype-gui-preview.png")
    args = parser.parse_args(argv)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_local_dependencies()
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel
    from dockdack.gui_service import Instrument
    from dockdack.history import DailyBar, DailyHistory
    from dockdack.lstm30_adapter import atomic_json
    from dockdack.mark1_prototype_gui import Mark1PrototypeWatchlistDialog
    from dockdack.mark1_prototype_inference import PrototypePredictor
    from dockdack.models import Market, Quote
    from dockdack.watchlist import WatchItem
    sys.path.insert(0, str(ROOT / "tests"))
    from test_autotrade import FakeTradingService

    now = datetime(args.day.year, args.day.month, args.day.day, 14, tzinfo=timezone.utc)
    instruments = [Instrument(Market.DOMESTIC, "005930", "KRX"), Instrument(Market.US, "AAPL", "ND")]
    inputs = {inst.market.value: historical_input(args.database_dir / f"{inst.market.value}_daily_clean.sqlite3",
                                                inst.symbol, inst.exchange, args.day) for inst in instruments}
    predictors = {inst.market.value: PrototypePredictor(args.bundle, inst.market.value) for inst in instruments}
    expected = {market: predictors[market].predict([[float(value) for value in row[1:]] for row in rows],
                                                  current_price=price)
                for market, (rows, price) in inputs.items()}

    class HistoricalService(FakeTradingService):
        def quote(self, inst):
            self.quote_calls += 1
            return Quote(inst.market, inst.symbol, "과거 시가 재생", inst.exchange,
                         inputs[inst.market.value][1], inst.currency)

        def history(self, inst, days):
            self.history_calls += 1
            rows, price = inputs[inst.market.value]
            bars = [DailyBar(date.fromisoformat(row[0]), *(Decimal(str(value)) for value in row[1:])) for row in rows]
            # The day being replayed is not completed: never supply its future OHLCV.
            bars.append(DailyBar(args.day, price, price, price, price, Decimal(0)))
            return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, days, tuple(bars))

    def flat_position(stock):
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": "0", "sellable_quantity": "0", "average_price": None, "fetched_at": now.isoformat()}

    app = QApplication.instance() or QApplication([])
    font_id = QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if families:
        app.setFont(QFont(families[0], 10))
    service = HistoricalService()

    def idle(window):
        deadline, quiet = time.monotonic() + 45, 0
        while quiet < 3:
            app.processEvents()
            quiet = 0 if (window.worker or window._inspection_worker or
                          getattr(window, "_activity_worker", None) or getattr(window, "_schedule_probe", None)) else quiet + 1
            QTest.qWait(10)
            if time.monotonic() > deadline:
                raise RuntimeError("Historical GUI replay worker did not finish")

    with tempfile.TemporaryDirectory(prefix="mark1-prototype-preview-") as folder, \
            patch("requests.sessions.Session.request", side_effect=AssertionError("Offline replay: network forbidden")) as network:
        window = Mark1PrototypeWatchlistDialog(
            service, runtime_dir=folder, predictors=predictors, clock=lambda: now,
            position_provider=flat_position,
            items=[WatchItem(inst, "과거 시가 재생 / 가상 미보유", 31) for inst in instruments],
            checkpoint_paths={market: args.bundle / "manifest.json" for market in predictors},
        )
        try:
            if families:
                window.setStyleSheet(window.styleSheet() + f"\nQWidget {{ font-family: '{families[0]}'; }}")
            banner = QLabel(f"실제 저장 모델 연결 검증 · {args.day} 과거 시가 재생 · 현재 시세/실계좌 아님 · 가상 미보유 · 주문 없음")
            banner.setWordWrap(True)
            banner.setStyleSheet("background: #203c57; color: white; font-size: 15px; font-weight: 700; padding: 12px;")
            window.layout().insertWidget(0, banner)
            window.resize(1600, 1000)
            window.show()
            # Both markets' historical windows are replayed at one clock instant;
            # this offline fixture intentionally ignores current session gating.
            original_preferences = window._apply_execution_preferences
            def replay_preferences():
                original_preferences()
                window.engine.session_only_poll = False
            with patch.object(window, "_apply_execution_preferences", side_effect=replay_preferences):
                window.start_session()  # Historical fake service ONLY; never arms.
                window.engine.session_only_poll = False
            idle(window)
            window.stop_monitoring()
            idle(window)
            window.workspace_tabs.setCurrentIndex(window.workspace_tabs.count() - 1)
            window._update_mark1_table()
            result_rows = []
            for item in window.lstm_items:
                market = item.instrument.market.value
                row = window.lstm_bridge.diagnostics.get(item.id, {})
                actual = row.get("prediction", {}).get("probability_success")
                reference = expected[market]["probability_success"]
                if actual is None or abs(actual - reference) > 1e-12:
                    raise AssertionError(f"Model -> GUI mismatch {market}: {row}; errors={window.errors}")
                result_rows.append({"market": market, "symbol": item.instrument.symbol, "probability_success": actual,
                                    "direct_inference_probability": reference, "absolute_error": abs(actual - reference),
                                    "action": row.get("action"), "reason": row.get("reason")})
            if window.engine.orders_enabled or service.submitted or window.store.attempts():
                raise AssertionError("Offline prototype replay must never create orders or attempts")
            if window.arm_button.isEnabled():
                raise AssertionError("Prototype order-arming button must stay disabled")
            network.assert_not_called()
            window.message.setText("실제 저장 모델 → 확률 → 외부 신호 → GUI 일치 확인 · 과거 데이터 재생 · 네트워크/주문 0건")
            app.processEvents()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.output.resolve())):
                raise RuntimeError("Could not save GUI screenshot")
            result = {"title": "mark1 prototype actual-model historical GUI integration proof", "day": args.day.isoformat(),
                      "not_live_quotes": True, "synthetic_flat_positions": True, "network_calls": network.call_count,
                      "orders": len(service.submitted), "order_attempts": len(window.store.attempts()),
                      "monitoring_after_replay": window.monitoring, "orders_permitted": window.engine.orders_enabled,
                      "read_only_source_databases": True, "rows": result_rows}
            atomic_json(args.output.with_suffix(".json"), result)
            print(args.output.resolve())
            print(result)
        finally:
            window._close_when_idle = True
            window._pending_environment = None
            window.stop_monitoring()
            window.pool.waitForDone()
            window.inspection_pool.waitForDone()
            window.activity_pool.waitForDone()
            idle(window)
            window.shutdown()
            window.close()
            window.deleteLater()
            app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
