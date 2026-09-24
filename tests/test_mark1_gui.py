"""Actual dashboard + external-reader proofs; all broker services are fake."""

import importlib.util
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
HAS_ML = importlib.util.find_spec("torch") is not None
if HAS_QT:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from dockdack.mark1_gui import Mark1WatchlistDialog

from dockdack.gui_service import Instrument
from dockdack.exceptions import OrderNotSent
from dockdack.history import DailyBar, DailyHistory
from dockdack.lstm30_adapter import read_json
from dockdack.mark1_adapter import SOURCE_ID
from dockdack.market_schedule import EXTRA_CLOSURES, calendar_for
from dockdack.models import Market
from dockdack.watchlist import WatchItem
from test_autotrade import FakeTradingService, NOW, position


class CalendarFakeService(FakeTradingService):
    def history(self, inst, days):
        self.history_calls += 1
        calendar = calendar_for(inst.market, 2026)
        dates = [stamp.date() for stamp in calendar.sessions
                 if stamp.date() <= date(2026, 9, 11) and (inst.market, stamp.date()) not in EXTRA_CLOSURES][-days:]
        bars = tuple(DailyBar(day, Decimal(98), Decimal(102), Decimal(97), Decimal(100), Decimal(1234))
                     for day in dates)
        return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, days, bars)


@unittest.skipUnless(HAS_QT, "Install gui extra")
class Mark1GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for market in Market:
            calendar_for(market, 2026)
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "mark1"
        self.now = NOW
        self.service = CalendarFakeService()
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "Samsung", 31)
        self.predictor = SimpleNamespace(metadata={"market": "domestic", "buy_threshold": .5, "model_name": "fixture-gru"},
                                        buy_threshold=.5, predict=Mock(return_value={"probability_success": .7,
                                        "predicts_success": True, "buy_threshold": .5}))
        self.windows = []
        self.network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        self.network = self.network_patch.start()

    def clock(self):
        self.now += timedelta(milliseconds=1)
        return self.now

    def position_snapshot(self, stock):
        found = [row for row in self.service.positions if row.symbol == stock["symbol"]]
        quantity = sum((row.quantity for row in found), Decimal(0))
        sellable = sum((row.sellable_quantity for row in found), Decimal(0))
        cost = sum((row.quantity * row.average_price for row in found), Decimal(0))
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None, "fetched_at": self.clock().isoformat()}

    def window(self):
        window = Mark1WatchlistDialog(self.service, runtime_dir=self.root, predictors={self.item.instrument.market.value: self.predictor},
                                     quantity=1, max_krw="1000", max_usd="1000", items=[self.item],
                                     position_provider=self.position_snapshot, clock=self.clock)
        self.windows.append(window)
        window.show()
        self.app.processEvents()
        return window

    def wait_idle(self, window):
        deadline = time.monotonic() + 30
        idle = 0
        while idle < 3:
            self.app.processEvents()
            idle = 0 if (window.worker or window._inspection_worker or window.pending_auto_arm
                         or window._activity_worker or window._schedule_probe) else idle + 1
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def tearDown(self):
        try:
            for window in reversed(self.windows):
                window._close_when_idle = True
                window._pending_environment = None
                window.stop_monitoring()
                self.wait_idle(window)
                window.shutdown()
                window.close()
                window.deleteLater()
            self.app.processEvents()
            self.network.assert_not_called()
        finally:
            self.network_patch.stop()
            self.temp.cleanup()

    def test_construction_off_with_explicit_correct_strategy(self):
        window = self.window()
        self.assertFalse(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        self.assertEqual(window.external_source.text(), SOURCE_ID)
        self.assertIn("50% 초과", window.environment_notice.text())
        self.assertIn("-0.9%", window.environment_notice.text())
        self.assertIn("장중 선후관계 미검증", window.mark1_strategy.text())
        self.assertIn("성능·수익성 미입증", window.mark1_limitations.text())
        self.assertIn("fixture-gru", window.mark1_model_summary.text())
        self.assertNotIn("-0.8%", window.environment_notice.text())
        self.assertEqual(read_json(self.root / "strategy.json")["source_id"], SOURCE_ID)

    def test_real_gui_receives_signal_and_probability_without_order(self):
        window = self.window()
        window.start_session()
        self.wait_idle(window)
        window._update_mark1_table()
        self.assertTrue(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(window.mark1_model_table.item(0, 2).text(), "70.00%")
        self.assertEqual(window.mark1_model_table.item(0, 3).text(), "buy")
        self.assertEqual(window.mark1_model_table.item(0, 1).text(), "fixture-gru")
        with window.store.connection() as db:
            row = db.execute("SELECT source_id,payload FROM external_signals").fetchone()
        self.assertEqual((row["source_id"], json.loads(row["payload"])["action"]), (SOURCE_ID, "buy"))
        window.session_controller.report()
        self.assertEqual(read_json(self.root / "status.json")["strategy"], "mark1")
        self.assertEqual(read_json(self.root / "status.json")["source_id"], SOURCE_ID)

    def test_exact_half_probability_is_hold_in_actual_reader(self):
        self.predictor.predict.return_value = {"probability_success": .5, "predicts_success": False, "buy_threshold": .5}
        window = self.window()
        window.start_session()
        self.wait_idle(window)
        payload = read_json(self.root / "exchange/signals.json")
        self.assertEqual(payload["signals"][0]["action"], "hold")
        self.assertEqual(self.service.submitted, [])

    def test_sell_guard_at_point_nine_transmitted_without_submitting(self):
        self.service.positions = (position(),)
        self.service.prices = [Decimal("99.1")]
        window = self.window()
        window.start_session()
        self.wait_idle(window)
        payload = read_json(self.root / "exchange/signals.json")
        self.assertEqual(payload["signals"][0]["action"], "sell")
        self.assertEqual(payload["signals"][0]["cost_loss_pct"], "0.9")
        self.predictor.predict.assert_not_called()
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(window.engine.orders_enabled)

    def test_existing_unknown_runtime_rejected_before_mutation(self):
        self.root.mkdir()
        ledger = self.root / "watchlist.sqlite3"
        ledger.touch()
        with self.assertRaisesRegex(ValueError, "전용 폴더"):
            self.window()
        self.assertEqual(ledger.stat().st_size, 0)
        self.assertFalse((self.root / "strategy.json").exists())

    def ready_buy(self):
        window = self.window()
        window.start_session()
        self.wait_idle(window)
        rule = next(rule for rule in window.store.rules(self.item.id) if rule.status == "ready")
        return window, rule, window.snapshots[self.item.id]

    def test_new_preflight_price_must_still_exceed_half_probability(self):
        window, rule, snapshot = self.ready_buy()
        self.service.prices = [Decimal("101")]
        self.predictor.predict.side_effect = lambda bars, current_price: {
            "probability_success": .5 if current_price >= 101 else .7,
            "predicts_success": current_price < 101, "buy_threshold": .5}
        with self.assertRaisesRegex(ValueError, "50%"):
            window.engine._preflight(self.item, rule, snapshot)
        self.assertEqual(self.predictor.predict.call_args.kwargs["current_price"], Decimal("101"))
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(window.store.attempts(self.item.id), ())

    def test_valid_preflight_uses_actual_current_quote_and_remains_off(self):
        window, rule, snapshot = self.ready_buy()
        request, fresh, effective_rule = window.engine._preflight(self.item, rule, snapshot)
        self.assertEqual(effective_rule.id, rule.id)
        self.assertEqual(request.price, Decimal("100"))
        self.assertEqual(self.predictor.predict.call_args.kwargs["current_price"], fresh.quote.price)
        self.assertEqual(len(self.predictor.predict.call_args.args[0]), 30)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_rounded_buy_limit_price_must_also_pass_model_threshold(self):
        window, rule, snapshot = self.ready_buy()
        self.service.prices = [Decimal("100.2")]
        self.predictor.predict.reset_mock()
        self.predictor.predict.side_effect = lambda bars, current_price: {
            "probability_success": .7 if current_price == Decimal("100.2") else .4,
            "predicts_success": current_price == Decimal("100.2"), "buy_threshold": .5}
        with self.assertRaisesRegex(ValueError, "50%"):
            window.engine._preflight(self.item, rule, snapshot)
        prices = [call.kwargs["current_price"] for call in self.predictor.predict.call_args_list]
        self.assertEqual(prices, [Decimal("100.2"), Decimal("100")])
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_missing_market_model_blocks_preflight(self):
        window, rule, snapshot = self.ready_buy()
        window.engine.predictors.clear()
        with self.assertRaisesRegex(ValueError, "시장 모델"):
            window.engine._preflight(self.item, rule, snapshot)
        self.assertEqual(self.service.submitted, [])

    def test_gap_or_stale_snapshot_blocks_entry_recheck(self):
        window, rule, snapshot = self.ready_buy()
        broken = replace(snapshot, history=replace(snapshot.history, bars=snapshot.history.bars[:-10] + snapshot.history.bars[-9:]))
        with self.assertRaises(ValueError):
            window.engine._check_entry(self.item, broken, limit_price=Decimal("100"))
        stale = replace(snapshot, fetched_at=self.now - timedelta(seconds=16))
        with self.assertRaisesRegex(ValueError, "오래"):
            window.engine._check_entry(self.item, stale, limit_price=Decimal("100"))
        self.assertEqual(self.service.submitted, [])

    def test_slow_preflight_prediction_expiry_blocks(self):
        window, rule, snapshot = self.ready_buy()
        def slow(bars, current_price):
            self.now += timedelta(seconds=16)
            return {"probability_success": .7, "predicts_success": True, "buy_threshold": .5}
        self.predictor.predict.side_effect = slow
        with self.assertRaisesRegex(ValueError, "만료"):
            window.engine._preflight(self.item, rule, snapshot)
        self.assertEqual(self.service.submitted, [])

    def test_final_send_guard_rechecks_model_without_submitting(self):
        window, rule, snapshot = self.ready_buy()
        request, fresh, effective_rule = window.engine._preflight(self.item, rule, snapshot)
        self.assertEqual(effective_rule.id, rule.id)
        # Temporary fake service only. Do not call submit or the polling loop.
        window.engine.enable_orders("DEMO_AUTOTRADE")
        self.assertTrue(window.store.claim(rule, fresh.quote.price, self.clock()))
        self.predictor.predict.return_value = {"probability_success": .5, "predicts_success": False, "buy_threshold": .5}
        with self.assertRaisesRegex(OrderNotSent, "50%"):
            window.engine._before_order_send(self.item, rule, fresh)
        window.engine.disarm()
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(window.engine.orders_enabled)

    def test_final_send_guard_detects_off_during_model_check(self):
        window, rule, snapshot = self.ready_buy()
        _, fresh, _ = window.engine._preflight(self.item, rule, snapshot)
        window.engine.enable_orders("DEMO_AUTOTRADE")
        self.assertTrue(window.store.claim(rule, fresh.quote.price, self.clock()))
        def cancel(bars, current_price):
            window.engine.disarm()
            return {"probability_success": .7, "predicts_success": True, "buy_threshold": .5}
        self.predictor.predict.side_effect = cancel
        with self.assertRaises(OrderNotSent):
            window.engine._before_order_send(self.item, rule, fresh)
        self.assertFalse(window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_calendars_warmed_before_any_broker_work(self):
        from dockdack.lstm30_adapter import previous_trading_day
        with patch("dockdack.mark1_gui.calendar_for", wraps=calendar_for) as calendars, \
                patch("dockdack.mark1_gui.previous_trading_day", wraps=previous_trading_day) as completed:
            window = self.window()
        self.assertEqual({call.args[0] for call in calendars.call_args_list}, set(Market))
        self.assertEqual({call.args[0] for call in completed.call_args_list}, {"domestic", "us"})
        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        self.assertFalse(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)

    @unittest.skipUnless(HAS_ML and all((Path(__file__).resolve().parents[1] / "models/mark1" / f"{market}.pt").exists()
                             for market in ("domestic", "us")), "Local trained checkpoints are optional artifacts")
    def test_actual_trained_checkpoints_reach_fake_gui_reader_while_orders_off(self):
        from dockdack.mark1_inference import Predictor
        for market, symbol, exchange in ((Market.DOMESTIC, "005930", "KRX"), (Market.US, "AAPL", "ND")):
            with self.subTest(market=market):
                # Main v0.0 reads watched symbols only during their own session.
                self.now = NOW if market is Market.DOMESTIC else NOW.replace(hour=14)
                checkpoint = Path(__file__).resolve().parents[1] / "models/mark1" / f"{market.value}.pt"
                self.predictor = Predictor(checkpoint, device="cpu")
                self.item = WatchItem(Instrument(market, symbol, exchange), "Fake market data", 31)
                self.root = Path(self.temp.name) / f"real-model-{market.value}"
                window = self.window()
                self.assertFalse(window.monitoring)
                self.assertFalse(window.engine.orders_enabled)
                window.start_session()
                self.wait_idle(window)
                window._update_mark1_table()
                diagnostic = window.lstm_bridge.diagnostics[self.item.id]
                probability = diagnostic["prediction"]["probability_success"]
                self.assertTrue(0 <= probability <= 1)
                self.assertEqual(diagnostic["action"], "buy" if probability > .5 else "hold")
                self.assertIn(self.predictor.metadata["variant"], window.mark1_model_summary.text())
                self.assertEqual(window.mark1_model_table.item(0, 2).text(), f"{probability:.2%}")
                self.assertFalse(window.engine.orders_enabled)
                self.assertEqual(self.service.submitted, [])
                window.stop_monitoring()
                self.wait_idle(window)


if __name__ == "__main__":
    unittest.main()
