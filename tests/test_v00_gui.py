"""Offline ver 0.0 desktop integration: temporary ledger and fake models only."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
import tempfile
from decimal import Decimal as D
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import DesktopModelBridge, V00Window
    from dockdack.v00_widgets import OrderToast, SourceList

from dockdack.models import AccountSnapshot, Market, TradingMode
from dockdack.portfolio import PortfolioMarketState
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, position
from test_lstm30_adapter import NOW, chart, prediction


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class V00GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.store = WatchStore(self.folder / "ledger.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        # UI construction/selection never needs real credentials or networking.
        self.keys = patch("dockdack.gui_service.KiwoomConfig.from_env",
                          side_effect=AssertionError("test must not read real credentials"))
        self.keys.start()
        self.addCleanup(self.keys.stop)
        self.window = V00Window(self.service, self.store, builtin=False)
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.window.engine.clock = lambda: NOW
        self.drain_activity()

    def drain_activity(self):
        for _ in range(20):
            pool = getattr(self.window, "activity_pool", None)
            if pool is not None:
                pool.waitForDone(1000)
            self.app.processEvents()
            if getattr(self.window, "_activity_worker", None) is None and getattr(self.window, "_schedule_probe", None) is None:
                return
        self.fail("offline activity worker did not finish")

    def tearDown(self):
        self.window.worker = None
        self.window._inspection_worker = None
        self.drain_activity()
        self.window.close()
        for name in ("pool", "inspection_pool", "activity_pool"):
            pool = getattr(self.window, name, None)
            if pool is not None:
                pool.waitForDone(5000)
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_default_ten_percent_and_off_without_any_broker_read_or_order(self):
        self.assertEqual(self.window.buy_percent.value(), 10)
        self.assertTrue(self.window.percent_sizing.isChecked())
        self.assertEqual(self.window.engine.equity_buy_percent, D(10))
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.history_calls, 0)
        self.assertEqual(self.service.submitted, [])
        content = self.window.external_grid.parentWidget()
        self.assertTrue(content.testAttribute(Qt.WidgetAttribute.WA_StyledBackground))
        self.assertIn("#121b2a", content.styleSheet())

    def test_fullscreen_from_maximized_restores_resizable_desktop_without_orders(self):
        self.window.resize(1180, 820)
        self.window.show()
        self.app.processEvents()
        geometry = self.window.geometry()
        self.window.showMaximized()
        self.app.processEvents()
        self.window.window_controls.fullscreen_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.isFullScreen())
        self.window.window_controls.fullscreen_button.click()
        self.app.processEvents()
        self.assertFalse(self.window.isFullScreen())
        self.assertFalse(self.window.isMaximized())
        self.assertEqual(self.window.geometry(), geometry)
        self.window.resize(1120, 800)
        self.app.processEvents()
        self.assertEqual((self.window.width(), self.window.height()), (1120, 800))
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertFalse(self.service.submitted)

    def test_optional_builtin_and_multiple_named_json_sources_can_coexist(self):
        self.window.builtin_lstm.setChecked(True)
        self.window.additional_sources.add_row(source="alpha-one", path=str(self.folder / "alpha.json"))
        self.window.additional_sources.add_row(source="beta-two", path=str(self.folder / "beta.json"))
        self.window.configure_external()
        self.assertIsInstance(self.window.test_producer, DesktopModelBridge)
        self.assertEqual(set(self.window.engine.external_sources), {"lstm30-mark0", "alpha-one", "beta-two"})
        self.assertIsNone(self.window.test_producer.producer)  # Torch/model initialization is lazy.
        self.assertIsNone(self.window.test_producer.predictors)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_no_builtin_uses_external_sources_and_applies_gui_percentage(self):
        self.window.external_source.setText("custom-main")
        self.window.buy_percent.setValue(17.25)
        self.window.configure_external()
        self.assertIsNone(self.window.test_producer)
        self.assertEqual(set(self.window.engine.external_sources), {"custom-main"})
        self.assertEqual(self.window.engine.equity_buy_percent, D("17.25"))

    def test_duplicate_sources_and_output_as_input_are_rejected_without_orders(self):
        for source, path in ((self.window.external_source.text(), self.folder / "another.json"),
                             ("another", Path(self.window.chart_path.text()))):
            self.window.additional_sources.table.setRowCount(0)
            self.window.additional_sources.add_row(source=source, path=str(path))
            with self.assertRaises(ValueError):
                self.window.configure_external()
            self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_demo_real_selector_present_and_cancel_real_warning_does_not_switch_or_send(self):
        selector = self.window.environment_selector
        self.assertEqual(set(selector.buttons), {TradingMode.DEMO, TradingMode.REAL})
        self.assertFalse(selector.buttons[TradingMode.DEMO].isEnabled())
        self.assertTrue(selector.buttons[TradingMode.REAL].isEnabled())
        with patch("dockdack.watch_gui.confirm_environment", return_value=False) as confirm:
            selector.buttons[TradingMode.REAL].click()
        confirm.assert_called_once()
        self.assertIs(self.window.service, self.service)
        self.assertIs(self.window.store.mode, TradingMode.DEMO)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_explicit_real_selection_uses_isolated_fake_environment_and_stays_off(self):
        candidate = FakeTradingService()
        candidate.mode = TradingMode.REAL
        candidate.storage_scope = "b" * 64
        candidate.acknowledge_live_risk = Mock()
        with patch("dockdack.watch_gui.confirm_environment", return_value=True), \
                patch("dockdack.watch_gui.TradingService", return_value=candidate):
            self.window.environment_selector.buttons[TradingMode.REAL].click()
            self.drain_activity()
        self.assertIs(self.window.service, candidate)
        self.assertIs(self.window.store.mode, TradingMode.REAL)
        self.assertNotEqual(self.window.store.path, self.store.path)
        self.assertTrue(self.window.store.path.is_relative_to(self.folder))
        self.assertEqual(self.window.store.order_history(), ())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        candidate.acknowledge_live_risk.assert_called_once_with("REAL_TRADING_RISK_ACKNOWLEDGED")
        self.assertIn("실전", self.window.environment_selector.badge.text())
        self.assertFalse(candidate.submitted)

    def test_holdings_show_fallback_and_explicit_exit_prices_independent_of_watchlist(self):
        held = position(quantity=2, sellable=2)
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC,
                     AccountSnapshot(Market.DOMESTIC, "KRW", (held,)), NOW, NOW)}
        panel = self.window.portfolio_panel
        panel.apply(payload, now=NOW)
        table = panel.tables[Market.DOMESTIC]
        self.assertIn("101", table.item(0, 9).text())
        # KR formatting may use fractional target text or round for display.
        self.assertIn("99", table.item(0, 10).text())
        self.assertIn("평균매입가", table.item(0, 9).toolTip())
        panel.set_exit_targets({self.item.id: {"take_profit_price": D(120), "stop_loss_price": D(95), "source": "my-model"}})
        self.assertIn("120", table.item(0, 9).text())
        self.assertIn("95", table.item(0, 10).text())
        self.assertIn("매수 신호", table.item(0, 9).toolTip())
        self.store.remove_item(self.item.id)
        panel.apply(payload, now=NOW)
        self.assertEqual(table.rowCount(), 1)
        self.assertIn("120", table.item(0, 9).text())

    def test_unknown_holding_venue_is_isolated_and_visible_without_turning_orders_off(self):
        held = replace(position(), market=Market.US, symbol='LITE', exchange='미확인 거래소', currency='USD')
        accounts = {market: AccountSnapshot(market, 'KRW' if market is Market.DOMESTIC else 'USD',
                    (position(),) if market is Market.DOMESTIC else (held,)) for market in Market}
        self.service.safety_account = lambda inst: accounts[inst.market]
        self.window.engine.external_only = False
        self.window.engine.enable_orders('DEMO_AUTOTRADE')
        updates = []
        self.window._refresh_portfolio_worker(updates.append, force=True)
        for update in updates:
            self.window._progress(update)
        targets = next(value for kind, value in updates if kind == 'exit_targets')
        self.assertEqual(targets[self.item.id]['take_profit_price'], D(101))
        self.assertIsNone(targets['us:미확인 거래소:LITE']['take_profit_price'])
        table = self.window.portfolio_panel.tables[Market.US]
        self.assertEqual(table.rowCount(), 1)  # Never hide a held asset.
        self.assertEqual(table.item(0, 9).text(), '확인 필요 · 보류')
        self.assertIn('다른 종목 감시는 계속', table.item(0, 9).toolTip())
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        warnings = [e for e in self.store.events() if '해당 보유종목 자동매도 보류' in e['message']]
        self.assertEqual(len(warnings), 1)
        self.window._refresh_portfolio_worker(lambda _: None, force=True)
        self.assertEqual(len([e for e in self.store.events() if '해당 보유종목 자동매도 보류' in e['message']]), 1)

    def test_target_collection_does_not_swallow_storage_failure(self):
        payload = {Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC,
                     AccountSnapshot(Market.DOMESTIC, 'KRW', (position(),)), NOW, NOW)}
        with patch.object(self.store, 'exit_targets', side_effect=sqlite3.DatabaseError('journal unavailable')):
            with self.assertRaises(sqlite3.DatabaseError):
                self.window._portfolio_exit_targets_worker(payload)

    def test_worker_failure_retries_transient_read_but_disarms_on_broken_ledger(self):
        self.window.engine.external_only = False
        self.window.monitoring = True
        for error, stays_on in ((TimeoutError('temporary read failure'), True),
                                (sqlite3.DatabaseError('journal unavailable'), False)):
            with self.subTest(error=type(error).__name__):
                self.window.engine.enable_orders('DEMO_AUTOTRADE')
                with patch.object(self.window, 'reload_tables'), patch.object(self.window, '_reload_activity'), \
                        patch.object(self.window, '_update_health'):
                    self.window._completed(None, error)
                self.assertEqual(self.window.engine.orders_enabled, stays_on)
                self.assertTrue(self.window.timer.isActive())  # Monitoring still retries, never re-arms.
                self.window.timer.stop()
        self.assertEqual(self.service.submitted, [])

    def test_korean_us_venue_response_allows_domestic_warmup_and_explicit_activation(self):
        from dockdack.kiwoom import _us_position
        held = _us_position({'stk_cd': 'LITE', 'stex_nm': '나스닥', 'poss_qty': '1',
                             'sell_alowq': '1', 'frgn_stk_book_uv': '100', 'now_pric': '100'})
        self.assertEqual(held.exchange, 'ND')
        accounts = {market: AccountSnapshot(market, 'KRW' if market is Market.DOMESTIC else 'USD',
                    (position(),) if market is Market.DOMESTIC else (held,)) for market in Market}
        self.service.safety_account = lambda inst: accounts[inst.market]
        self.window.portfolio.clock = lambda: NOW
        self.window.engine.external_only = False
        self.window.monitoring = True
        self.window.hourly_ranking.setChecked(False)
        self.window._market_open = {Market.DOMESTIC: True, Market.US: False}
        self.window._manual_arm_pending = self.window.pending_auto_arm = True
        self.window._manual_external_error_baseline = 0
        self.window._refresh_portfolio_worker(self.window._progress, force=True)
        results = self.window.engine.poll(checkpoint=lambda: self.window._refresh_portfolio_worker(self.window._progress))
        self.assertFalse(any(isinstance(value, Exception) for value in results.values()))
        with patch('dockdack.portfolio.utc_now', return_value=NOW), patch.object(self.window, 'refresh_all'):
            self.assertTrue(self.window._advance_manual_activation('quotes', results, None))
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual(self.service.submitted, [])

    def test_activity_refresh_moves_sqlite_and_accounting_off_gui_thread(self):
        self.store.event("SYSTEM", "background collection verified", category="system")
        original = self.store.connection
        main_thread = get_ident()
        worker_threads = []
        def connection():
            caller = get_ident()
            if caller == main_thread:
                raise AssertionError("activity refresh read SQLite on GUI thread")
            worker_threads.append(caller)
            return original()
        with patch.object(self.store, "connection", side_effect=connection):
            self.window._reload_activity(force=True)
            self.drain_activity()
        self.assertTrue(worker_threads)
        self.assertEqual(self.window._last_log_error, "")
        view = self.window.operations_panel.logs["system"].table
        self.assertIn("background collection verified", view.item(0, 2).text())
        self.assertIs(self.window.order_history_panel.performance, self.window.trade_journal_panel.journal["performance"])
        self.assertFalse(self.service.submitted)

    def test_notifications_skip_history_and_nonorder_messages_and_advance_in_bounded_batches(self):
        self.store.event(self.item.id, "old order", category="order")
        events = []
        self.window._order_notifications_worker(events.append)
        self.assertEqual(events, [])
        self.store.event(self.item.id, "매수 체결 같은 문구라도 HOLD 신호", category="signal")
        self.window._order_notifications_worker(events.append)
        self.assertEqual(events, [])
        for index in range(61):
            self.store.event(self.item.id, f"new order {index}", category="order")
        self.window._order_notifications_worker(events.append)
        self.assertEqual(len(events[0][1]), 50)
        self.assertTrue(all("new order" in line for line in events[0][1]))
        self.window._order_notifications_worker(events.append)
        self.assertEqual(len(events[1][1]), 11)
        self.window._order_notifications_worker(events.append)
        self.assertEqual(len(events), 2)
        self.assertFalse(self.service.submitted)

    def test_toast_is_reused_bounded_plain_text_and_does_not_activate_window(self):
        toast = self.window.order_toast
        self.assertTrue(toast.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertEqual(toast.message.textFormat(), Qt.TextFormat.PlainText)
        with patch.object(self.window, "activateWindow") as activate:
            for index in range(30):
                toast.notify([f"<b>{index}-{line}</b>" + "x" * 300 for line in range(100)])
            activate.assert_not_called()
        self.assertEqual(len(self.window.findChildren(OrderToast)), 1)
        self.assertLess(len(toast.message.text()), 850)
        self.assertIn("외 97건", toast.message.text())
        self.assertTrue(toast.timer.isSingleShot())
        self.assertEqual(toast.timer.interval(), 7000)

    def test_source_editor_has_sixteen_row_cap_and_requires_complete_rows(self):
        editor = self.window.additional_sources
        for index in range(30):
            editor.add_row(source=f"model-{index}", path=str(self.folder / f"model-{index}.json"))
        self.assertEqual(editor.table.rowCount(), 16)
        self.assertEqual(len(editor.sources()), 16)
        editor.table.item(0, 1).setText("")
        with self.assertRaises(ValueError):
            editor.sources()


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class DesktopModelBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.service = FakeTradingService()
        self.engine = SimpleNamespace(_stop=Event(), clock=lambda: NOW,
            external_policy=SimpleNamespace(max_krw=D(10000000), max_usd=D(10000)),
            external_reader=SimpleNamespace(path=self.folder / "signals.json"))
        self.window = SimpleNamespace(service=self.service, engine=self.engine,
                                      store=SimpleNamespace(path=self.folder / "ledger.sqlite3"))
        self.predictor = SimpleNamespace(metadata={"market": "domestic"}, predict=Mock(return_value=prediction()))

    def payload(self):
        return json.loads(self.engine.external_reader.path.read_text(encoding="utf-8"))

    def test_fake_model_buy_adds_bracket_prices_and_never_sends_orders(self):
        bridge = DesktopModelBridge(self.window, predictors={"domestic": self.predictor})
        bridge.publish(chart())
        signal = self.payload()["signals"][0]
        self.assertEqual(signal["action"], "buy")
        self.assertEqual(D(signal["take_profit_price"]), D(101))
        self.assertEqual(D(signal["stop_loss_price"]), D("99.2"))
        self.predictor.predict.assert_called_once()
        self.assertFalse(self.service.submitted)

    def test_held_model_sell_is_converted_to_hold_for_independent_exit_pass(self):
        self.service.positions = (position(),)
        bridge = DesktopModelBridge(self.window, predictors={"domestic": self.predictor})
        data = chart()
        data["stocks"][0]["price"] = "101"
        bridge.publish(data)
        signal = self.payload()["signals"][0]
        self.assertEqual(signal["action"], "hold")
        self.assertFalse({"quantity", "max_notional", "cost_profit_pct", "cost_loss_pct"} & signal.keys())
        self.predictor.predict.assert_not_called()
        self.assertFalse(self.service.submitted)

    def test_real_mode_output_requires_matching_real_chart_and_uses_fake_models_only(self):
        self.service.mode = TradingMode.REAL
        bridge = DesktopModelBridge(self.window, predictors={"domestic": self.predictor})
        data = chart()
        with self.assertRaises(ValueError):
            bridge.publish(data)
        self.assertFalse(self.engine.external_reader.path.exists())
        data.update(trading_mode="real", source="kiwoom_real")
        bridge.publish(data)
        self.assertEqual(self.payload()["trading_mode"], "real")
        self.assertEqual(self.payload()["signals"][0]["action"], "buy")
        self.assertFalse(self.service.submitted)

    def test_stopped_bridge_does_not_initialize_model_or_write_signal(self):
        self.engine._stop.set()
        bridge = DesktopModelBridge(self.window, predictors={"domestic": self.predictor})
        bridge.publish(chart())
        self.assertIsNone(bridge.producer)
        self.assertFalse(self.engine.external_reader.path.exists())
        self.predictor.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
