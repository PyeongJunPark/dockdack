"""Window-management regressions; all brokers and stores are isolated test doubles."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QSignalSpy, QTest
    from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
    from dockdack.gui import TradingWindow
    from dockdack.watch_gui import WatchlistDialog

from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService
from test_gui import FakeService


@unittest.skipUnless(HAS_QT, "Install the gui extra to run window-control tests")
class WindowControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WatchStore(Path(self.temp.name) / "window-test.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "테스트")
        self.store.save_item(self.item)
        self.window = WatchlistDialog(self.service, self.store)
        self.window.hourly_ranking.setChecked(False)
        self.window.resize(1180, 820)
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        self.extra_windows = []

    def tearDown(self):
        for window in self.extra_windows:
            window.timer.stop()
            window.close()
            window.deleteLater()
        self.window.stop_monitoring()
        self.assertIsNone(self.window.worker, "Window controls must never start a broker worker")
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def process(self):
        self.app.processEvents()

    def simulate_running_session(self):
        """Arm only the in-memory fake engine without polling or submitting."""
        self.store.add_rule(TriggerRule.create(self.item, "price_ge", "buy", 1, Decimal(1000), Decimal(100)))
        self.window.monitoring = True
        self.window.engine.enable_orders("DEMO_AUTOTRADE")
        self.window.timer.start(600_000)
        self.window.schedule_timer.start(600_000)
        self.window.update_controls()

    def assert_session_untouched(self):
        self.assertTrue(self.window.monitoring)
        self.assertTrue(self.window.engine.orders_enabled)
        self.assertFalse(self.window.engine._stop.is_set())
        self.assertTrue(self.window.timer.isActive())
        self.assertTrue(self.window.schedule_timer.isActive())
        self.assertIsNone(self.window.worker)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.history_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def assert_native_controls_without_duplicate_buttons(self, window):
        for flag in (Qt.WindowType.WindowMinimizeButtonHint, Qt.WindowType.WindowMaximizeButtonHint,
                     Qt.WindowType.WindowCloseButtonHint):
            self.assertTrue(window.windowFlags() & flag)
        controls = window.window_controls
        self.assertFalse(hasattr(controls, "minimize_button"))
        self.assertFalse(hasattr(controls, "maximize_button"))
        button_labels = {button.text() for button in window.findChildren(QPushButton)}
        self.assertTrue({"최소화", "최대화", "이전 크기"}.isdisjoint(button_labels))
        self.assertTrue(controls.fullscreen_button.isVisible())
        self.assertTrue(controls.fullscreen_button.isEnabled())

    def test_watchlist_has_native_controls_and_only_fullscreen_toolbar_button(self):
        self.assert_native_controls_without_duplicate_buttons(self.window)

    def test_native_maximize_restores_normal_geometry(self):
        geometry = self.window.geometry()
        self.window.showMaximized()
        self.process()
        self.assertTrue(self.window.isMaximized())
        self.window.showNormal()
        self.process()
        self.assertFalse(self.window.isMaximized())
        self.assertFalse(self.window.isFullScreen())
        self.assertEqual(self.window.geometry(), geometry)

    def test_fullscreen_toolbar_button_restores_normal_window_geometry(self):
        geometry = self.window.geometry()
        self.window.window_controls.fullscreen_button.click()
        self.process()
        self.assertTrue(self.window.isFullScreen())
        self.window.window_controls.fullscreen_button.click()
        self.process()
        self.assertFalse(self.window.isFullScreen())
        self.assertFalse(self.window.isMaximized())
        self.assertEqual(self.window.geometry(), geometry)

    def test_fullscreen_returns_to_prior_maximized_state(self):
        controls = self.window.window_controls
        normal_geometry = self.window.geometry()
        self.window.showMaximized()
        self.process()
        self.assertTrue(self.window.isMaximized())
        controls.toggle_fullscreen()
        self.process()
        self.assertTrue(self.window.isFullScreen())
        controls.leave_fullscreen()
        self.process()
        self.assertFalse(self.window.isFullScreen())
        self.assertTrue(self.window.isMaximized())
        self.window.showNormal()
        self.process()
        self.assertEqual(self.window.geometry(), normal_geometry)

    def test_native_titlebar_state_changes_refresh_toolbar_labels(self):
        controls = self.window.window_controls
        self.window.showMaximized()  # Same state transition as the native maximize button.
        self.process()
        self.assertEqual(controls.fullscreen_button.text(), "전체화면 · F11")
        self.assertEqual(controls.fullscreen_button.accessibleName(), "전체화면 · F11")
        self.window.showNormal()
        self.process()
        self.assertEqual(controls.fullscreen_button.text(), "전체화면 · F11")
        self.window.showFullScreen()
        self.process()
        self.assertIn("창모드", controls.fullscreen_button.text())
        self.assertEqual(controls.fullscreen_button.accessibleName(), "창모드 · F11")
        self.window.showNormal()
        self.process()
        self.assertIn("전체화면", controls.fullscreen_button.text())

    def test_minimize_and_taskbar_restore_preserve_fullscreen_return_state(self):
        self.simulate_running_session()
        controls = self.window.window_controls
        self.window.showMaximized()
        controls.toggle_fullscreen()
        self.process()
        self.window.showMinimized()
        self.process()
        self.assertTrue(self.window.isMinimized())
        self.assert_session_untouched()
        # A native taskbar restore removes only the minimized flag, preserving
        # fullscreen/maximized flags instead of forcing the window to normal.
        self.window.setWindowState(self.window.windowState() & ~Qt.WindowState.WindowMinimized)
        self.window.show()
        self.window.activateWindow()
        self.process()
        self.assertFalse(self.window.isMinimized())
        self.assertTrue(self.window.isFullScreen())
        controls.leave_fullscreen()
        self.process()
        self.assertTrue(self.window.isMaximized())
        self.assert_session_untouched()

    def test_f11_toggles_fullscreen_from_focused_input_and_escape_only_leaves_fullscreen(self):
        rejected = QSignalSpy(self.window.rejected)
        self.window.symbol_input.setFocus()
        self.process()
        QTest.keyClick(self.window.symbol_input, Qt.Key.Key_F11)
        self.process()
        self.assertTrue(self.window.isFullScreen())
        QTest.keyClick(self.window.symbol_input, Qt.Key.Key_F11)
        self.process()
        self.assertFalse(self.window.isFullScreen())
        QTest.keyClick(self.window.symbol_input, Qt.Key.Key_F11)
        self.process()
        QTest.keyClick(self.window.symbol_input, Qt.Key.Key_Escape)
        self.process()
        self.assertFalse(self.window.isFullScreen())
        self.assertTrue(self.window.isVisible())
        self.assertEqual(rejected.count(), 0)

    def test_escape_in_windowed_watchlist_does_not_reject_or_stop_armed_session(self):
        self.simulate_running_session()
        rejected = QSignalSpy(self.window.rejected)
        self.window.symbol_input.setFocus()
        self.process()
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.process()
        self.assertTrue(self.window.isVisible())
        self.assertEqual(rejected.count(), 0)
        self.assert_session_untouched()

    def test_parent_shortcuts_do_not_hijack_active_modal_confirmation(self):
        self.simulate_running_session()
        for fullscreen in (False, True):
            with self.subTest(parent_fullscreen=fullscreen):
                if fullscreen:
                    self.window.window_controls.toggle_fullscreen()
                    self.process()
                confirmation = QMessageBox(QMessageBox.Icon.Question, "모의 확인 테스트", "진행할까요?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self.window)
                confirmation.setDefaultButton(QMessageBox.StandardButton.No)
                confirmation.setWindowModality(Qt.WindowModality.ApplicationModal)
                finished = QSignalSpy(confirmation.finished)
                try:
                    confirmation.show()
                    confirmation.activateWindow()
                    self.process()
                    self.assertIs(self.app.activeModalWidget(), confirmation)
                    QTest.keyClick(confirmation, Qt.Key.Key_F11)
                    self.process()
                    self.assertEqual(self.window.isFullScreen(), fullscreen)
                    self.assertFalse(confirmation.isFullScreen())
                    QTest.keyClick(confirmation, Qt.Key.Key_Escape)
                    self.process()
                    self.assertFalse(confirmation.isVisible())
                    self.assertEqual(finished.count(), 1)
                    self.assertEqual(confirmation.result(), QMessageBox.StandardButton.No)
                    self.assertEqual(self.window.isFullScreen(), fullscreen)
                    self.assertTrue(self.window.isVisible())
                    self.assert_session_untouched()
                finally:
                    confirmation.close()
                    confirmation.deleteLater()
                    self.process()

    def test_minimize_restore_and_window_mode_changes_do_not_touch_trading_lifecycle(self):
        self.simulate_running_session()
        lifecycle = ((self.window, "stop_monitoring"), (self.window, "start_monitoring"),
                     (self.window, "refresh_all"), (self.window.engine, "disarm"),
                     (self.window.engine, "stop"), (self.window.engine, "enable_orders"))
        with ExitStack() as stack:
            spies = [stack.enter_context(patch.object(owner, name, wraps=getattr(owner, name))) for owner, name in lifecycle]
            self.window.showMinimized()
            self.process()
            self.assertTrue(self.window.isMinimized())
            self.assert_session_untouched()
            self.window.showNormal()
            self.window.activateWindow()
            self.process()
            self.assertFalse(self.window.isMinimized())
            self.window.showMaximized()
            self.window.window_controls.toggle_fullscreen()
            self.window.window_controls.leave_fullscreen()
            self.process()
            self.assert_session_untouched()
            for spy in spies:
                spy.assert_not_called()

    def test_main_trading_window_shares_native_controls_and_f11_escape_behavior(self):
        service = FakeService()
        window = TradingWindow(service, store=self.window.store)
        self.extra_windows.append(window)
        window.timer.stop()
        window.show()
        window.activateWindow()
        self.process()
        self.assert_native_controls_without_duplicate_buttons(window)
        window.showMaximized()
        self.process()
        self.assertTrue(window.isMaximized())
        QTest.keyClick(window, Qt.Key.Key_F11)
        self.process()
        self.assertTrue(window.isFullScreen())
        QTest.keyClick(window, Qt.Key.Key_Escape)
        self.process()
        self.assertFalse(window.isFullScreen())
        self.assertTrue(window.isMaximized())
        self.assertTrue(window.isVisible())
        self.assertIsNone(window._worker)
        self.assertEqual(service.submitted, [])


if __name__ == "__main__":
    unittest.main()
