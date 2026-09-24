"""Advanced-settings navigation is presentation only; temp DB and fake service."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_v00_mark1_trigger as fixtures
from dockdack.models import TradingMode

if fixtures.HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import MARK1_TRIGGER, MARK11_TRIGGER


@unittest.skipUnless(fixtures.HAS_QT, "Install the gui extra")
class AdvancedSettingsGuiTests(unittest.TestCase):
    drain_activity = fixtures.V00Mark1TriggerTests.drain_activity
    tearDown = fixtures.V00Mark1TriggerTests.tearDown

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixtures.V00Mark1TriggerTests.setUp(self)
        self.service.safety_account = Mock(side_effect=AssertionError("UI navigation must not read accounts"))
        self.service.safety_orders = Mock(side_effect=AssertionError("UI navigation must not read broker orders"))
        self.service.safety_executions = Mock(side_effect=AssertionError("UI navigation must not read executions"))

    def assert_no_trading_io(self):
        self.assertEqual(self.service.submitted, [])
        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        self.service.safety_account.assert_not_called()
        self.service.safety_orders.assert_not_called()
        self.service.safety_executions.assert_not_called()

    def select_models(self, *models):
        for model, check in self.window.external_model_checks.items():
            check.setChecked(model in models)
        self.drain_activity()

    def advanced_visible(self):
        return self.window.workspace_tabs.isTabVisible(self.window.workspace_tabs.indexOf(self.window.tabs))

    def state(self):
        return (self.window._external_config_key(), self.window._chosen_external_models(),
                self.window._chosen_trigger(), self.window.external_mode.isChecked(),
                self.window.engine.orders_enabled, self.window.monitoring,
                self.window.pending_auto_arm, self.service.mode, self.store.mode)

    @contextmanager
    def blocked_state(self, **values):
        previous = {name: getattr(self.window, name) for name in values}
        try:
            for name, value in values.items():
                setattr(self.window, name, value)
            self.window.update_controls()
            yield
        finally:
            for name, value in previous.items():
                setattr(self.window, name, value)
            self.window.update_controls()

    def test_advanced_tab_hidden_by_default_and_controls_reparented_out_of_external_grid(self):
        window = self.window
        self.assertFalse(self.advanced_visible())
        self.assertFalse(window.advanced_settings_button.isChecked())
        self.assertTrue(window.advanced_settings_button.isCheckable())
        self.assertTrue(window.advanced_settings_button.isEnabled())
        self.assertEqual(window.advanced_settings_button.text(), "기타·고급 설정")
        for control in (window.external_mode, window.legacy_trigger_label, window.model_trigger):
            self.assertEqual(window.external_grid.indexOf(control), -1)
            self.assertTrue(window.advanced_mode_panel.isAncestorOf(control))
            self.assertGreaterEqual(window.advanced_mode_panel.layout().indexOf(control), 0)
        self.assertIs(window.tabs.widget(0), window.advanced_mode_panel)
        self.assertIn("수동 가격·이동평균", window.advanced_mode_note.text())
        self.assert_no_trading_io()

    def test_footer_opens_and_closes_the_same_tab_objects_without_duplicates(self):
        window = self.window
        pages = tuple(window.workspace_tabs.widget(index) for index in range(window.workspace_tabs.count()))
        settings = tuple(window.tabs.widget(index) for index in range(window.tabs.count()))
        for _ in range(3):
            window.advanced_settings_button.click()
            self.assertTrue(self.advanced_visible())
            self.assertIs(window.workspace_tabs.currentWidget(), window.tabs)
            self.assertEqual(window.advanced_settings_button.text(), "고급 설정 닫기")
            window.advanced_settings_button.click()
            self.assertFalse(self.advanced_visible())
            self.assertIs(window.workspace_tabs.currentWidget(), window.signal_connection_page)
            self.assertEqual(window.advanced_settings_button.text(), "기타·고급 설정")
            self.assertEqual(tuple(window.workspace_tabs.widget(index) for index in range(window.workspace_tabs.count())), pages)
            self.assertEqual(tuple(window.tabs.widget(index) for index in range(window.tabs.count())), settings)
        self.assert_no_trading_io()

    def test_navigation_preserves_both_models_builtin_limits_policy_and_order_permission(self):
        window = self.window
        self.select_models(MARK1_TRIGGER, MARK11_TRIGGER)
        window.model_trigger.setCurrentIndex(window.model_trigger.findData("lstm30"))
        self.drain_activity()
        window.external_quantity.setValue(7)
        window.external_krw.setValue(123456)
        window.external_usd.setValue(4321)
        window.buy_percent.setValue(12.5)
        window.configure_external()
        state = self.state()
        sources = dict(window.engine.external_sources)
        feeds = dict(window._prototype_feeds)
        validators = dict(window.engine.source_validators)
        with patch.object(window.engine, "enable_orders", wraps=window.engine.enable_orders) as arm, \
                patch.object(window.engine, "disarm", wraps=window.engine.disarm) as disarm, \
                patch.object(window, "configure_external", wraps=window.configure_external) as configure:
            window.advanced_settings_button.click()
            window.advanced_settings_button.click()
            self.assertEqual(self.state(), state)
            self.assertEqual(window.engine.external_sources, sources)
            self.assertEqual(window._prototype_feeds, feeds)
            self.assertEqual(window.engine.source_validators, validators)
            arm.assert_not_called()
            disarm.assert_not_called()
            configure.assert_not_called()
        self.assertFalse(window.engine.orders_enabled)
        self.assert_no_trading_io()

    def test_all_models_off_keeps_external_mode_on_until_user_explicitly_selects_manual(self):
        window = self.window
        self.select_models(MARK1_TRIGGER, MARK11_TRIGGER)
        self.assertTrue(window.external_mode.isChecked())
        self.select_models()
        self.assertEqual(window._chosen_external_models(), ())
        self.assertEqual(window._chosen_trigger(), "none")
        self.assertTrue(window.external_mode.isChecked())
        self.assertTrue(window.external_mode.isEnabled())
        self.assertNotIn("수동 규칙", window.connection_summary.text())
        self.assertFalse(window.engine.orders_enabled)
        self.assert_no_trading_io()

    def test_manual_mode_is_explicit_and_summary_remains_visible_when_advanced_hidden(self):
        window = self.window
        window.advanced_settings_button.click()
        self.assertTrue(window.external_mode.isEnabled())
        self.assertIn("해제하면 수동 규칙", window.external_mode.text())
        window.external_mode.click()
        self.assertFalse(window.external_mode.isChecked())
        self.assertIn("수동 규칙", window.connection_summary.text())
        window.advanced_settings_button.click()
        self.assertFalse(self.advanced_visible())
        self.assertIn("수동 규칙", window.connection_summary.text())
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.monitoring)
        self.assert_no_trading_io()

    def test_selecting_prototype_requires_external_mode_and_disables_its_manual_switch(self):
        window = self.window
        window.advanced_settings_button.click()
        window.external_mode.click()
        self.assertFalse(window.external_mode.isChecked())
        for model in (MARK1_TRIGGER, MARK11_TRIGGER):
            with self.subTest(model=model):
                self.select_models(model)
                self.assertTrue(window.external_mode.isChecked())
                self.assertFalse(window.external_mode.isEnabled())
                before = self.state()
                window.external_mode.click()
                self.assertEqual(self.state(), before)
        self.assert_no_trading_io()

    def test_builtin_selection_disables_manual_switch_and_turning_it_off_preserves_external_mode(self):
        window = self.window
        window.model_trigger.setCurrentIndex(window.model_trigger.findData("lstm30"))
        self.drain_activity()
        self.assertTrue(window.external_mode.isChecked())
        self.assertFalse(window.external_mode.isEnabled())
        window.model_trigger.setCurrentIndex(window.model_trigger.findData("none"))
        self.drain_activity()
        self.assertTrue(window.external_mode.isChecked())
        self.assertTrue(window.external_mode.isEnabled())
        self.assert_no_trading_io()

    def test_all_prototypes_off_reenables_real_selector_without_switching_environment(self):
        window = self.window
        self.select_models(MARK1_TRIGGER, MARK11_TRIGGER)
        self.assertFalse(window.environment_selector.buttons[TradingMode.REAL].isEnabled())
        self.select_models(MARK11_TRIGGER)
        self.assertFalse(window.environment_selector.buttons[TradingMode.REAL].isEnabled())
        with patch.object(window, "request_environment", wraps=window.request_environment) as switch:
            self.select_models()
            self.assertTrue(window.environment_selector.buttons[TradingMode.REAL].isEnabled())
            switch.assert_not_called()
        self.assertIs(window.service, self.service)
        self.assertIs(window.store, self.store)
        self.assertIs(self.service.mode, TradingMode.DEMO)
        self.assertIs(self.store.mode, TradingMode.DEMO)
        self.assertFalse(window.engine.orders_enabled)
        self.assert_no_trading_io()

    def test_advanced_controls_inherit_monitoring_worker_pending_and_confirmation_locks(self):
        window = self.window
        window.advanced_settings_button.click()
        conditions = ({"monitoring": True}, {"worker": SimpleNamespace()}, {"pending_auto_arm": True},
                      {"_workspace_worker": SimpleNamespace()}, {"_confirming_orders": True},
                      {"_pending_environment": (self.service, self.store)}, {"_confirming_environment": True})
        for condition in conditions:
            with self.subTest(condition=tuple(condition)), self.blocked_state(**condition):
                self.assertFalse(window.external_panel.isEnabled())
                self.assertFalse(window.advanced_mode_panel.isEnabled())
                self.assertFalse(window.external_mode.isEnabled())
                self.assertFalse(window.model_trigger.isEnabled())
                self.assertFalse(window.rule_panel.isEnabled())
                self.assertTrue(window.advanced_settings_button.isEnabled())
                before = self.state()
                window.external_mode.click()
                self.assertEqual(self.state(), before)
        self.assertTrue(window.advanced_mode_panel.isEnabled())
        self.assertTrue(window.external_mode.isEnabled())
        self.assert_no_trading_io()

    def test_opening_while_monitoring_is_readonly_and_does_not_disarm_or_stop(self):
        window = self.window
        self.select_models(MARK1_TRIGGER)
        # Simulate an already-authorized fake in-memory session. No polling,
        # account reads or order worker is started by this regression test.
        window.engine._armed.set()
        window.engine._stop.clear()
        try:
            with self.blocked_state(monitoring=True), \
                    patch.object(window.engine, "disarm", wraps=window.engine.disarm) as disarm, \
                    patch.object(window, "stop_monitoring", wraps=window.stop_monitoring) as stop:
                before = self.state()
                window.advanced_settings_button.click()
                self.assertTrue(self.advanced_visible())
                self.assertFalse(window.advanced_mode_panel.isEnabled())
                window.advanced_settings_button.click()
                self.assertFalse(self.advanced_visible())
                self.assertEqual(self.state(), before)
                self.assertTrue(window.engine.orders_enabled)
                self.assertTrue(window.monitoring)
                disarm.assert_not_called()
                stop.assert_not_called()
        finally:
            window.engine._armed.clear()
        self.assert_no_trading_io()


if __name__ == "__main__":
    unittest.main()
