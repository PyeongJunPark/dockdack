"""Offline concurrent external prototype controls in the normal desktop."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.v00_app import (DesktopModelBridge, ExternalFeedGroup, MARK1_TRIGGER,
                                 MARK11_TRIGGER, MARK12_TRIGGER, V00Window)

from dockdack.models import TradingMode
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW


class FakeExternalFeed:
    def __init__(self, window, model, policy, path, *, bundle_root=None):
        self.source_id, self.model, self.path = policy.source_id, model, path
        self.bundle_root, self.policy = bundle_root, policy
        self.status, self.diagnostics = model + ' · 외부 연결 대기', {}
        self.close = Mock()
        self.publish = Mock()
        self.validate_execution = Mock()


@unittest.skipUnless(HAS_QT, 'Install the gui extra')
class V00Mark1TriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.service = FakeTradingService()
        self.store = WatchStore(self.folder / 'ledger.sqlite3')
        self.store.save_item(WatchItem(self.service.resolve('005930'), '삼성전자'))
        self.keys = patch('dockdack.gui_service.KiwoomConfig.from_env',
                          side_effect=AssertionError('No real credentials in offline tests'))
        self.network = patch('requests.sessions.Session.request',
                             side_effect=AssertionError('No network in offline tests'))
        self.keys.start()
        self.network.start()
        self.addCleanup(self.keys.stop)
        self.addCleanup(self.network.stop)
        self.window = V00Window(self.service, self.store, builtin=False, external_feed_factory=FakeExternalFeed)
        self.window.engine.clock = lambda: NOW
        for timer in self.window.findChildren(QTimer):
            timer.stop()
        self.drain_activity()

    def drain_activity(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            if self.window._activity_worker is None and self.window._schedule_probe is None and self.window._workspace_worker is None:
                return
            time.sleep(.01)  # Release the GIL for numerical/worker imports.
        self.fail('Offline activity worker did not finish')

    def tearDown(self):
        self.window.stop_monitoring()
        self.drain_activity()
        self.window.close()
        for name in ('pool', 'inspection_pool', 'activity_pool'):
            getattr(self.window, name).waitForDone(5000)
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def select(self, name):
        for model, check in self.window.external_model_checks.items():
            check.setChecked(model == name)
        builtin = name if name in ('none', 'lstm30') else 'none'
        self.window.model_trigger.setCurrentIndex(self.window.model_trigger.findData(builtin))
        self.drain_activity()

    def test_selector_is_inline_in_existing_normal_window_and_starts_off(self):
        count = self.window.tabs.count()
        limits = (self.window.external_quantity.value(), self.window.external_krw.value(),
                  self.window.external_usd.value(), self.window.buy_percent.value())
        self.select(MARK1_TRIGGER)
        self.assertIs(type(self.window), V00Window)
        self.assertEqual(self.window.tabs.count(), count)
        self.assertEqual(self.window.model_trigger.count(), 2)
        self.assertEqual(self.window.model_trigger.findData(MARK1_TRIGGER), -1)
        self.assertEqual(self.window.model_trigger.findData(MARK11_TRIGGER), -1)
        self.assertEqual(self.window.model_trigger.findData(MARK12_TRIGGER), -1)
        notice_row = self.window.external_grid.getItemPosition(self.window.external_grid.indexOf(self.window.model_notice))[0]
        sources_row = self.window.external_grid.getItemPosition(self.window.external_grid.indexOf(self.window.additional_sources))[0]
        self.assertEqual(self.window.external_grid.indexOf(self.window.model_trigger), -1)
        self.assertTrue(self.window.advanced_mode_panel.isAncestorOf(self.window.model_trigger))
        self.assertFalse(self.window.workspace_tabs.isTabVisible(self.window.workspace_tabs.indexOf(self.window.tabs)))
        self.assertLess(notice_row, sources_row)
        self.assertFalse(self.window.builtin_lstm.isVisible())
        self.assertEqual(self.window.external_source.text(), 'external-model')
        self.assertIn('−0.9%', self.window.model_notice.text())
        self.assertIn('연구 검증 미통과', self.window.model_notice.text())
        self.assertIn('0건', self.window.model_notice.text())
        self.assertEqual(self.store.items()[0].days, 31)
        self.assertEqual(limits, (self.window.external_quantity.value(), self.window.external_krw.value(),
                                self.window.external_usd.value(), self.window.buy_percent.value()))
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertEqual((self.service.quote_calls, self.service.history_calls), (0, 0))
        self.assertFalse(self.service.submitted)

    def test_mark1_configures_regular_bridge_and_final_execution_validator_lazily(self):
        from dockdack.mark1_trigger import SOURCE_ID
        self.select(MARK1_TRIGGER)
        with patch.object(self.window.engine, 'configure_source_validators',
                          wraps=self.window.engine.configure_source_validators) as register:
            self.window.configure_external()
        self.assertIsInstance(self.window.test_producer, ExternalFeedGroup)
        bridge = self.window._prototype_feeds[MARK1_TRIGGER]
        self.assertIsInstance(bridge, FakeExternalFeed)
        register.assert_called_with({SOURCE_ID: bridge.validate_execution})
        self.assertEqual(set(self.window.engine.external_sources), {'external-model', SOURCE_ID})
        bridge.publish.assert_not_called()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertFalse(self.service.submitted)

    def test_old_constructor_checkbox_and_predictor_contract_still_works(self):
        predictors = {'domestic': Mock()}
        self.window._model_predictors = predictors
        self.window.builtin_lstm.setChecked(True)
        self.assertEqual(self.window._chosen_trigger(), 'lstm30')
        self.window.configure_external()
        self.assertIsInstance(self.window.test_producer, DesktopModelBridge)
        self.assertIs(self.window.test_producer.predictors, predictors)
        self.window.builtin_lstm.setChecked(False)
        self.assertEqual(self.window._chosen_trigger(), 'none')
        self.assertIsNone(self.window.test_producer)
        self.assertFalse(self.service.submitted)

    def test_random_generator_cannot_be_mixed_with_mark1(self):
        self.select(MARK1_TRIGGER)
        self.window.random_demo.setChecked(True)
        with self.assertRaisesRegex(ValueError, '하나만'):
            self.window.configure_external()
        with self.assertRaisesRegex(ValueError, '하나만'):
            self.window.confirm_automation()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_real_environment_switch_is_blocked_even_when_called_directly(self):
        self.select(MARK1_TRIGGER)
        self.assertFalse(self.window.environment_selector.buttons[TradingMode.REAL].isEnabled())
        with patch('dockdack.watch_gui.confirm_environment') as confirm:
            self.window.request_environment(TradingMode.REAL)
        confirm.assert_not_called()
        self.assertIs(self.window.service.mode, TradingMode.DEMO)
        self.assertIsNone(self.window._pending_environment)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_real_service_tampering_is_rejected_at_configuration_and_confirmation(self):
        self.select(MARK1_TRIGGER)
        self.service.mode = TradingMode.REAL
        try:
            with self.assertRaisesRegex(ValueError, '모의투자 전용'):
                self.window.configure_external()
            with self.assertRaisesRegex(ValueError, '모의투자 전용'):
                self.window.confirm_automation()
        finally:
            self.service.mode = TradingMode.DEMO
        self.assertFalse(self.service.submitted)

    def test_selecting_mark1_in_real_mode_is_rejected_without_mode_switch(self):
        self.service.mode = TradingMode.REAL
        try:
            self.select(MARK1_TRIGGER)
            self.assertEqual(self.window._chosen_trigger(), 'none')
            self.assertIs(self.window.service.mode, TradingMode.REAL)
            self.assertFalse(self.window.engine.orders_enabled)
        finally:
            self.service.mode = TradingMode.DEMO
        self.assertFalse(self.service.submitted)

    def test_switching_trigger_disarms_and_cancels_pending_activation(self):
        self.window.monitoring = True
        self.window.pending_auto_arm = self.window._manual_arm_pending = True
        self.select(MARK1_TRIGGER)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window._manual_arm_pending)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertTrue(self.window.engine._stop.is_set())
        self.assertFalse(self.service.submitted)

    def test_pending_real_environment_cannot_complete_after_mark1_selection(self):
        candidate = FakeTradingService()
        candidate.mode = TradingMode.REAL
        self.window._pending_environment = (candidate, self.store)
        self.select(MARK1_TRIGGER)
        self.window._finish_environment_switch()
        self.assertIs(self.window.service, self.service)
        self.assertIsNone(self.window._pending_environment)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_normal_confirmation_exposes_mark1_warning_without_enabling_orders(self):
        from PySide6.QtWidgets import QMessageBox
        self.select(MARK1_TRIGGER)
        self.assertIn('50% 초과', self.window.builtin_confirmation_notice)
        self.assertIn('연구 검증 미통과', self.window.builtin_confirmation_notice)
        with patch('dockdack.watch_gui.QMessageBox.question', return_value=QMessageBox.StandardButton.No) as confirm:
            self.assertFalse(self.window.confirm_automation())
        self.assertIn('연구 검증 미통과', confirm.call_args.args[2])
        self.assertIn('−0.9%', confirm.call_args.args[2])
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.service.submitted)

    def test_explicit_mark1_constructor_rejects_real_service_before_gui_setup(self):
        service = FakeTradingService()
        service.mode = TradingMode.REAL
        with self.assertRaisesRegex(ValueError, '모의투자에서만'):
            V00Window(service, self.store, trigger=MARK1_TRIGGER)
        self.assertFalse(service.submitted)

    def test_unknown_trigger_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '알 수 없는'):
            V00Window(self.service, self.store, trigger='unknown')

    def test_mark11_selector_shows_distinct_source_target_and_risk_without_auto_start(self):
        count = self.window.tabs.count()
        self.select(MARK11_TRIGGER)
        self.assertEqual(self.window.tabs.count(), count)
        self.assertIn('mark1.1 prototype', self.window.external_model_checks[MARK11_TRIGGER].text())
        self.assertIn('mark1.1 prototype', self.window.model_status.text())
        self.assertEqual(self.window.external_source.text(), 'external-model')
        self.assertIn('+0.5%', self.window.model_notice.text())
        self.assertIn('−0.4%', self.window.model_notice.text())
        self.assertIn('50% 초과', self.window.model_notice.text())
        self.assertIn('123건', self.window.model_notice.text())
        self.assertIn('비용 반영 손실', self.window.model_notice.text())
        self.assertIn('장중 진입 성과 미검증', self.window.model_notice.text())
        self.assertNotIn('신호 0건', self.window.model_notice.text())
        self.assertEqual(self.store.items()[0].days, 31)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.service.submitted)

    def test_mark11_bridge_and_final_validator_are_lazy_and_model_specific(self):
        from dockdack.mark1_1_trigger import SOURCE_ID
        bundle = self.folder / 'saved-mark11'
        self.window._mark11_bundle = bundle
        self.select(MARK11_TRIGGER)
        with patch.object(self.window.engine, 'configure_source_validators',
                          wraps=self.window.engine.configure_source_validators) as register:
            self.window.configure_external()
        bridge = self.window._prototype_feeds[MARK11_TRIGGER]
        self.assertIsInstance(bridge, FakeExternalFeed)
        self.assertEqual(bridge.bundle_root, bundle)
        register.assert_called_with({SOURCE_ID: bridge.validate_execution})
        self.assertEqual(set(self.window.engine.external_sources), {'external-model', SOURCE_ID})
        self.assertEqual(set(self.window.engine.source_validators), {SOURCE_ID})
        bridge.publish.assert_not_called()
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertFalse(self.service.submitted)

    def test_switching_between_prototypes_clears_prior_model_callback_and_identity(self):
        for old, new, source in ((MARK1_TRIGGER, MARK11_TRIGGER, 'mark1-1-prototype-demo-trigger'),
                                 (MARK11_TRIGGER, MARK1_TRIGGER, 'mark1-prototype-demo-trigger')):
            with self.subTest(old=old, new=new):
                self.select(old)
                self.window.configure_external()
                self.assertEqual(len(self.window.engine.source_validators), 1)
                self.window.monitoring = True
                self.window.pending_auto_arm = self.window._manual_arm_pending = True
                self.select(new)
                self.assertEqual(self.window.engine.source_validators, {})
                self.assertIsNone(self.window.test_producer)
                self.assertEqual(self.window.external_source.text(), 'external-model')
                self.assertFalse(self.window.monitoring)
                self.assertFalse(self.window.pending_auto_arm)
                self.assertFalse(self.window._manual_arm_pending)
                self.assertFalse(self.window.engine.orders_enabled)
        self.select('none')
        self.assertEqual(self.window.external_source.text(), 'external-model')
        self.assertEqual(self.window.engine.source_validators, {})
        self.assertFalse(self.service.submitted)

    def test_mark11_confirmation_exposes_new_targets_and_never_arms_itself(self):
        from PySide6.QtWidgets import QMessageBox
        self.select(MARK11_TRIGGER)
        with patch('dockdack.watch_gui.QMessageBox.question', return_value=QMessageBox.StandardButton.No) as confirm:
            self.assertFalse(self.window.confirm_automation())
        notice = confirm.call_args.args[2]
        self.assertIn('mark1.1 prototype', notice)
        self.assertIn('+0.5%', notice)
        self.assertIn('−0.4%', notice)
        self.assertIn('연구 검증 미통과', notice)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.pending_auto_arm)
        self.assertFalse(self.service.submitted)

    def test_mark11_real_mode_and_random_generator_are_blocked(self):
        self.select(MARK11_TRIGGER)
        self.assertFalse(self.window.environment_selector.buttons[TradingMode.REAL].isEnabled())
        with patch('dockdack.watch_gui.confirm_environment') as confirm:
            self.window.request_environment(TradingMode.REAL)
        confirm.assert_not_called()
        self.window.random_demo.setChecked(True)
        with self.assertRaisesRegex(ValueError, '하나만'):
            self.window.configure_external()
        self.window.random_demo.setChecked(False)
        self.service.mode = TradingMode.REAL
        try:
            with self.assertRaisesRegex(ValueError, '모의투자 전용'):
                self.window.configure_external()
            with self.assertRaisesRegex(ValueError, '모의투자 전용'):
                self.window.confirm_automation()
            with self.assertRaisesRegex(ValueError, '모의투자에서만'):
                V00Window(self.service, self.store, trigger=MARK11_TRIGGER)
            self.select('none')
            self.select(MARK11_TRIGGER)
            self.assertEqual(self.window._chosen_trigger(), 'none')
            self.assertIs(self.window.service.mode, TradingMode.REAL)
        finally:
            self.service.mode = TradingMode.DEMO
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_pending_real_environment_cannot_complete_after_mark11_selection(self):
        candidate = FakeTradingService()
        candidate.mode = TradingMode.REAL
        self.window._pending_environment = (candidate, self.store)
        self.select(MARK11_TRIGGER)
        self.window._finish_environment_switch()
        self.assertIs(self.window.service, self.service)
        self.assertIsNone(self.window._pending_environment)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_both_external_models_can_be_enabled_with_separate_files_and_validators(self):
        for model in (MARK1_TRIGGER, MARK11_TRIGGER):
            self.window.external_model_checks[model].setChecked(True)
        self.window.configure_external()
        self.assertEqual(set(self.window._prototype_feeds), {MARK1_TRIGGER, MARK11_TRIGGER})
        feeds = list(self.window._prototype_feeds.values())
        self.assertNotEqual(feeds[0].path, feeds[1].path)
        self.assertEqual(set(self.window.engine.source_validators), {feed.source_id for feed in feeds})
        self.assertEqual(self.window._chosen_trigger(), 'none')
        self.window.test_producer.publish({'offline': True})
        for feed in feeds:
            feed.publish.assert_called_once_with({'offline': True})
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_disabled_feed_closes_and_cannot_leave_active_validator(self):
        for model in (MARK1_TRIGGER, MARK11_TRIGGER):
            self.window.external_model_checks[model].setChecked(True)
        self.window.configure_external()
        feeds = list(self.window._prototype_feeds.values())
        self.window.external_model_checks[MARK1_TRIGGER].setChecked(False)
        for feed in feeds:
            feed.close.assert_called_once()
        self.assertEqual(self.window.engine.source_validators, {})
        self.window.configure_external()
        self.assertEqual(set(self.window._prototype_feeds), {MARK11_TRIGGER})
        self.assertEqual(set(self.window.engine.source_validators), {'mark1-1-prototype-demo-trigger'})

    def test_third_research_model_is_opt_in_and_all_three_remain_separate_and_off(self):
        self.assertFalse(self.window.external_model_checks[MARK12_TRIGGER].isChecked())
        self.select(MARK12_TRIGGER)
        self.assertEqual(self.window._chosen_external_models(), (MARK12_TRIGGER,))
        self.assertIn('연구 검증 미통과', self.window.model_notice.text())
        self.assertIn('−6.90%', self.window.model_notice.text())
        self.window.configure_external()
        feed = self.window._prototype_feeds[MARK12_TRIGGER]
        self.assertEqual(feed.source_id, 'mark1-2-prototype-demo-trigger')
        self.assertEqual(set(self.window.engine.source_validators), {feed.source_id})
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        for check in self.window.external_model_checks.values():
            check.setChecked(True)
        self.window.configure_external()
        feeds = list(self.window._prototype_feeds.values())
        self.assertEqual(len(feeds), 3)
        self.assertEqual(len({feed.source_id for feed in feeds}), 3)
        self.assertEqual(len({feed.path for feed in feeds}), 3)
        self.assertEqual(set(self.window.engine.source_validators), {feed.source_id for feed in feeds})
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_duplicate_managed_source_or_path_rejected_without_arming(self):
        self.select(MARK1_TRIGGER)
        self.window.additional_sources.add_row(source='mark1-prototype-demo-trigger', path=str(self.folder / 'other.json'))
        with self.assertRaisesRegex(ValueError, '중복'):
            self.window.configure_external()
        self.assertEqual(self.window.engine.source_validators, {})
        self.assertFalse(self.window.engine.orders_enabled)

    def test_legacy_prototype_cli_migrates_to_external_checkbox(self):
        migrated = V00Window(self.service, self.store, trigger=MARK11_TRIGGER,
                             external_feed_factory=FakeExternalFeed)
        try:
            self.assertEqual(migrated._chosen_trigger(), 'none')
            self.assertEqual(migrated._chosen_external_models(), (MARK11_TRIGGER,))
            self.assertFalse(migrated.engine.orders_enabled)
        finally:
            migrated.close()
            for name in ('pool', 'inspection_pool', 'activity_pool'):
                getattr(migrated, name).waitForDone(10000)
            self.app.processEvents()
            migrated.close()


if __name__ == '__main__':
    unittest.main()
