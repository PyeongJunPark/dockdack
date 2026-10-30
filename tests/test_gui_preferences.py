"""Desktop choices survive restart without reviving a trading session."""
from __future__ import annotations

import os
import importlib.util
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.ui.v00_app import (MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER,
                                    MARK14_TRIGGER, ALL_MODEL_IDS, INTRADAY_MODEL_IDS,
                                    TARGET_HORIZON_MODEL_IDS, PREOPEN_MODEL_IDS,
                                    MINUTE_TRANSFER_IDS, MINUTE_HEDGE_IDS,
                                    PROTOTYPE_SOURCES, WATCH_MODEL_IDS, V00Window)

from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService


@unittest.skipUnless(HAS_QT, 'Install the gui extra')
class GuiPreferencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.service = FakeTradingService()
        self.store = WatchStore(self.folder / 'first' / 'watchlist.sqlite3')
        self.keys = patch('dockdack.gui_service.KiwoomConfig.from_env',
                          side_effect=AssertionError('desktop settings must not read credentials'))
        self.keys.start()
        self.addCleanup(self.keys.stop)
        self.windows = []
        self.addCleanup(self.close_windows)

    def make_window(self, store=None, **kwargs):
        window = V00Window(self.service, store or self.store, builtin=False, **kwargs)
        self.windows.append(window)
        for timer in window.findChildren(QTimer):
            timer.stop()
        self.wait_idle(window)
        return window

    def wait_idle(self, window):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for name in ('pool', 'inspection_pool', 'activity_pool'):
                getattr(window, name).waitForDone(250)
            self.app.processEvents()
            if all(getattr(window, name) is None for name in
                   ('worker', '_inspection_worker', '_workspace_worker', '_activity_worker', '_schedule_probe')):
                return
        self.fail('GUI background reader did not stop')

    def close_windows(self):
        for window in reversed(self.windows):
            window.close()
            self.wait_idle(window)
            window.close()
            window.deleteLater()
        self.app.processEvents()

    def test_settings_survive_restart_but_monitoring_and_orders_remain_off(self):
        first = self.make_window()
        first.interval.setValue(87)
        first.hourly_ranking.setChecked(False)
        first.exchange_input.setCurrentIndex(first.exchange_input.findData('ND'))
        first.external_model_checks[MARK1_TRIGGER].setChecked(True)
        first.external_source.setText('my-signal-source')
        first.signal_path.setText(str(self.folder / 'custom-signals.json'))
        first.chart_path.setText(str(self.folder / 'custom-charts.json'))
        first.external_quantity.setValue(25)
        first.external_krw.setValue(250_000)
        first.external_usd.setValue(250)
        first.buy_percent.setValue(7.5)
        first.mark14_panel.model_buy_percents[MARK1_TRIGGER].setValue(4.25)
        first.order_popups.setChecked(False)
        first.additional_sources.add_row(source='second', path=str(self.folder / 'second.json'))
        first.watch_market_tabs.setCurrentIndex(1)
        first.workspace_tabs.setCurrentWidget(first.tabs)
        first.workspace_tabs.setCurrentWidget(first.watch_page)
        first.close()

        stored = self.store.load_ui_preferences()
        self.assertEqual(stored['interval'], 87)
        self.assertNotIn('monitoring', stored)
        self.assertNotIn('orders_enabled', stored)
        self.assertNotIn('trading_mode', stored)

        second = self.make_window()
        self.assertEqual(second.interval.value(), 87)
        self.assertFalse(second.hourly_ranking.isChecked())
        self.assertEqual(second.exchange_input.currentData(), 'ND')
        self.assertTrue(second.external_model_checks[MARK1_TRIGGER].isChecked())
        self.assertEqual(second.external_source.text(), 'my-signal-source')
        self.assertEqual(second.signal_path.text(), str(self.folder / 'custom-signals.json'))
        self.assertEqual(second.chart_path.text(), str(self.folder / 'custom-charts.json'))
        self.assertEqual(second.external_quantity.value(), 25)
        self.assertEqual(second.external_krw.value(), 250_000)
        self.assertEqual(second.external_usd.value(), 250)
        self.assertEqual(second.buy_percent.value(), 7.5)
        self.assertEqual(second.mark14_panel.model_buy_percents[MARK1_TRIGGER].value(), 4.25)
        self.assertEqual(second.engine.source_buy_percents['mark1-prototype-demo-trigger'],
                         Decimal('4.25'))
        self.assertFalse(second.order_popups.isChecked())
        self.assertNotIn('close_all_at_market_end', stored)
        self.assertFalse(hasattr(second.engine, "close_liquidator"))
        self.assertEqual(second.additional_sources.raw_sources(), (('second', str(self.folder / 'second.json')),))
        self.assertEqual(second.watch_market_tabs.currentIndex(), 1)
        self.assertIs(second.workspace_tabs.currentWidget(), second.watch_page)
        self.assertTrue(second.workspace_tabs.isTabVisible(second.workspace_tabs.indexOf(second.tabs)))
        self.assertTrue(second.advanced_settings_button.isHidden())
        self.assertFalse(second.monitoring)
        self.assertFalse(second.pending_auto_arm)
        self.assertFalse(second.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_item_lookback_is_stored_with_watch_item_not_global_preferences(self):
        item = WatchItem(self.service.resolve('005930'), '삼성전자', 64)
        self.store.save_item(item)
        first = self.make_window()
        first.close()
        self.assertNotIn('days_input', self.store.load_ui_preferences())
        second = self.make_window()
        second.reload_tables(items=self.store.items())
        self.assertEqual(second.selected_item().days, 64)

    def test_legacy_common_percent_seeds_each_model_once(self):
        self.store.save_ui_preferences({'version': 3, 'buy_percent': 7.5,
                                        'external_models': [MARK1_TRIGGER]})
        window = self.make_window()
        self.assertEqual(window.buy_percent.value(), 7.5)
        self.assertTrue(all(field.value() == (1 if model in MINUTE_TRANSFER_IDS else 7.5)
                            for model, field in window.mark14_panel.model_buy_percents.items()))
        self.assertEqual(window.engine.source_buy_percents['mark1-prototype-demo-trigger'],
                         Decimal('7.5'))
        self.assertTrue(all(not window.external_model_checks[model].isChecked()
                            for model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS)))
        self.assertTrue(all(model in window.mark14_panel.model_buy_percents
                            for model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS)))
        for view in window.watch_tables.values():
            self.assertFalse(view.isColumnHidden(1 + WATCH_MODEL_IDS.index(MARK1_TRIGGER)))
            self.assertTrue(all(view.isColumnHidden(1 + WATCH_MODEL_IDS.index(model))
                                for model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS)))
            self.assertFalse(view.isColumnHidden(view.columnCount() - 2))
            self.assertFalse(view.isColumnHidden(view.columnCount() - 1))
        self.assertFalse(window.engine.orders_enabled)

    def test_corrupt_model_percent_falls_back_to_valid_common_value(self):
        self.store.save_ui_preferences({'version': 4, 'buy_percent': 8,
                                        'model_buy_percents': {MARK1_TRIGGER: -2,
                                                               MARK11_TRIGGER: '99',
                                                               MARK12_TRIGGER: 5.5}})
        window = self.make_window()
        self.assertEqual(window.mark14_panel.model_buy_percents[MARK1_TRIGGER].value(), 8)
        self.assertEqual(window.mark14_panel.model_buy_percents[MARK11_TRIGGER].value(), 8)
        self.assertEqual(window.mark14_panel.model_buy_percents[MARK12_TRIGGER].value(), 5.5)
        self.assertFalse(window.engine.orders_enabled)

    def test_every_model_keeps_its_own_percent_after_restart(self):
        first = self.make_window()
        for index, model in enumerate(ALL_MODEL_IDS, 1):
            first.mark14_panel.model_buy_percents[model].setValue(index)
        first.close()
        second = self.make_window()
        self.assertEqual(set(second.mark14_panel.model_buy_percents), set(ALL_MODEL_IDS))
        for index, model in enumerate(ALL_MODEL_IDS, 1):
            self.assertEqual(second.mark14_panel.model_buy_percents[model].value(), index)
            if model in MINUTE_HEDGE_IDS:
                self.assertNotIn(PROTOTYPE_SOURCES[model], second.engine.source_buy_percents)
            else:
                self.assertEqual(second.engine.source_buy_percents[PROTOTYPE_SOURCES[model]], Decimal(index))
        self.assertFalse(second.engine.orders_enabled)

    def test_bulk_model_selection_is_saved_and_restored_without_arming(self):
        first = self.make_window()
        first.mark14_panel.select_all_button.click()
        first.close()
        self.assertEqual(set(self.store.load_ui_preferences()['external_models']), set(ALL_MODEL_IDS))
        second = self.make_window()
        self.assertEqual(set(second._chosen_external_models()), set(ALL_MODEL_IDS))
        self.assertEqual(second.mark14_panel.selection_count.text(),
                         f'{len(ALL_MODEL_IDS)} / {len(ALL_MODEL_IDS)}개 선택')
        second.mark14_panel.clear_all_button.click()
        second.close()
        third = self.make_window()
        self.assertEqual(third._chosen_external_models(), ())
        self.assertFalse(third.monitoring)
        self.assertFalse(third.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_bulk_buy_percent_survives_restart_and_preserves_model_choices(self):
        first = self.make_window()
        first.mark14_panel.clear_all_button.click()
        first.external_model_checks[MARK1_TRIGGER].setChecked(True)
        self.wait_idle(first)
        research_percents = {
            model: first.mark14_panel.model_buy_percents[model].value()
            for model in MINUTE_HEDGE_IDS
        }
        first.mark14_panel.bulk_buy_percent.setValue(2.75)
        first.mark14_panel.apply_bulk_buy_percent_button.click()
        first.close()

        stored = self.store.load_ui_preferences()
        self.assertEqual(stored['external_models'], [MARK1_TRIGGER])
        for model in ALL_MODEL_IDS:
            expected = research_percents[model] if model in MINUTE_HEDGE_IDS else 2.75
            self.assertEqual(stored['model_buy_percents'][model], expected)

        second = self.make_window()
        self.assertEqual(second.mark14_panel.bulk_buy_percent.value(), 2.75)
        self.assertEqual(second._chosen_external_models(), (MARK1_TRIGGER,))
        for model in ALL_MODEL_IDS:
            expected = research_percents[model] if model in MINUTE_HEDGE_IDS else 2.75
            self.assertEqual(second.mark14_panel.model_buy_percents[model].value(), expected)
            if model in MINUTE_HEDGE_IDS:
                self.assertNotIn(PROTOTYPE_SOURCES[model], second.engine.source_buy_percents)
            else:
                self.assertEqual(second.engine.source_buy_percents[PROTOTYPE_SOURCES[model]],
                                 Decimal('2.75'))
        self.assertFalse(second.monitoring)
        self.assertFalse(second.pending_auto_arm)
        self.assertFalse(second.engine.orders_enabled)
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_bulk_target_without_applying_survives_restart_without_changing_model_sizes(self):
        first = self.make_window()
        original = {model: field.value()
                    for model, field in first.mark14_panel.model_buy_percents.items()}
        first.mark14_panel.bulk_buy_percent.setValue(4.25)
        first.close()
        second = self.make_window()
        self.assertEqual(second.mark14_panel.bulk_buy_percent.value(), 4.25)
        self.assertEqual({model: field.value()
                          for model, field in second.mark14_panel.model_buy_percents.items()}, original)
        self.assertFalse(second.engine.orders_enabled)
        self.assertFalse(second.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_unfinished_buy_percent_text_is_committed_before_window_close(self):
        from PySide6.QtTest import QTest
        first = self.make_window()
        field = first.mark14_panel.model_buy_percents[MARK1_TRIGGER]
        field.lineEdit().selectAll()
        QTest.keyClicks(field.lineEdit(), '7.25')
        self.assertEqual(field.value(), 10)
        self.assertTrue(field.lineEdit().text().startswith('7.25'))
        first.close()
        second = self.make_window()
        self.assertEqual(second.mark14_panel.model_buy_percents[MARK1_TRIGGER].value(), 7.25)
        self.assertEqual(second.engine.source_buy_percents[PROTOTYPE_SOURCES[MARK1_TRIGGER]],
                         Decimal('7.25'))
        self.assertFalse(second.engine.orders_enabled)
        self.assertFalse(second.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_unfinished_bulk_target_is_saved_on_close_without_applying(self):
        from PySide6.QtTest import QTest
        first = self.make_window()
        original = dict(first.engine.source_buy_percents)
        field = first.mark14_panel.bulk_buy_percent
        field.lineEdit().selectAll()
        QTest.keyClicks(field.lineEdit(), '6.25')
        self.assertEqual(field.value(), 1)
        first.close()
        second = self.make_window()
        self.assertEqual(second.mark14_panel.bulk_buy_percent.value(), 6.25)
        self.assertEqual(second.engine.source_buy_percents, original)
        self.assertFalse(second.engine.orders_enabled)
        self.assertFalse(second.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_typed_common_percent_survives_close_without_changing_models(self):
        from PySide6.QtTest import QTest
        first = self.make_window()
        original = dict(first.engine.source_buy_percents)
        first.buy_percent.lineEdit().selectAll()
        QTest.keyClicks(first.buy_percent.lineEdit(), '8.25')
        self.assertEqual(first.buy_percent.value(), 8.25)
        first.close()
        second = self.make_window()
        self.assertEqual(second.buy_percent.value(), 8.25)
        self.assertEqual(second.engine.equity_buy_percent, Decimal('8.25'))
        self.assertEqual(second.engine.source_buy_percents, original)
        self.assertFalse(second.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_environment_request_commits_percent_edits_before_saving(self):
        from PySide6.QtTest import QTest
        from dockdack.models import TradingMode
        window = self.make_window()
        fields = ((window.buy_percent, '8.25'),
                  (window.mark14_panel.bulk_buy_percent, '3.25'),
                  (window.mark14_panel.model_buy_percents[MARK1_TRIGGER], '6.25'))
        for field, text in fields:
            field.lineEdit().selectAll()
            QTest.keyClicks(field.lineEdit(), text)
        with patch('dockdack.watch_gui.WatchlistDialog.request_environment') as switch:
            window.request_environment(TradingMode.DEMO)
        switch.assert_called_once_with(TradingMode.DEMO)
        saved = self.store.load_ui_preferences()
        self.assertEqual(saved['buy_percent'], 8.25)
        self.assertEqual(saved['bulk_buy_percent'], 3.25)
        self.assertEqual(saved['model_buy_percents'][MARK1_TRIGGER], 6.25)
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_bulk_apply_commits_typed_target_without_enter(self):
        from PySide6.QtTest import QTest
        window = self.make_window()
        field = window.mark14_panel.bulk_buy_percent
        field.lineEdit().selectAll()
        QTest.keyClicks(field.lineEdit(), '3.75')
        self.assertEqual(field.value(), 1)
        with patch.object(window, '_queue_save_preferences',
                          wraps=window._queue_save_preferences) as save:
            window.mark14_panel.apply_bulk_buy_percent_button.click()
            save.assert_called_once_with()
        self.assertEqual(field.value(), 3.75)
        self.assertTrue(all(value == Decimal('3.75')
                            for value in window.engine.source_buy_percents.values()))
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_typed_bulk_noop_saves_target_once_without_reapplying_model_sizes(self):
        from PySide6.QtTest import QTest
        window = self.make_window()
        panel = window.mark14_panel
        panel.bulk_buy_percent.setValue(3.75)
        panel.apply_bulk_buy_percent_button.click()
        panel.bulk_buy_percent.setValue(1)
        panel.bulk_buy_percent.lineEdit().selectAll()
        QTest.keyClicks(panel.bulk_buy_percent.lineEdit(), '3.75')
        with patch.object(window, '_queue_save_preferences',
                          wraps=window._queue_save_preferences) as save, \
                patch.object(window, '_apply_execution_preferences') as sizing:
            panel.apply_bulk_buy_percent_button.click()
            save.assert_called_once_with()
            sizing.assert_not_called()
        self.assertEqual(panel.bulk_buy_percent.value(), 3.75)
        window.close()
        second = self.make_window()
        self.assertEqual(second.mark14_panel.bulk_buy_percent.value(), 3.75)
        self.assertTrue(all(value == Decimal('3.75')
                            for value in second.engine.source_buy_percents.values()))
        self.assertFalse(second.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_legacy_uniform_model_percents_restore_bulk_input_without_reapplying(self):
        model_sizing = {model: (9 if model in MINUTE_HEDGE_IDS else 2.75)
                        for model in ALL_MODEL_IDS}
        self.store.save_ui_preferences({'version': 4, 'model_buy_percents': model_sizing})
        window = self.make_window()
        self.assertEqual(window.mark14_panel.bulk_buy_percent.value(), 2.75)
        self.assertEqual({model: field.value()
                          for model, field in window.mark14_panel.model_buy_percents.items()}, model_sizing)
        self.assertFalse(window.engine.orders_enabled)
        self.assertFalse(window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_invalid_bulk_percent_uses_safe_uniform_fallback(self):
        model_sizing = {model: 2.75 for model in ALL_MODEL_IDS}
        for invalid in ('99', False, -1, 101, None):
            with self.subTest(invalid=invalid):
                self.store.save_ui_preferences({'version': 4, 'bulk_buy_percent': invalid,
                                                'model_buy_percents': model_sizing})
                window = self.make_window()
                self.assertEqual(window.mark14_panel.bulk_buy_percent.value(), 2.75)
                self.assertTrue(all(field.value() == 2.75
                                    for field in window.mark14_panel.model_buy_percents.values()))
                self.assertFalse(window.engine.orders_enabled)
                window.close()

    def test_bulk_input_is_store_scoped_when_switching_preferences(self):
        first = self.make_window()
        first.mark14_panel.bulk_buy_percent.setValue(4.25)
        first.mark14_panel.model_buy_percents[MARK1_TRIGGER].setValue(6.5)
        first._save_preferences()
        other = WatchStore(self.folder / 'other' / 'watchlist.sqlite3')
        first.store = other
        first._restore_preferences(after_switch=True)
        self.wait_idle(first)
        self.assertEqual(first.mark14_panel.bulk_buy_percent.value(), 1)
        self.assertEqual(first.mark14_panel.model_buy_percents[MARK1_TRIGGER].value(), 10)
        first.store = self.store
        first._restore_preferences(after_switch=True)
        self.wait_idle(first)
        self.assertEqual(first.mark14_panel.bulk_buy_percent.value(), 4.25)
        self.assertEqual(first.mark14_panel.model_buy_percents[MARK1_TRIGGER].value(), 6.5)
        self.assertFalse(first.engine.orders_enabled)
        self.assertFalse(first.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_old_log_tab_selection_opens_logs_inside_advanced_settings(self):
        for selection in ({'workspace_tab_id': 'logs'}, {'workspace_tab': 3}):
            with self.subTest(selection=selection):
                self.store.save_ui_preferences({'version': 3, 'advanced_visible': False, **selection})
                window = self.make_window()
                self.assertTrue(window.workspace_tabs.isTabVisible(window.workspace_tabs.indexOf(window.tabs)))
                self.assertIs(window.workspace_tabs.currentWidget(), window.tabs)
                self.assertIs(window.tabs.currentWidget(), window.operations_panel)
                window.close()

    def test_restored_no_model_does_not_upgrade_short_watch_history(self):
        self.store.save_item(WatchItem(self.service.resolve('005930'), '삼성전자', 20))
        self.store.save_ui_preferences({'version': 1, 'external_models': [], 'model_trigger': 'none'})
        window = self.make_window()
        for _ in range(20):
            window.pool.waitForDone(1000)
            self.app.processEvents()
            if window._workspace_worker is None:
                break
        self.assertEqual(window._chosen_external_models(), ())
        self.assertEqual(self.store.items()[0].days, 20)
        self.assertFalse(window.engine.orders_enabled)

    def test_explicit_launch_model_choices_override_saved_selection(self):
        self.store.save_ui_preferences({
            'version': 1, 'external_models': [MARK1_TRIGGER], 'model_trigger': 'lstm30',
            'interval': 87,
        })
        window = self.make_window(restore_model_choices=False)
        self.assertEqual(window._chosen_external_models(), ())
        self.assertEqual(window._chosen_trigger(), 'none')
        self.assertEqual(window.interval.value(), 87)
        self.assertFalse(window.engine.orders_enabled)

    def test_default_selected_model_can_extend_short_watch_history(self):
        self.store.save_item(WatchItem(self.service.resolve('005930'), '삼성전자', 30))
        window = self.make_window(external_models=(MARK1_TRIGGER,))
        self.wait_idle(window)
        self.assertEqual(self.store.items()[0].days, 31)

    def test_v2_model_choices_add_new_preopen_models_once_then_respect_v3_off(self):
        old_enabled = {MARK1_TRIGGER, MARK14_TRIGGER}
        old_disabled = {MARK11_TRIGGER, MARK12_TRIGGER}
        new_models = set(PREOPEN_MODEL_IDS) - {MARK14_TRIGGER}
        self.store.save_ui_preferences({
            'version': 2, 'external_models': sorted(old_enabled), 'model_trigger': 'none',
        })
        first = self.make_window()
        selected = set(first._chosen_external_models())
        self.assertTrue(old_enabled <= selected)
        self.assertFalse(old_disabled & selected)
        self.assertEqual(selected & new_models, new_models)
        self.assertFalse(first.engine.orders_enabled)

        disabled_new = sorted(new_models)[0]
        first.external_model_checks[disabled_new].setChecked(False)
        first.close()
        stored = self.store.load_ui_preferences()
        self.assertEqual(stored['version'], 4)
        self.assertNotIn(disabled_new, stored['external_models'])

        second = self.make_window()
        restored = set(second._chosen_external_models())
        self.assertTrue(old_enabled <= restored)
        self.assertFalse(old_disabled & restored)
        self.assertFalse(second.external_model_checks[disabled_new].isChecked())
        self.assertEqual(restored & (new_models - {disabled_new}), new_models - {disabled_new})
        self.assertFalse(second.monitoring)
        self.assertFalse(second.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_preferences_are_store_scoped_and_invalid_values_cannot_arm(self):
        self.store.save_ui_preferences({
            'version': 1, 'interval': -5, 'external_models': ['unknown'],
            'orders_enabled': True, 'monitoring': True, 'pending_auto_arm': True,
            'external_krw': -100, 'close_all_at_market_end': False,
        })
        first = self.make_window()
        self.assertEqual(first.interval.value(), 30)
        self.assertEqual(first.external_krw.value(), 10_000_000)
        self.assertFalse(hasattr(first, 'close_all_at_market_end'))
        self.assertFalse(first.monitoring)
        self.assertFalse(first.engine.orders_enabled)
        self.assertEqual(first._chosen_external_models(), ())

        other = WatchStore(self.folder / 'second' / 'watchlist.sqlite3')
        second = self.make_window(other)
        self.assertEqual(second.store.load_ui_preferences(), {})
        self.assertEqual(second.interval.value(), 30)
        self.assertFalse(hasattr(second, 'close_all_at_market_end'))
        self.assertFalse(hasattr(second.engine, "close_liquidator"))
        self.assertFalse(second.monitoring)
        self.assertFalse(second.engine.orders_enabled)

    def test_malformed_stored_json_uses_defaults(self):
        with self.store.connection() as db:
            db.execute('INSERT INTO settings(key,value) VALUES(?,?)',
                       ('desktop_ui_preferences_v1', '{broken'))
        window = self.make_window()
        self.assertEqual(window.interval.value(), 30)
        self.assertFalse(window.monitoring)
        self.assertFalse(window.engine.orders_enabled)


    def test_exposure_limits_survive_restart_and_stay_locked_while_monitoring(self):
        first = self.make_window()
        first.symbol_exposure_percent.setValue(15)
        first.model_exposure_percent.setValue(25)
        first.close()
        self.wait_idle(first)
        second = self.make_window()
        self.assertEqual(second.symbol_exposure_percent.value(), 15)
        self.assertEqual(second.engine.model_exposure_percent, Decimal(25))
        second.monitoring = True
        second.update_controls()
        self.assertFalse(second.symbol_exposure_percent.isEnabled())
        self.assertFalse(second.model_exposure_percent.isEnabled())
        second.monitoring = False

    def test_exposure_limit_defaults_do_not_carry_to_another_account(self):
        first = self.make_window()
        first.symbol_exposure_percent.setValue(15)
        first._save_preferences()
        second = self.make_window(WatchStore(self.folder / 'second' / 'watchlist.sqlite3'))
        self.assertEqual(second.symbol_exposure_percent.value(), 0)
        self.assertEqual(second.model_exposure_percent.value(), 0)


if __name__ == '__main__':
    unittest.main()
