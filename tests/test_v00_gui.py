"""Offline ver 0.0 desktop integration: temporary ledger and fake models only."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
from dataclasses import replace
from datetime import timedelta
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
    from PySide6.QtWidgets import QApplication, QLabel
    from dockdack.v00_app import (DesktopModelBridge, ExternalFeedGroup, MARK1_TRIGGER,
                                 MARK11_TRIGGER, MARK12_TRIGGER, MARK14_TRIGGER,
                                 PREOPEN_MODEL_IDS, PROTOTYPE_NOTICES, V00Window,
                                 desktop_model_choices)
    from dockdack.v00_widgets import OrderToast, SourceList

from dockdack.history import DailyHistory
from dockdack.models import AccountSnapshot, Market, Quote, TradingMode
from dockdack.portfolio import PortfolioMarketState
from dockdack.watchlist import MarketSnapshot, WatchItem, WatchStore
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
            if (getattr(self.window, "_activity_worker", None) is None and getattr(self.window, "_schedule_probe", None) is None
                    and getattr(self.window, '_workspace_worker', None) is None):
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
        from dockdack.version import APP_RELEASE
        self.assertIn(APP_RELEASE, self.window.message.text())
        self.assertIn('개장 10분 전 / 개장 / 매 정시', self.window.hourly_ranking.text())
        self.assertIn('거래량 TOP100', self.window.ranking_button.text())
        self.assertIn('09:30', self.window.hourly_ranking.toolTip())
        self.assertTrue(any(f'DOCKDACK  ver {APP_RELEASE}' == widget.text()
                            for widget in self.window.findChildren(QLabel)))
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

    def test_new_startup_selects_all_models_but_never_arms_orders(self):
        trigger, models = desktop_model_choices(None, False, [])
        self.assertEqual(trigger, 'none')
        self.assertEqual(models, list(PROTOTYPE_NOTICES))
        self.assertEqual(self.window.signal_connection_page.tabText(
            self.window.signal_connection_page.indexOf(self.window.mark14_panel)), '장전 모델')
        self.assertEqual(self.window.signal_connection_page.count(), 2)
        self.assertEqual(self.window.signal_connection_page.tabText(
            self.window.signal_connection_page.indexOf(self.window.external_panel)), '장중 모델')
        self.assertEqual(self.window.workspace_tabs.tabText(
            self.window.workspace_tabs.indexOf(self.window.signal_connection_page)), 'AI 추론모델 연결')
        self.assertEqual(self.window.workspace_tabs.currentWidget(), self.window.watch_page)
        self.assertEqual([self.window.workspace_tabs.tabText(index)
                          for index in range(self.window.workspace_tabs.count())][:7],
                         ['관심종목·차트', '보유종목', '실제 주문·체결', '매매일지',
                          '모델 성과', 'AI 추론모델 연결', '서버·감시 로그'])
        self.assertEqual(self.window.watch_tables[Market.DOMESTIC].columnCount(), 7)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_mark14_connects_own_demo_source_without_starting_child_or_orders(self):
        self.window.external_model_checks[MARK14_TRIGGER].setChecked(True)
        self.drain_activity()
        self.window.configure_external()
        feed = self.window._prototype_feeds[MARK14_TRIGGER]
        self.assertEqual(feed.source_id, 'mark1-4-prototype-demo-trigger')
        self.assertIn(feed.source_id, self.window.engine.external_sources)
        self.assertFalse(feed.client.is_alive)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_every_checked_preopen_model_gets_its_own_demo_source_without_arming(self):
        self.assertEqual(len(PREOPEN_MODEL_IDS), 9)
        self.assertEqual(set(desktop_model_choices(None, False, [])[1]),
                         set(PROTOTYPE_NOTICES))
        # This fixture passes builtin=False directly; the real desktop launcher
        # applies desktop_model_choices before constructing the window.
        for model in PREOPEN_MODEL_IDS:
            check = self.window.external_model_checks[model]
            check.blockSignals(True)
            check.setChecked(True)
            check.blockSignals(False)
        self.window.configure_external()
        sources = set()
        paths = set()
        for model in PREOPEN_MODEL_IDS:
            feed = self.window._prototype_feeds[model]
            self.assertEqual(feed.source_id, model + '-demo-trigger')
            self.assertIn(feed.source_id, self.window.engine.external_sources)
            self.assertFalse(feed.client.is_alive)
            if model == MARK14_TRIGGER:
                self.assertTrue(feed.risk_notice)
            else:
                self.assertIn('모의 전용', PROTOTYPE_NOTICES[model])
            sources.add(feed.source_id)
            paths.add(self.window._prototype_output_path(model))
        self.assertEqual(len(sources), len(PREOPEN_MODEL_IDS))
        self.assertEqual(len(paths), len(PREOPEN_MODEL_IDS))
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.window.monitoring)
        self.assertEqual(self.service.submitted, [])

    def test_mark14_preopen_display_is_score_not_a_fourth_probability(self):
        feeds = self._score_feeds()
        check = self.window.external_model_checks[MARK14_TRIGGER]
        check.blockSignals(True)
        check.setChecked(True)
        check.blockSignals(False)
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        self.assertIn('52.5%', self.window.latest_model_summary.text())
        self.window._progress(('mark14_preopen', {
            'market': 'domestic', 'state': 'prepared', 'session_open': NOW.isoformat(),
            'candidates': [{'symbol': '005930', 'score': 0.82, 'threshold': 0.45,
                            'selected': True, 'out_of_training_universe': True}],
        }))
        table = self.window.mark14_panel.tables['domestic']
        self.assertEqual(table.rowCount(), 1)
        self.assertIn('0.820000', table.item(0, 1).text())
        self.assertIn('학습종목 밖', table.item(0, 3).text())
        self.assertIn('52.5%', self.window.latest_model_summary.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_mark14_preopen_is_frozen_once_without_orders(self):
        from dockdack.market_schedule import session_on
        session = session_on(Market.DOMESTIC, NOW.date())
        self.assertIsNotNone(session)
        moment = session.opened - timedelta(minutes=5)
        self.window.engine.clock = lambda: moment
        result = {'market': 'domestic', 'state': 'prepared',
                  'session_open': session.opened.isoformat(), 'candidates': []}
        feed = SimpleNamespace(client=SimpleNamespace(is_alive=True), close=Mock(),
                               prepare_preopen=Mock(return_value=result))
        self.window._prototype_feeds = {MARK14_TRIGGER: feed}
        events = []
        gathered = SimpleNamespace(ok=True, candidates=tuple({'symbol': str(i)} for i in range(100)))
        with patch('dockdack.mark1_4_preopen.collect_preopen_candidates', return_value=gathered) as collect:
            self.window._preopen_checkpoint(events.append)
            self.window._preopen_checkpoint(events.append)
        self.assertEqual(collect.call_count, 1)
        self.assertEqual(feed.prepare_preopen.call_count, 1)
        self.assertEqual(events, [('mark14_preopen', result)])
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_all_preopen_models_share_one_verified_top100_batch(self):
        from dockdack.market_schedule import session_on
        session = session_on(Market.DOMESTIC, NOW.date())
        self.assertIsNotNone(session)
        self.window.engine.clock = lambda: session.opened - timedelta(minutes=5)
        result = {'market': 'domestic', 'state': 'prepared',
                  'session_open': session.opened.isoformat(), 'candidates': []}
        feeds = {model: SimpleNamespace(client=SimpleNamespace(is_alive=True), close=Mock(),
                                        prepare_preopen=Mock(return_value=result))
                 for model in PREOPEN_MODEL_IDS}
        self.window._prototype_feeds = feeds
        events = []
        gathered = SimpleNamespace(ok=True, candidates=tuple({'symbol': str(i)} for i in range(100)))
        with patch('dockdack.mark1_4_preopen.collect_preopen_candidates', return_value=gathered) as collect:
            self.window._preopen_checkpoint(events.append)
            self.window._preopen_checkpoint(events.append)
        self.assertEqual(collect.call_count, 1)
        self.assertTrue(all(feed.prepare_preopen.call_count == 1 for feed in feeds.values()))
        self.assertEqual(len(events), len(PREOPEN_MODEL_IDS))
        self.assertEqual({value.get('model_id', MARK14_TRIGGER) for event, value in events},
                         set(PREOPEN_MODEL_IDS))
        self.assertTrue(all(event == 'mark14_preopen' for event, _ in events))
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_only_live_prepared_demo_plan_protects_frozen_open_watch_ids(self):
        from dockdack.market_schedule import session_on
        session = session_on(Market.DOMESTIC, NOW.date())
        selected = self.item.id
        feed = SimpleNamespace(client=SimpleNamespace(is_alive=True), close=Mock(), plans={
            'domestic': {'state': 'prepared', 'scored_count': 100, 'selected_count': 1,
                         'session_open': session.opened.isoformat(),
                         'candidates': [{'watch_id': selected, 'selected': True}]}})
        self.window._prototype_feeds = {MARK14_TRIGGER: feed}
        opening = session.opened + timedelta(minutes=1)
        self.assertEqual(self.window._frozen_open_watch_ids(Market.DOMESTIC, opening), (selected,))
        self.assertEqual(self.window._frozen_open_watch_ids(Market.US, opening), ())
        self.assertEqual(self.window._frozen_open_watch_ids(Market.DOMESTIC,
                         session.opened + timedelta(minutes=5)), ())
        feed.client.is_alive = False
        self.assertEqual(self.window._frozen_open_watch_ids(Market.DOMESTIC, opening), ())
        self.assertFalse(self.window.engine.orders_enabled)

    def test_selected_watch_ids_from_independent_preopen_models_survive_open_rerank(self):
        from dockdack.market_schedule import session_on
        session = session_on(Market.DOMESTIC, NOW.date())
        second = WatchItem(self.service.resolve('000660'), 'SK하이닉스')
        self.store.save_item(second)

        def feed_for(watch_id, alive=True):
            return SimpleNamespace(client=SimpleNamespace(is_alive=alive), close=Mock(), plans={
                'domestic': {'state': 'prepared', 'scored_count': 100, 'selected_count': 1,
                             'session_open': session.opened.isoformat(),
                             'candidates': [{'watch_id': watch_id, 'selected': True}]}})

        first, second_feed = PREOPEN_MODEL_IDS[1], PREOPEN_MODEL_IDS[-1]
        self.window._prototype_feeds = {first: feed_for(self.item.id),
                                        second_feed: feed_for(second.id)}
        opening = session.opened + timedelta(minutes=1)
        self.assertEqual(self.window._frozen_open_watch_ids(Market.DOMESTIC, opening),
                         (self.item.id, second.id))
        self.window.engine.clock = lambda: opening
        self.assertEqual(self.window._poll_priority_watch_ids(), (self.item.id, second.id))
        self.assertEqual(self.window._frozen_open_watch_ids(Market.DOMESTIC,
                         session.opened + timedelta(minutes=5)), ())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

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

    def test_small_window_keeps_all_watch_columns_and_chart_reachable(self):
        self.window.workspace_tabs.setCurrentWidget(self.window.watch_page)
        self.window.resize(800, 520)
        self.window.show()
        self.app.processEvents()
        # Real status text can occupy two lines even before a price is loaded.
        self.window.order_status_detail.setText('시세 감시는 계속됩니다.\n자동주문은 꺼져 있습니다.')
        self.window.connection_summary.setText('외부 AI 모델 2개 · 감시 OFF\n모델별 연결은 신호 연결에서 확인')
        self.window.latest_model_summary.setText('최근 완료 조회 삼성전자 · 09/15 10:00\n활성 모델 평균 52.5% · 매매 판단에 사용 안 함')
        self.app.processEvents()
        view = self.window.watch_tables[Market.DOMESTIC]
        viewport = self.window.workspace_scroll.viewport()
        self.assertEqual(self.window.watch_splitter.orientation(), Qt.Orientation.Vertical)
        self.assertEqual(self.window.workspace_scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(view.horizontalScrollBar().maximum(), 0)
        self.assertEqual(view.columnCount(), 7)
        self.assertFalse(hasattr(self.window, 'connection_shortcut'))
        self.assertGreater(view.height(), 150)
        first_row = view.visualItemRect(view.item(0, 0))
        first_row_bottom = view.viewport().mapTo(viewport, first_row.bottomLeft()).y()
        self.assertLess(first_row_bottom, viewport.height(), '첫 관심종목 행이 상단 화면에서 잘림')
        self.assertGreaterEqual(self.window.ranking_button.width(), self.window.ranking_button.sizeHint().width())
        self.assertGreaterEqual(self.window.hourly_ranking.width(), self.window.hourly_ranking.sizeHint().width())
        self.assertGreaterEqual(self.window.environment_selector.badge.width(),
                                self.window.environment_selector.badge.sizeHint().width())
        for width in (900, 950, 1000):
            self.window.resize(width, 520)
            self.app.processEvents()
            self.assertGreaterEqual(self.window.ranking_button.width(), self.window.ranking_button.sizeHint().width())
            self.assertGreaterEqual(self.window.hourly_ranking.width(), self.window.hourly_ranking.sizeHint().width())
        self.window.resize(800, 520)
        self.app.processEvents()
        scroll = self.window.workspace_scroll.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        chart_y = self.window.chart.mapTo(viewport, self.window.chart.rect().topLeft()).y()
        scroll.setValue(min(scroll.maximum(), max(0, chart_y - 20)))
        self.app.processEvents()
        chart_y = self.window.chart.mapTo(viewport, self.window.chart.rect().topLeft()).y()
        self.assertLess(chart_y, viewport.height())
        self.assertGreater(chart_y + self.window.chart.height(), 0)
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

    def _score_snapshot(self, *, item=None, price='100', fetched_at=NOW):
        item = item or self.item
        inst = item.instrument
        quote = Quote(inst.market, inst.symbol, item.name, inst.exchange, D(price), inst.currency)
        history = DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, 0, ())
        return MarketSnapshot(quote, history, fetched_at)

    def _score_feeds(self, *, mark1='0.638', mark11='0.412', mark12=None):
        feeds = {}
        decisions = [(MARK1_TRIGGER, mark1, 'PREDICTED_DAILY_BARRIER_SUCCESS'),
                     (MARK11_TRIGGER, mark11, 'BELOW_OR_EQUAL_BUY_THRESHOLD')]
        if mark12 is not None:
            decisions.append((MARK12_TRIGGER, mark12, 'PREDICTED_DAILY_BARRIER_SUCCESS'))
        for model, probability, reason in decisions:
            check = self.window.external_model_checks[model]
            check.blockSignals(True)
            check.setChecked(True)
            check.blockSignals(False)
            diagnostics = {}
            def publish(exported, *, data=diagnostics, value=probability, result=reason):
                stock = exported['stocks'][0]
                data[stock['watch_id']] = {
                    'watch_id': stock['watch_id'], 'reason': result,
                    'reference_price': stock['price'],
                    'prediction': {'probability_success': value}}
            feeds[model] = SimpleNamespace(source_id=model, diagnostics=diagnostics, status=model + ' connected',
                                           _ready=True, publish=Mock(side_effect=publish), close=Mock())
        self.window._prototype_feeds = feeds
        self.window.test_producer = ExternalFeedGroup(feeds)
        return feeds

    def test_model_scores_show_distinct_buy_and_hold_after_each_symbol_progress(self):
        feeds = self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertIn('MK1.0 추정확률', table.horizontalHeaderItem(4).text())
        self.assertIn('MK1.1 추정확률', table.horizontalHeaderItem(5).text())
        self.assertIn('mark1.0 prototype', self.window.external_model_checks[MARK1_TRIGGER].text())
        self.assertIn('MK1.0과 MK1.2', self.window.model_score_summary.toolTip())
        self.assertEqual(table.item(0, 4).text(), '매수 판정\n63.8%')
        self.assertEqual(table.item(0, 5).text(), '대기\n41.2%')
        self.assertIn('현재가 100 KRW', self.window.model_score_summary.text())
        self.assertIn('63.8%', self.window.model_score_summary.text())
        self.assertIn('41.2%', self.window.model_score_summary.text())
        self.assertIn('조회', table.item(0, 4).toolTip())
        self.assertIn('실제 적중률·수익률 보장 아님', table.item(0, 4).toolTip())
        other = WatchItem(self.service.resolve('000660'), 'SK하이닉스')
        self.store.save_item(other)
        self.window.reload_tables(items=self.store.items(), rules=[])
        self.assertEqual(table.item(self.window._watch_rows[self.item.id], 4).text(), '매수 판정\n63.8%')
        self.assertIn('현재가 없음', table.item(self.window._watch_rows[other.id], 4).text())
        self.assertEqual([feed.publish.call_count for feed in feeds.values()], [1, 1])
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_third_model_score_is_distinct_and_keeps_last_valid_probability(self):
        feeds = self._score_feeds(mark12='0.552')
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertIn('MK1.2 추정확률', table.horizontalHeaderItem(6).text())
        self.assertEqual(table.item(0, 4).text(), '매수 판정\n63.8%')
        self.assertEqual(table.item(0, 5).text(), '대기\n41.2%')
        self.assertEqual(table.item(0, 6).text(), '매수 판정\n55.2%')
        self.assertIn('55.2%', self.window.model_score_summary.text())
        self.assertIn('53.4%', self.window.latest_model_summary.text())
        previous = self.window.latest_model_summary.text()
        self.window._progress((self.item.id, self._score_snapshot(price='101'), 1, 1))
        self.assertIn('55.2%', table.item(0, 6).text())
        self.assertIn('최근 추정', table.item(0, 6).text() + table.item(0, 6).toolTip())
        self.assertIn('100 KRW', table.item(0, 6).toolTip())
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self.assertEqual([feed.publish.call_count for feed in feeds.values()], [1, 1, 1])
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_model_score_keeps_last_valid_across_quote_error_age_and_invalid_result(self):
        feeds = self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertIn('63.8%', table.item(0, 4).text())
        previous = self.window.latest_model_summary.text()
        self.window._progress((self.item.id, self._score_snapshot(price='101'), 1, 1))
        self.assertIn('63.8%', table.item(0, 4).text())
        self.assertIn('최근 추정', table.item(0, 4).text() + table.item(0, 4).toolTip())
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self.window._progress((self.item.id, ValueError('quote unavailable'), 1, 1))
        self.assertIn('63.8%', table.item(0, 4).text())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        self.window.engine.clock = lambda: NOW + timedelta(seconds=16)
        self.window._refresh_model_scores()
        self.assertIn('63.8%', table.item(0, 4).text())
        self.assertIn('조회', table.item(0, 4).toolTip())
        self.window.engine.clock = lambda: NOW
        feeds[MARK1_TRIGGER].diagnostics[self.item.id]['prediction']['probability_success'] = 'NaN'
        self.window._refresh_model_scores()
        self.assertIn('63.8%', table.item(0, 4).text())
        self.assertEqual(table.item(0, 5).text(), '대기\n41.2%')
        self.assertFalse(self.service.submitted)

    def test_failed_one_model_keeps_its_last_valid_score_and_other_model(self):
        feeds = self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        feeds[MARK1_TRIGGER].publish.side_effect = RuntimeError('model child stopped')
        self.window.test_producer.publish(chart())
        self.window._refresh_model_scores()
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertIn('63.8%', table.item(0, 4).text())
        self.assertEqual(table.item(0, 5).text(), '대기\n41.2%')
        self.assertFalse(self.service.submitted)

    def test_invalid_probability_without_prior_valid_score_is_not_displayed(self):
        feeds = self._score_feeds(mark1='NaN')
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertNotIn('%', table.item(0, 4).text())
        self.assertEqual(table.item(0, 5).text(), '대기\n41.2%')
        self.assertIn('모델 추정확률 —', self.window.latest_model_summary.text())
        self.assertNotIn('%', self.window.latest_model_summary.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_latest_query_summary_requires_every_enabled_model_on_same_quote(self):
        feeds = self._score_feeds(mark12='0.552')
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        summary = self.window.latest_model_summary
        self.assertLess(self.window.layout().indexOf(summary),
                        self.window.layout().indexOf(self.window.sweep_progress))
        self.assertIn('005930', summary.text())
        self.assertIn('53.4%', summary.text())  # (63.8 + 41.2 + 55.2) / 3
        previous = summary.text()
        self.assertFalse(self.window.engine.orders_enabled)

        self.window._progress((self.item.id, self._score_snapshot(price='101'), 1, 1))
        self.assertEqual(summary.text(), previous)
        for index, (model, probability) in enumerate(((MARK1_TRIGGER, '0.700'),
                                                      (MARK11_TRIGGER, '0.500'),
                                                      (MARK12_TRIGGER, '0.900')), 1):
            feeds[model].diagnostics[self.item.id] = {
                'watch_id': self.item.id,
                'reason': 'PREDICTED_DAILY_BARRIER_SUCCESS' if probability != '0.500' else 'BELOW_OR_EQUAL_BUY_THRESHOLD',
                'reference_price': '101',
                'prediction': {'probability_success': probability},
                '_display_quote_fetched_at': NOW.isoformat(),
                '_display_price': '101',
            }
            self.window._refresh_model_scores()
            if index < 3:
                self.assertEqual(summary.text(), previous)
        self.assertIn('70.0%', summary.text())  # (70.0 + 50.0 + 90.0) / 3
        self.assertNotEqual(summary.text(), previous)

        # An unchecked model must not count toward coverage or the arithmetic mean.
        check = self.window.external_model_checks[MARK12_TRIGGER]
        check.blockSignals(True)
        check.setChecked(False)
        check.blockSignals(False)
        self.window._refresh_model_scores()
        self.assertIn('60.0%', summary.text())  # (70.0 + 50.0) / 2
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_same_price_newer_quote_does_not_reuse_old_model_average(self):
        feeds = self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        previous = self.window.latest_model_summary.text()
        self.assertIn('52.5%', previous)
        refreshed = NOW + timedelta(seconds=1)
        self.window.engine.clock = lambda: refreshed
        self.window._progress((self.item.id, self._score_snapshot(fetched_at=refreshed), 1, 1))
        self.assertIn('최근 추정', self.window.watch_tables[Market.DOMESTIC].item(0, 4).text())
        self.assertIn('63.8%', self.window.watch_tables[Market.DOMESTIC].item(0, 4).text())
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        for feed in feeds.values():
            feed.diagnostics[self.item.id]['_display_quote_fetched_at'] = refreshed.isoformat()
        self.window._refresh_model_scores()
        self.assertIn('52.5%', self.window.latest_model_summary.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_latest_query_summary_ignores_selected_chart_and_failed_query(self):
        self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 2))
        other = WatchItem(self.service.resolve('000660'), 'SK하이닉스')
        self.store.save_item(other)
        self.window.reload_tables(items=self.store.items(), rules=[])
        table = self.window.watch_tables[Market.DOMESTIC]
        table.setCurrentCell(self.window._watch_rows[other.id], 0)
        self.app.processEvents()
        self.assertEqual(self.window.selected_item().id, other.id)
        self.assertIn('005930', self.window.latest_model_summary.text())
        self.assertIn('52.5%', self.window.latest_model_summary.text())
        previous = self.window.latest_model_summary.text()
        self.window._progress((other.id, ValueError('quote unavailable'), 2, 2))
        self.assertIn('005930', self.window.latest_model_summary.text())
        self.window._progress((other.id, self._score_snapshot(item=other, price='200'), 2, 2))
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        table.setCurrentCell(self.window._watch_rows[self.item.id], 0)
        self.app.processEvents()
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        american = WatchItem(self.service.resolve('AAPL'), 'Apple')
        self.store.save_item(american)
        self.window.reload_tables(items=self.store.items(), rules=[])
        self.window._progress((american.id, self._score_snapshot(item=american, price='250'), 3, 3))
        self.assertEqual(self.window.latest_model_summary.text(), previous)
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)

    def test_changing_model_connection_discards_old_display_scores_without_orders(self):
        feeds = self._score_feeds()
        self.window.test_producer.publish(chart())
        self.window._progress((self.item.id, self._score_snapshot(), 1, 1))
        table = self.window.watch_tables[Market.DOMESTIC]
        self.assertIn('63.8%', table.item(0, 4).text())
        self.assertIn('52.5%', self.window.latest_model_summary.text())

        # This is an actual user-level connection change, unlike the signal-blocked
        # checkbox in the mean-only test. Old model scores must not cross it.
        self.window.external_model_checks[MARK11_TRIGGER].setChecked(False)
        self.window._refresh_model_scores()
        self.assertNotIn('63.8%', table.item(0, 4).text())
        self.assertIn('연결 꺼짐', table.item(0, 5).text())
        self.assertIn('모델 추정확률 —', self.window.latest_model_summary.text())
        self.assertFalse(self.window.engine.orders_enabled)
        self.assertFalse(self.service.submitted)
        self.assertTrue(all(feed.close.called for feed in feeds.values()))

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
