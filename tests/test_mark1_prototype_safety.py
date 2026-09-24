"""Independent offline checks for prototype orders, price queries and shutdown."""
from __future__ import annotations

import copy
import importlib.util
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dockdack.autotrade import AutoTrader
from dockdack.exceptions import OrderNotSent
from dockdack.gui_service import Instrument
from dockdack.lstm30_close import CLOSE_CONFIRMATION, CloseLiquidator
from dockdack.mark1_prototype_adapter import PrototypeAutoTrader, PrototypeReadOnlyService
from dockdack.models import Market, TradingMode
from dockdack.watchlist import TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class BrokerFixture:
    def __init__(self):
        self.mode = TradingMode.DEMO
        self.account_domestic = Mock(return_value='account-read')
        self.account_us = Mock(return_value='us-account-read')
        self.place_order = Mock(side_effect=AssertionError('broker mutation reached'))
        self.cancel_order = Mock(side_effect=AssertionError('broker mutation reached'))
        self.top_volume = Mock(return_value=('broker-rank',))


class ServiceFixture(FakeTradingService):
    def __init__(self):
        super().__init__()
        self.fixture_broker = BrokerFixture()
        self.brokers = {Market.DOMESTIC: self.fixture_broker}
        self.broker_calls = 0
        self.quote_spy = Mock(return_value='quote-read')
        self.top_volume = Mock(return_value=('service-rank',))

    def broker(self, market):
        self.broker_calls += 1
        return self.fixture_broker

    def quote(self, *args, **kwargs):
        return self.quote_spy(*args, **kwargs)


class PrototypeFacadeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.service = ServiceFixture()
        self.facade = PrototypeReadOnlyService(self.service)
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('network forbidden'))
        self.request = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.request.assert_not_called)

    def test_all_service_mutations_block_before_forwarding(self):
        for method in PrototypeReadOnlyService._WRITES:
            with self.subTest(method=method), self.assertRaises(OrderNotSent):
                getattr(self.facade, method)(object(), confirmation='DEMO_AUTOTRADE')
        self.assertFalse(self.facade.live_risk_acknowledged)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.service.broker_calls, 0)

    def test_broker_mutations_block_through_method_and_public_mapping(self):
        for broker in (self.facade.broker(Market.DOMESTIC), self.facade.brokers[Market.DOMESTIC]):
            for name in ('place_order', 'cancel_order', 'modify_order', 'amend_order',
                         'build_order', 'build_order_at_current_price', 'buy', 'sell'):
                with self.subTest(method=name), self.assertRaises(OrderNotSent):
                    getattr(broker, name)(object())
        self.service.fixture_broker.place_order.assert_not_called()
        self.service.fixture_broker.cancel_order.assert_not_called()

    def test_facade_does_not_expose_unknown_transport_or_factory(self):
        for name in ('factory', '_prepared', '_http_for', 'anything_new'):
            with self.subTest(method=name), self.assertRaises(AttributeError):
                getattr(self.facade, name)
        broker = self.facade.broker(Market.DOMESTIC)
        for name in ('_http_for', 'config', 'anything_new'):
            with self.subTest(method=name), self.assertRaises(AttributeError):
                getattr(broker, name)

    def test_readonly_calls_and_broker_snapshots_remain_available(self):
        self.assertEqual(self.facade.quote('instrument'), 'quote-read')
        self.assertEqual(self.facade.broker(Market.DOMESTIC).account_domestic(), 'account-read')
        self.service.quote_spy.assert_called_once_with('instrument')
        self.service.fixture_broker.account_domestic.assert_called_once_with()
        self.assertEqual(self.service.submitted, [])

    def test_wrapper_never_changes_or_disarms_original_service(self):
        before = self.service.__dict__.copy()
        with self.assertRaises(OrderNotSent):
            self.facade.submit(object())
        self.assertEqual(self.service.__dict__, before)
        self.assertIsNot(self.facade.brokers, self.service.brokers)
        self.assertIsNot(self.facade.brokers[Market.DOMESTIC], self.service.fixture_broker)

    def test_real_mode_rejected_at_construction(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            PrototypeReadOnlyService(self.service)
        self.assertEqual(self.service.broker_calls, 0)

    def test_previously_obtained_read_cannot_escape_later_mode_change(self):
        read = self.facade.quote
        self.service.mode = TradingMode.REAL
        with self.assertRaises((ValueError, OrderNotSent)):
            read('instrument')
        self.service.quote_spy.assert_not_called()

    def test_previously_obtained_broker_read_checks_mode_at_call_time(self):
        broker = self.facade.broker(Market.DOMESTIC)
        read = broker.account_domestic
        self.service.fixture_broker.mode = TradingMode.REAL
        with self.assertRaises((ValueError, OrderNotSent)):
            read()
        self.service.fixture_broker.account_domestic.assert_not_called()

    def test_new_volume_ranking_is_readonly_and_retains_dynamic_mode_guards(self):
        service_read = self.facade.top_volume
        broker_read = self.facade.broker(Market.DOMESTIC).top_volume
        self.assertEqual(service_read(Market.DOMESTIC, 100), ('service-rank',))
        self.assertEqual(broker_read(Market.DOMESTIC, 100), ('broker-rank',))
        self.service.top_volume.assert_called_once_with(Market.DOMESTIC, 100)
        self.service.fixture_broker.top_volume.assert_called_once_with(Market.DOMESTIC, 100)
        self.service.mode = self.service.fixture_broker.mode = TradingMode.REAL
        for method in (service_read, broker_read):
            with self.assertRaises(ValueError):
                method(Market.DOMESTIC, 100)
        self.assertEqual(self.service.top_volume.call_count, 1)
        self.assertEqual(self.service.fixture_broker.top_volume.call_count, 1)
        self.assertEqual(self.service.submitted, [])


class PrototypeEngineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = FakeTradingService()
        self.item = WatchItem(Instrument(Market.DOMESTIC, '005930', 'KRX'), 'fixture', 31)
        self.store = WatchStore(Path(self.temp.name) / 'prototype.sqlite3', mode=TradingMode.DEMO)
        self.store.save_item(self.item)
        self.engine = PrototypeAutoTrader(self.service, self.store, items=[self.item], clock=lambda: NOW)
        self.rule = TriggerRule.create(self.item, kind='price_ge', side='buy', quantity=1,
                                       max_notional=Decimal(1000), threshold=Decimal(95))
        self.store.add_rule(self.rule)
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('network forbidden'))
        self.request = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.request.assert_not_called)

    def test_forced_parent_event_and_all_confirmation_strings_cannot_arm(self):
        for confirmation in ('DEMO_AUTOTRADE', 'REAL_AUTOTRADE', '', None):
            self.engine._armed.set()
            self.assertFalse(self.engine.orders_enabled)
            with self.assertRaises(ValueError):
                self.engine.enable_orders(confirmation)
            self.assertFalse(self.engine.orders_enabled)
            self.assertFalse(self.engine._armed.is_set())

    def test_direct_base_enable_dispatches_permanent_permission_guard(self):
        with self.assertRaises(OrderNotSent):
            AutoTrader.enable_orders(self.engine, 'DEMO_AUTOTRADE')
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.store.attempts(), ())

    def test_every_order_stage_rejects_even_when_parent_event_forced(self):
        for name in ('_preflight', '_execute', '_execute_once', '_before_order_send'):
            self.engine._armed.set()
            with self.subTest(stage=name), self.assertRaises(OrderNotSent):
                getattr(self.engine, name)(self.item, self.rule, None)
            self.assertFalse(self.engine._armed.is_set())
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_matching_trigger_poll_records_quotes_but_never_claims_order(self):
        self.engine._armed.set()
        self.engine.poll()
        self.engine.poll()
        self.assertEqual(self.service.quote_calls, 2)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.store.rules()[0].status, 'ready')

    def test_even_explicit_close_all_cannot_bypass_engine_or_service(self):
        closing = CloseLiquidator(self.engine.service, self.store, self.engine,
                                  enabled=True, confirmation=CLOSE_CONFIRMATION, clock=lambda: NOW)
        self.engine._armed.set()
        with self.assertRaises(OrderNotSent):
            closing._send_one({'market': 'domestic', 'symbol': '005930', 'exchange': 'KRX'})
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.store.attempts(), ())

    def test_prototype_disarm_stop_does_not_affect_separate_existing_engine(self):
        other_store = WatchStore(Path(self.temp.name) / 'existing.sqlite3', mode=TradingMode.DEMO)
        other_store.save_item(self.item)
        other_store.add_rule(self.rule)
        existing = AutoTrader(self.service, other_store, clock=lambda: NOW)
        existing.enable_orders('DEMO_AUTOTRADE')
        self.engine.stop()
        self.engine.disarm()
        self.assertTrue(existing.orders_enabled)
        self.assertFalse(existing._stop.is_set())
        self.assertEqual(self.service.submitted, [])
        existing.disarm()

    def test_changed_service_mode_blocks_poll_without_any_quote(self):
        self.service.mode = TradingMode.REAL
        with self.assertRaises(ValueError):
            self.engine._ensure_environment()
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_new_portfolio_target_display_uses_prototype_point_nine_not_main_point_eight(self):
        targets = self.engine.holding_exit_targets(position())
        self.assertEqual(targets['take_profit_price'], Decimal('101'))
        self.assertEqual(targets['stop_loss_price'], Decimal('99.1'))
        self.assertNotEqual(targets['stop_loss_price'], Decimal('99.2'))
        self.assertIn('0.9', targets['source'])
        self.assertNotIn('0.8', targets['source'])
        self.assertEqual(self.service.quote_calls, 0)
        self.assertEqual(self.service.submitted, [])

    def test_forced_main_holdings_pass_observes_targets_without_creating_order(self):
        self.service.positions = (position(),)
        self.service.prices = [Decimal('99.1')]
        self.engine.enable_holdings_exits = True
        self.engine._armed.set()
        events = []
        before_rules = tuple(self.store.rules())
        self.engine._holdings_pass(set(), progress=events.append)
        quotes = [payload for kind, payload in events if kind == 'holding_quote']
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0]['targets']['stop_loss_price'], Decimal('99.1'))
        self.assertTrue(any(kind == 'holdings_progress' and payload['phase'] == 'complete'
                            for kind, payload in events))
        self.assertEqual(tuple(self.store.rules()), before_rules)
        self.assertEqual(self.store.attempts(), ())
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(self.engine.orders_enabled)


class PrototypePriceQuerySafetyTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('network forbidden'))
        self.request = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.request.assert_not_called)

    @staticmethod
    def two_charts():
        from test_mark1_adapter import mark1_chart
        first = mark1_chart()
        second = copy.deepcopy(first)
        second['export_id'] = 'price-query-second-export'
        second['stocks'][0]['price'] = '102'
        return first, second

    def test_main_mark0_price_independent_cache_still_reuses_completed_history(self):
        from test_lstm30_adapter import producer, prediction
        model = SimpleNamespace(metadata={'market': 'domestic'}, predict=Mock(return_value=prediction()))
        adapter = producer(predictor=model)
        first, second = self.two_charts()
        adapter(first)
        adapter(second)
        self.assertEqual(model.predict.call_count, 1)

    def test_mark1_current_price_hook_is_not_using_main_mark0_cache(self):
        from test_mark1_adapter import producer, prediction
        adapter, model = producer()
        model.predict.side_effect = [prediction(.7), prediction(.4)]
        first, second = self.two_charts()
        original = copy.deepcopy((first, second))
        initial, _ = adapter(first)
        final, _ = adapter(second)
        self.assertEqual(initial['signals'][0]['action'], 'buy')
        self.assertEqual(final['signals'][0]['action'], 'hold')
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual([call.kwargs['current_price'] for call in model.predict.call_args_list],
                         [Decimal('100'), Decimal('102')])
        self.assertEqual((first, second), original)

    def test_prototype_price_query_reinfers_but_exact_export_replay_remains_immutable(self):
        from test_mark1_prototype_adapter import producer
        adapter, model = producer()
        high = copy.deepcopy(model.predict.return_value)
        low = {**high, 'probability_success': .4, 'predicts_success': False}
        model.predict.side_effect = [high, low]
        first, second = self.two_charts()
        initial, _ = adapter(first)
        final, _ = adapter(second)
        replay, _ = adapter(second)
        self.assertEqual(initial['signals'][0]['action'], 'buy')
        self.assertEqual(final['signals'][0]['action'], 'hold')
        self.assertEqual(replay, final)
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual([call.kwargs['current_price'] for call in model.predict.call_args_list],
                         [Decimal('100'), Decimal('102')])


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Qt imports required; no visible GUI is created')
class PrototypeWorkerDrainSafetyTests(unittest.TestCase):
    def test_all_four_worker_types_retain_lock_until_completion(self):
        from dockdack.lstm30_gui import LSTM30WatchlistDialog
        workers = ('worker', '_inspection_worker', '_activity_worker', '_schedule_probe')
        timers = ('timer', 'schedule_timer', 'environment_timer', 'order_status_timer', 'health_timer', 'close_timer')
        for active in workers:
            state = SimpleNamespace(**{name: None for name in workers},
                **{name: Mock() for name in timers}, stop_monitoring=Mock(),
                session_controller=Mock(), session_lock=Mock(), _lstm_released=False,
                _pending_environment=object(), _close_when_idle=False)
            setattr(state, active, object())
            with self.subTest(active_worker=active):
                self.assertFalse(LSTM30WatchlistDialog.shutdown(state))
                state.session_lock.release.assert_not_called()
                state.session_controller.close.assert_not_called()
                self.assertTrue(state._close_when_idle)
                self.assertIsNone(state._pending_environment)
                for name in timers:
                    getattr(state, name).stop.assert_called_once_with()
                setattr(state, active, None)
                self.assertTrue(LSTM30WatchlistDialog.shutdown(state))
                state.session_lock.release.assert_called_once_with()
                self.assertTrue(state._lstm_released)
                self.assertTrue(LSTM30WatchlistDialog.shutdown(state))
                state.session_lock.release.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
