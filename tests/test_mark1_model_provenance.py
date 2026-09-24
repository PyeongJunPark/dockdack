"""Offline durable identity tests: no broker requests, no running GUI or real DB."""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

from dockdack.autotrade import AutoTrader
from dockdack.execution_policy import holding_exit_targets
from dockdack.models import Market, TradingMode
from dockdack.signal_bridge import (
    ExternalPolicy, export_charts, ingest_signals, mark1_prototype_origin,
    prototype_family, prototype_order_label, prototype_record_family,
)
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position

OLD = "mark1-prototype-demo-trigger"
NEW = "mark1-1-prototype-demo-trigger"


def signal_record(source=NEW):
    family = prototype_family(source)
    payload = {"signal_id": family.id + ":fixture", "market": "domestic", "exchange": "KRX",
               "symbol": "005930", "action": "buy"}
    return {"source_id": source, "signal_id": payload["signal_id"], "payload": json.dumps(payload),
            "decision": "buy", "watch_id": "domestic:KRX:005930", "rule_id": "test-buy"}


def ledger_metadata(record):
    return {"external_source_id": record["source_id"], "external_signal_id": record["signal_id"],
            "external_payload": record["payload"], "external_decision": record["decision"],
            "external_watch_id": record["watch_id"]}


class PrototypeFamilyTests(unittest.TestCase):
    def test_fractional_policy_and_titles(self):
        self.assertEqual((prototype_family(OLD).take_profit, prototype_family(OLD).stop_loss), (D('.01'), D('.009')))
        self.assertEqual((prototype_family(NEW).take_profit, prototype_family(NEW).stop_loss), (D('.005'), D('.004')))
        self.assertEqual(prototype_family(NEW).title, "mark1.1 prototype")

    def test_both_families_remain_real_blocked_even_when_renamed(self):
        for source in (OLD, NEW, "MARK1_1_PROTOTYPE", "mark1.1-prototype"):
            self.assertTrue(mark1_prototype_origin(source))
        for family in ("mark1-prototype", "mark1-1-prototype"):
            self.assertTrue(mark1_prototype_origin("renamed", {"signal_id": family + ":saved"}))
            self.assertTrue(mark1_prototype_origin("renamed", {"origin_strategy": family}))

    def test_cross_family_and_renamed_source_cannot_claim_ownership(self):
        for source, signal_id in ((OLD, "mark1-1-prototype:wrong"), (NEW, "mark1-prototype:wrong"),
                                  ("renamed", "mark1-1-prototype:wrong"), (NEW, "unmarked")):
            with self.subTest(source=source, signal_id=signal_id), self.assertRaises(ValueError):
                prototype_family(source, {"signal_id": signal_id})

    def test_legacy_original_buy_without_optional_new_metadata_is_valid(self):
        self.assertEqual(prototype_record_family(signal_record(OLD)).id, "mark1-prototype")

    def test_record_identity_and_instrument_links_must_agree(self):
        for change in ({"signal_id": "mark1-prototype:forged"}, {"watch_id": "domestic:KRX:000660"},
                       {"decision": "sell"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prototype_record_family({**signal_record(), **change}, action="buy")

    def test_mismatched_payload_metadata_is_not_accepted(self):
        record = signal_record()
        payload = json.loads(record["payload"])
        for change in ({"strategy_id": "mark1-prototype"}, {"strategy_id": "forged"}, {"trading_mode": "real"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                prototype_record_family({**record, "payload": json.dumps({**payload, **change})})

    def test_buy_label_does_not_guess_unknown_or_sell_from_latest_selection(self):
        row = {"side": "buy", "watch_id": "domestic:KRX:005930", **ledger_metadata(signal_record())}
        self.assertEqual(prototype_order_label(row), "mark1.1 prototype")
        self.assertEqual(prototype_order_label({**row, "side": "sell"}), "매수 출처 미확인")
        self.assertEqual(prototype_order_label({"side": "buy", "model_title": "mark1.1 prototype"}), "미확인 / 수동·외부")
        self.assertIn("불일치", prototype_order_label({**row, "external_source_id": OLD}))

    def test_new_family_cannot_register_real_source_or_validator(self):
        from test_trading_environment import RealFakeService
        with tempfile.TemporaryDirectory() as directory:
            service = RealFakeService()
            store = WatchStore(Path(directory) / 'real-offline.sqlite3', mode=TradingMode.REAL,
                               storage_scope=service.storage_scope)
            engine = AutoTrader(service, store, clock=lambda: NOW)
            policy = ExternalPolicy(NEW, 1, D('500'), D('1000'))
            with self.assertRaises(ValueError):
                engine.configure_external_sources([(policy, lambda: None)])
            with self.assertRaises(ValueError):
                engine.configure_source_validators({NEW: lambda *args, **kwargs: None})
            self.assertEqual(service.submitted, [])


class HoldingFamilyTests(unittest.TestCase):
    def targets(self, source=NEW, *, saved_source=None, record=None, mode=TradingMode.DEMO, average=D('110')):
        original = signal_record(source) if record is None else record
        saved = {"source": saved_source or source, "rule_id": "test-buy", "take_profit_price": D('999'),
                 "stop_loss_price": D('888')}
        store = SimpleNamespace(mode=mode, exit_targets=lambda _: saved, external_for_rule=lambda _: original)
        return holding_exit_targets(store, replace(position(), average_price=average))

    def test_each_position_keeps_own_policy_on_actual_broker_average(self):
        old, new = self.targets(OLD), self.targets(NEW)
        self.assertEqual((old['take_profit_price'], old['stop_loss_price']), (D('111.1'), D('109.01')))
        self.assertEqual((new['take_profit_price'], new['stop_loss_price']), (D('110.55'), D('109.56')))
        self.assertEqual(new['model_title'], 'mark1.1 prototype')
        self.assertEqual(new['buy_signal_id'], 'mark1-1-prototype:fixture')

    def test_renamed_nonreserved_bracket_uses_original_record(self):
        self.assertEqual(self.targets(OLD, saved_source='renamed')['model_id'], 'mark1-prototype')

    def test_cross_family_bracket_cannot_rewrite_old_holding(self):
        with self.assertRaisesRegex(ValueError, '원본 매수'):
            self.targets(OLD, saved_source=NEW)

    def test_unknown_average_has_provenance_but_no_executable_targets(self):
        result = self.targets(average=D('NaN'))
        self.assertEqual(result['model_title'], 'mark1.1 prototype')
        self.assertIsNone(result['take_profit_price'])
        self.assertIsNone(result['stop_loss_price'])

    def test_model_owned_holding_cannot_be_used_in_real(self):
        with self.assertRaisesRegex(ValueError, '모의'):
            self.targets(mode=TradingMode.REAL)


class DurableOrderProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / 'offline.sqlite3')
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve('005930'), days=31)
        self.store.save_item(self.item)
        AutoTrader(self.service, self.store, clock=lambda: NOW).snapshot(self.item)
        self.chart = export_charts(self.store, Path(self.temp.name) / 'chart.json', now=NOW)

    def ingest(self, source=NEW, **overrides):
        family = prototype_family(source)
        entry = {**json.loads(signal_record(source)['payload']), 'export_id': self.chart['export_id'],
                 'quantity': 1, 'max_notional': '500', 'generated_at': NOW.isoformat(),
                 'expires_at': (NOW + timedelta(minutes=2)).isoformat(),
                 'strategy_id': family.id, 'model_title': family.title, 'model_version': '20260924-v1',
                 'model_manifest_sha256': 'a' * 64, **overrides}
        ingest_signals(self.store, {'schema_version': 1, 'source_id': source, 'trading_mode': 'demo',
                                   'signals': [entry]}, ExternalPolicy(source, 1, D('500'), D('1000')), now=NOW)
        return self.store.rules(statuses=('ready',))[0]

    def test_order_history_keeps_source_on_restart_and_target_change(self):
        rule = self.ingest()
        self.store.claim(rule, D('100'), NOW)
        self.store.finish(rule.id, 'accepted', 'offline fixture', 'order-1')
        self.store.set_exit_targets(self.item.id, D('101'), D('99.1'), source=OLD, rule_id='other-buy')
        restarted = WatchStore(self.store.path)
        row = restarted.order_history()[0]
        self.assertEqual(row['external_source_id'], NEW)
        self.assertEqual(prototype_order_label(row), 'mark1.1 prototype')
        self.assertEqual(json.loads(row['external_payload'])['model_version'], '20260924-v1')
        self.assertEqual(self.service.submitted, [])

    def test_retry_order_uses_root_signal_and_does_not_duplicate_rows(self):
        rule = self.ingest()
        self.store.claim(rule, D('100'), NOW)
        self.store.finish(rule.id, 'rejected', 'offline fixture')
        retry = self.store.retry_rule(rule)
        self.store.claim(retry, D('100'), NOW + timedelta(seconds=1))
        self.store.finish(retry.id, 'accepted', 'offline fixture', 'order-2')
        rows = self.store.order_history(limit=None)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(prototype_order_label(row) == 'mark1.1 prototype' for row in rows))

    def test_bad_optional_model_identity_rejected_before_persistence(self):
        for changes in ({'strategy_id': 'mark1-prototype'}, {'model_title': 'mark1 prototype'},
                        {'model_manifest_sha256': 'not-a-hash'}, {'model_version': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.ingest(**changes)
        self.assertEqual(self.store.rules(statuses=('ready',)), ())

    def test_filled_order_preserves_signal_identity_without_reading_holdings(self):
        rule = self.ingest(OLD)
        self.store.claim(rule, D('100'), NOW)
        self.store.finish(rule.id, 'accepted', 'offline fixture', 'order-1')
        self.store.record_execution(rule.id, filled_quantity=D('1'), remaining_quantity=D('0'),
                                    fill_price=D('100'), observed_at=NOW)
        self.store.finish(rule.id, 'filled', 'offline fixture')
        row = self.store.order_history()[0]
        self.assertEqual(row['status'], 'filled')
        self.assertEqual(prototype_order_label(row), 'mark1 prototype')
        self.assertEqual(self.store.exit_targets(self.item.id), None)


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'Install the gui extra')
class ModelAttributionGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_holdings_model_visible_next_to_name_and_unknown_not_guessed(self):
        from dockdack.portfolio import PortfolioMarketState
        from dockdack.portfolio_gui import PortfolioPanel
        from test_portfolio import account, NOW as PORTFOLIO_NOW
        panel = PortfolioPanel()
        self.addCleanup(panel.close)
        panel.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account(), PORTFOLIO_NOW, PORTFOLIO_NOW)})
        self.assertEqual(panel.table.item(0, 11).text(), '미확인 / 수동·외부')
        panel.set_exit_targets({'domestic:KRX:005930': {'model_title': 'mark1.1 prototype',
                                'take_profit_price': D('100.5'), 'stop_loss_price': D('99.6'),
                                'buy_signal_id': 'mark1-1-prototype:test'}})
        self.assertEqual(panel.table.item(0, 11).text(), 'mark1.1 prototype')
        self.assertLess(panel.table.horizontalHeader().visualIndex(11), panel.table.horizontalHeader().visualIndex(9))
        self.assertIn('mark1-1-prototype:test', panel.table.item(0, 11).toolTip())

    def test_both_order_and_daily_journal_show_persisted_buy_model(self):
        from PySide6.QtCore import QDate
        from dockdack.activity_snapshot import LedgerSnapshot
        from dockdack.operations_gui import OrderHistoryPanel
        from dockdack.trade_journal import daily_trade_journal
        from dockdack.trade_journal_gui import DailyTradeJournalPanel
        from test_trade_journal import order
        ledger = (order(1, **ledger_metadata(signal_record())),)
        snapshot = LedgerSnapshot(1, ledger, daily_trade_journal(ledger))
        executions = OrderHistoryPanel()
        self.addCleanup(executions.close)
        executions.apply_snapshot(snapshot)
        self.assertEqual(executions.table.item(0, 13).text(), 'mark1.1 prototype')
        journal = DailyTradeJournalPanel(SimpleNamespace(mode='demo'))
        self.addCleanup(journal.close)
        journal.dates['domestic'].setDate(QDate(2026, 9, 15))
        journal.apply_snapshot(snapshot)
        self.assertEqual(journal.tables['domestic'].item(0, 9).text(), 'mark1.1 prototype')


if __name__ == '__main__':
    unittest.main()
