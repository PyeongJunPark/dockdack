"""Offline checks for the DEMO daily-trained / five-minute order boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from dockdack.gui_service import Instrument
from dockdack.history import DailyBar, DailyHistory
from dockdack.market_schedule import session_on
from dockdack.minute_feed import MinuteFeedSnapshot
from dockdack.minute_model_catalog import (
    MINUTE_HEDGE_IDS, MINUTE_RESEARCH_BY_ID, MINUTE_TRANSFER_IDS,
)
from dockdack.models import AccountSnapshot, Market, MinuteBar, OrderSide, Position, Quote, TradingMode
from dockdack.research.daily_proxy_minute import DAILY_PROXY_CONFIGS
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, export_charts, prototype_record_family, read_signal_file
from dockdack.signals.daily_proxy_minute import DailyProxyMinuteFeed, MinuteResearchHoldFeed
from dockdack.trading.autotrade import AutoTrader
from dockdack.trading.model_exit_schedule import timed_exit_due
from dockdack.watchlist import MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService


KST = ZoneInfo("Asia/Seoul")
MODEL_ID = "mark1-29-prototype"
NOW = datetime(2026, 9, 29, 11, 1, tzinfo=KST)
WATCH_ID = "domestic:KRX:005930"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from dockdack.ui.v00_app import V00Window


class DailyProxyMinuteBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.now = NOW
        self.config = DAILY_PROXY_CONFIGS[MODEL_ID]
        self.bundle = self.root / self.config.bundle_name
        self.bundle.mkdir()
        self.weight_path = self.bundle / f"{self.config.architecture}.pt"
        self.weight_path.write_bytes(b"offline model fixture")
        self.manifest = {"models": {self.config.architecture: {
            "state_sha256": sha256(self.weight_path.read_bytes()).hexdigest()}}}
        self.manifest_path = self.bundle / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.artifact = {"training_symbols": frozenset({"005930"})}
        self.load_patcher = patch(
            "dockdack.signals.daily_proxy_minute.load_configured_daily_proxy",
            return_value=(self.artifact, self.manifest))
        self.load_model = self.load_patcher.start()
        self.addCleanup(self.load_patcher.stop)
        self.candidate = True
        self.trained = True
        self.lifecycle = True
        self.infer_patcher = patch(
            "dockdack.signals.daily_proxy_minute.infer_daily_proxy",
            side_effect=self._infer)
        self.infer_model = self.infer_patcher.start()
        self.addCleanup(self.infer_patcher.stop)

        bars = tuple(MinuteBar(
            Market.DOMESTIC, "005930", "KRX",
            datetime(2026, 9, 29, 9, 10, tzinfo=KST) + timedelta(minutes=5 * index),
            Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"),
            Decimal("1000"), "KRW") for index in range(self.config.lookback + 2))
        self.snapshot = MinuteFeedSnapshot(
            "005930", "KRX", NOW.date(), bars, NOW.astimezone(timezone.utc),
            bars[-1].timestamp, "a" * 64, self.root / "minute.jsonl",
            self.root / "minute.receipt.json", 1, False, False)
        self.minute_feed = SimpleNamespace(get_complete_bars=Mock(return_value=self.snapshot))
        self.store = SimpleNamespace(
            mode=TradingMode.DEMO, path=self.root / "ledger.sqlite3",
            prototype_inventory=Mock(return_value={"reconciled": True, "lots": []}),
            prototype_pending_buys=Mock(return_value=[]),
            external_for_rule=Mock())
        self.service = SimpleNamespace(
            mode=TradingMode.DEMO,
            broker=Mock(side_effect=AssertionError("unexpected broker access")),
            submit=Mock(side_effect=AssertionError("unexpected order")))
        self.engine = SimpleNamespace(
            _stop=Event(), clock=lambda: self.now, _validate_snapshot=Mock(),
            enable_orders=Mock(side_effect=AssertionError("unexpected arming")))
        self.window = SimpleNamespace(service=self.service, store=self.store, engine=self.engine)
        self.accounts = SimpleNamespace(get=Mock(return_value=(SimpleNamespace(positions=()), None)))
        self.policy = ExternalPolicy(
            MINUTE_RESEARCH_BY_ID[MODEL_ID].source_id, 1, Decimal("10000"), Decimal(0))
        self.feed = DailyProxyMinuteFeed(
            self.window, MODEL_ID, self.policy, self.root / "signal.json",
            minute_feed=self.minute_feed, account_snapshots=self.accounts,
            models_root=self.root)
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"))
        self.stock = {"watch_id": WATCH_ID, "market": "domestic", "symbol": "005930",
                      "exchange": "KRX", "status": "ok", "price": "100",
                      "quote_fetched_at": self.now.isoformat()}

    def _infer(self, artifact, bars, *, architecture, as_of):
        self.assertIs(artifact, self.artifact)
        self.assertEqual(architecture, self.config.architecture)
        self.assertEqual(len(bars), self.config.lookback + 2)
        return {"candidate": self.candidate,
                "probability_proxy": .8, "validation_threshold": .6,
                "signal_bar_label": bars[-3].timestamp.isoformat(),
                "symbol_in_daily_training_universe": self.trained,
                "lifecycle_fits_session": self.lifecycle}

    def chart(self, *, export_id="export-1", **changes):
        return {"source": "kiwoom_demo", "trading_mode": "demo",
                "export_id": export_id, "stocks": [{**self.stock, **changes}]}

    def signal(self):
        payload = read_signal_file(self.feed.output_path)
        self.assertEqual(payload["source_id"], self.policy.source_id)
        self.assertEqual(payload["trading_mode"], "demo")
        return payload["signals"][0]

    def execution(self, row=None, *, quote="100", fetched_at=None, limit="100"):
        row = row or self.signal()
        self.store.external_for_rule.return_value = {
            "source_id": self.policy.source_id, "signal_id": row["signal_id"],
            "watch_id": WATCH_ID, "decision": "buy", "payload": json.dumps(row)}
        rule = SimpleNamespace(id="offline-rule", side=OrderSide.BUY)
        fresh = SimpleNamespace(
            fetched_at=self.now if fetched_at is None else fetched_at,
            quote=SimpleNamespace(price=Decimal(quote)))
        return rule, fresh, Decimal(limit)

    def test_buy_has_model_provenance_and_rechecks_both_execution_stages(self):
        self.feed.publish(self.chart())
        row = self.signal()
        self.assertEqual(row["action"], "buy")
        self.assertEqual(row["quantity"], 1)
        self.assertEqual(row["max_notional"], "10000")
        self.assertEqual(row["strategy_id"], MODEL_ID)
        self.assertEqual(row["model_version"], self.config.bundle_name)
        self.assertEqual(row["model_manifest_sha256"],
                         sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertEqual(prototype_record_family({
            "source_id": self.policy.source_id, "signal_id": row["signal_id"],
            "watch_id": WATCH_ID, "decision": "buy", "payload": json.dumps(row)},
            watch_id=WATCH_ID, action="buy").id, MODEL_ID)
        self.assertEqual(self.feed.diagnostics[WATCH_ID]["minute_receipt_sha256"], "a" * 64)
        self.minute_feed.get_complete_bars.assert_called_once_with(
            "005930", "KRX", now=NOW, count=self.config.lookback + 2)
        self.accounts.get.assert_called_once_with(self.stock, window=self.window)
        self.load_model.assert_called_once_with(self.root, MODEL_ID)
        rule, fresh, limit = self.execution(row)
        for stage in ("preflight", "final_send"):
            self.feed.validate_execution(self.item, rule, fresh, limit, stage=stage)
        self.assertEqual(self.infer_model.call_count, 3)
        self.assertEqual(self.engine._validate_snapshot.call_count, 2)
        self.service.broker.assert_not_called()
        self.service.submit.assert_not_called()
        self.engine.enable_orders.assert_not_called()

    def test_hold_when_score_or_inventory_cannot_authorize_buy(self):
        self.candidate = False
        self.feed.publish(self.chart())
        self.assertEqual(self.signal()["action"], "hold")
        self.assertEqual(self.feed.diagnostics[WATCH_ID]["reason"], "MINUTE_THRESHOLD_NOT_MET")
        self.accounts.get.assert_not_called()
        self.candidate = True
        self.store.prototype_inventory.return_value = {"reconciled": False, "lots": []}
        self.feed.publish(self.chart(export_id="export-2"))
        self.assertEqual(self.signal()["action"], "hold")
        self.assertEqual(self.feed.diagnostics[WATCH_ID]["reason"], "MINUTE_INPUT_UNAVAILABLE")
        self.assertFalse(self.feed._decisions)
        self.service.submit.assert_not_called()

    def test_symbol_outside_frozen_training_universe_skips_minute_request(self):
        self.feed.publish(self.chart(symbol="000660", watch_id="domestic:KRX:000660"))
        self.assertEqual(self.signal()["action"], "hold")
        self.assertEqual(self.feed.diagnostics["domestic:KRX:000660"]["reason"],
                         "SYMBOL_OUTSIDE_DAILY_TRAINING")
        self.minute_feed.get_complete_bars.assert_not_called()
        self.infer_model.assert_not_called()
        self.accounts.get.assert_not_called()
        self.assertFalse(self.feed._decisions)

    def test_data_only_candidate_reaches_real_engine_fake_buy_fill_and_sell(self):
        """Exercise JSON ingestion, both send checks, lot creation and TP SELL."""
        store = WatchStore(self.root / "roundtrip.sqlite3")
        store.save_item(self.item)
        service = FakeTradingService()
        bars, day = [], NOW.date() - timedelta(days=1)
        while len(bars) < 31:
            if session_on(Market.DOMESTIC, day) is not None:
                bars.append(DailyBar(day, Decimal("100"), Decimal("101"),
                                     Decimal("99"), Decimal("100"), Decimal("1000")))
            day -= timedelta(days=1)
        history = DailyHistory(Market.DOMESTIC, "005930", "KRX", "KRW", 31,
                               tuple(reversed(bars)))
        service.history = Mock(return_value=history)
        service.safety_account = lambda instrument: AccountSnapshot(
            instrument.market, instrument.currency, service.positions,
            cash=Decimal("20000"), total_evaluation=Decimal("0"),
            available_to_order=Decimal("20000"))
        engine = AutoTrader(service, store, clock=lambda: self.now)
        engine.prototype_lots_enabled = True
        engine.enable_holdings_exits = True
        engine.external_only = True
        engine.equity_buy_percent = Decimal("10")
        engine.source_buy_percents[self.policy.source_id] = Decimal("1")
        window = SimpleNamespace(service=service, store=store, engine=engine)
        accounts = SimpleNamespace(get=lambda stock, window: (
            service.safety_account(self.item.instrument), None))
        feed = DailyProxyMinuteFeed(
            window, MODEL_ID, self.policy, self.root / "roundtrip-signal.json",
            minute_feed=self.minute_feed, account_snapshots=accounts, models_root=self.root)
        reader = SignalFileReader(store, feed.output_path, self.policy,
                                  clock=lambda: self.now)
        engine.configure_external_sources([(self.policy, reader)])
        engine.configure_source_validators({self.policy.source_id: feed.validate_execution})
        snapshot = engine.snapshot(self.item)
        feed.publish(export_charts(store, self.root / "roundtrip-chart.json", now=self.now))
        engine.enable_orders("DEMO_AUTOTRADE")
        engine._read_external()
        buy, = store.rules(self.item.id, statuses=("ready",))
        self.assertTrue(engine._execute(self.item, buy, snapshot))
        self.assertEqual((service.submitted[0].side, service.submitted[0].quantity),
                         (OrderSide.BUY, 1))
        self.assertEqual(store.attempts()[0]["status"], "accepted")
        store.record_execution(buy.id, filled_quantity=Decimal(1),
                               remaining_quantity=Decimal(0), fill_price=Decimal("100"),
                               observed_at=self.now)
        store.finish(buy.id, "filled", "offline fill")
        self.now += timedelta(minutes=1)
        service.prices = [Decimal("103")]
        service.positions = (Position(
            Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
            Decimal(1), Decimal(1), Decimal("100"), Decimal("103"),
            Decimal("103"), Decimal("3"), Decimal("3")),)
        position = service.positions[0]
        sell_snapshot = MarketSnapshot(service.quote(self.item.instrument), history, self.now)
        engine._lot_holdings_exits(self.item, position, sell_snapshot,
                                   engine.holding_exit_targets(position), set())
        self.assertEqual(len(service.submitted), 2, store.events())
        self.assertEqual((service.submitted[1].side, service.submitted[1].quantity),
                         (OrderSide.SELL, 1))
        sale, = [row for row in store.order_history(limit=None) if row["side"] == "sell"]
        self.assertEqual(sale["status"], "accepted")
        self.assertEqual(sale["prototype_lot_id"], buy.id)
        store.record_execution(sale["rule_id"], filled_quantity=Decimal(1),
                               remaining_quantity=Decimal(0), fill_price=Decimal("103"),
                               observed_at=self.now)
        store.finish(sale["rule_id"], "filled", "offline sell fill")
        inventory = store.prototype_inventory(self.item.id, broker_quantity=Decimal(0),
                                               broker_sellable=Decimal(0))
        self.assertTrue(inventory["reconciled"])
        self.assertEqual(inventory["lots"][0]["quantity_remaining"], 0)

    def test_stale_chart_quote_or_real_mode_clears_prior_buy(self):
        self.feed.publish(self.chart())
        row = self.signal()
        self.assertEqual(row["action"], "buy")
        self.feed.publish(self.chart(export_id="export-2",
                                     quote_fetched_at=(NOW - timedelta(seconds=16)).isoformat()))
        self.assertEqual(self.signal()["action"], "hold")
        self.assertFalse(self.feed._decisions)
        self.assertEqual(self.minute_feed.get_complete_bars.call_count, 1)
        rule, fresh, limit = self.execution(row)
        with self.assertRaisesRegex(ValueError, "ID|검증 연결"):
            self.feed.validate_execution(self.item, rule, fresh, limit)
        self.service.mode = TradingMode.REAL
        self.feed.publish(self.chart(export_id="export-3"))
        self.assertEqual(self.signal()["action"], "hold")
        self.assertFalse(self.feed._ready)
        self.service.submit.assert_not_called()

    def test_changed_manifest_or_weights_block_execution(self):
        for changed_path, changed_bytes in ((self.manifest_path, b"changed manifest"),
                                            (self.weight_path, b"changed weights")):
            with self.subTest(path=changed_path.name):
                self.feed.publish(self.chart(export_id=changed_path.name))
                row = self.signal()
                self.assertEqual(row["action"], "buy")
                rule, fresh, limit = self.execution(row)
                original = changed_path.read_bytes()
                changed_path.write_bytes(changed_bytes)
                with self.assertRaisesRegex(ValueError, "번들이 변경"):
                    self.feed.validate_execution(self.item, rule, fresh, limit)
                changed_path.write_bytes(original)

    def test_execution_rejects_stale_quote_new_slot_price_gap_and_wrong_signal(self):
        self.feed.publish(self.chart())
        row = self.signal()
        rule, fresh, limit = self.execution(row)
        with self.assertRaisesRegex(ValueError, "시세·시간"):
            self.feed.validate_execution(
                self.item, rule,
                SimpleNamespace(fetched_at=NOW - timedelta(seconds=16), quote=fresh.quote), limit)
        with self.assertRaisesRegex(ValueError, "가격 범위"):
            self.feed.validate_execution(self.item, rule, fresh, Decimal("102"))
        with self.assertRaisesRegex(ValueError, "가격 범위"):
            self.feed.validate_execution(
                self.item, rule,
                SimpleNamespace(fetched_at=NOW, quote=SimpleNamespace(price=Decimal("106"))), limit)
        self.now = NOW + timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "새 완료 5분봉"):
            self.feed.validate_execution(
                self.item, rule,
                SimpleNamespace(fetched_at=self.now, quote=fresh.quote), limit)
        self.now = NOW
        altered = {**row, "model_manifest_sha256": "0" * 64}
        rule, fresh, limit = self.execution(altered)
        with self.assertRaisesRegex(ValueError, "번들·신호 ID"):
            self.feed.validate_execution(self.item, rule, fresh, limit)

    def test_hedge_research_emits_hold_and_cannot_validate_order(self):
        model_id = MINUTE_HEDGE_IDS[0]
        policy = ExternalPolicy(MINUTE_RESEARCH_BY_ID[model_id].source_id,
                                1, Decimal("10000"), Decimal(0))
        feed = MinuteResearchHoldFeed(self.window, model_id, policy,
                                      self.root / "hedge-signal.json")
        feed.publish(self.chart())
        payload = read_signal_file(feed.output_path)
        self.assertEqual(payload["source_id"], policy.source_id)
        self.assertEqual(payload["signals"][0]["action"], "hold")
        self.assertNotIn("quantity", payload["signals"][0])
        self.assertEqual(feed.diagnostics[WATCH_ID]["reason"], "HEDGE_RESEARCH_ORDER_HOLD")
        with self.assertRaisesRegex(ValueError, "주문이 허용되지"):
            feed.validate_execution(None, None, None, None)
        self.minute_feed.get_complete_bars.assert_not_called()
        self.service.broker.assert_not_called()
        self.service.submit.assert_not_called()


class MinuteLotExitBridgeTests(unittest.TestCase):
    """Use a temporary durable lot, then run the real AutoTrader exit selector."""

    def _filled_lot(self, directory, model_id, fill):
        store = WatchStore(Path(directory) / "lot-ledger.sqlite3")
        item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "offline")
        store.save_item(item)
        rule = TriggerRule("offline-buy-" + model_id, item.id, TriggerKind.EXTERNAL,
                           OrderSide.BUY, 1, Decimal("10000"))
        store.add_rule(rule)
        source = MINUTE_RESEARCH_BY_ID[model_id].source_id
        signal_id = model_id + ":offline-fill"
        payload = {"signal_id": signal_id, "strategy_id": model_id,
                   "market": "domestic", "exchange": "KRX", "symbol": "005930",
                   "action": "buy"}
        with store.connection() as db:
            db.execute("INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (source, signal_id, json.dumps(payload), rule.id, item.id,
                        (fill - timedelta(minutes=1)).isoformat(),
                        (fill + timedelta(hours=1)).isoformat(), "offline-export",
                        (fill - timedelta(minutes=1)).isoformat(), "buy", "ready"))
        self.assertTrue(store.claim(rule, Decimal("100"),
                                    fill - timedelta(minutes=1), prototype_lots=True))
        store.finish(rule.id, "accepted", "offline acknowledgment", "OFFLINE-ORDER")
        store.record_execution(rule.id, filled_quantity=Decimal(1),
                               remaining_quantity=Decimal(0), fill_price=Decimal("100"),
                               observed_at=fill)
        store.finish(rule.id, "filled", "offline fill")
        inventory = store.prototype_inventory(
            item.id, broker_quantity=Decimal(1), broker_sellable=Decimal(1))
        self.assertTrue(inventory["reconciled"])
        lot, = inventory["lots"]
        self.assertEqual(lot["buy_fill_observed_at"], fill.isoformat())
        self.assertEqual((lot["take_profit_price"], lot["stop_loss_price"]),
                         (Decimal("103.00"), Decimal("98.00")))
        return store, item, inventory, lot

    def test_all_nine_models_use_own_fill_time_tp_sl_and_holdings_sell_route(self):
        session = session_on(Market.DOMESTIC, NOW.date())
        self.assertIsNotNone(session)
        fill = session.opened + timedelta(minutes=30)
        for model_id in MINUTE_TRANSFER_IDS:
            minutes = MINUTE_RESEARCH_BY_ID[model_id].horizon_bars * 5
            with self.subTest(model_id=model_id), TemporaryDirectory() as directory:
                store, item, inventory, lot = self._filled_lot(directory, model_id, fill)
                service = SimpleNamespace(
                    mode=TradingMode.DEMO,
                    submit=Mock(side_effect=AssertionError("unexpected broker order")))
                clock = [fill + timedelta(minutes=1)]
                engine = AutoTrader(service, store, clock=lambda: clock[0])
                engine._armed.set()  # Isolated exit selector only; _execute is fake.
                engine.prototype_lots_enabled = True
                position = Position(
                    Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
                    Decimal(1), Decimal(1), Decimal("100"), Decimal("100"),
                    Decimal("100"), Decimal(0), Decimal(0))
                targets = engine.holding_exit_targets(position)
                self.assertTrue(targets["reconciled"])
                self.assertFalse(timed_exit_due(
                    lot, Market.DOMESTIC, fill + timedelta(minutes=minutes - 1)))
                self.assertTrue(timed_exit_due(
                    lot, Market.DOMESTIC, fill + timedelta(minutes=minutes)))
                with patch.object(engine, "_execute", return_value=True) as execute:
                    for price, expected in (("100", None),
                                            ("103", TriggerKind.PRICE_GE),
                                            ("98", TriggerKind.PRICE_LE)):
                        snapshot = SimpleNamespace(quote=SimpleNamespace(price=Decimal(price)))
                        engine._lot_holdings_exits(item, position, snapshot, targets, set())
                        if expected is None:
                            execute.assert_not_called()
                        else:
                            sent_rule = execute.call_args.args[1]
                            self.assertEqual((sent_rule.kind, sent_rule.side, sent_rule.quantity),
                                             (expected, OrderSide.SELL, 1))
                            self.assertEqual(store.prototype_sell_allocation(
                                sent_rule.id)["lot_id"], lot["lot_id"])
                            execute.reset_mock()
                    clock[0] = fill + timedelta(minutes=minutes)
                    snapshot = SimpleNamespace(quote=SimpleNamespace(price=Decimal("100")))
                    engine._lot_holdings_exits(item, position, snapshot, targets, set())
                    timed_rule = execute.call_args.args[1]
                    self.assertEqual((timed_rule.kind, timed_rule.side),
                                     (TriggerKind.TIME_EXIT, OrderSide.SELL))
                    self.assertEqual(store.prototype_sell_allocation(
                        timed_rule.id)["lot_id"], lot["lot_id"])
                service.submit.assert_not_called()

    def test_preclose_escape_uses_same_lot_route_before_long_horizon(self):
        session = session_on(Market.DOMESTIC, NOW.date())
        fill = session.closed - timedelta(minutes=20)
        with TemporaryDirectory() as directory:
            store, item, inventory, lot = self._filled_lot(
                directory, MINUTE_TRANSFER_IDS[-1], fill)
            now = session.closed - timedelta(minutes=4)
            self.assertTrue(timed_exit_due(lot, Market.DOMESTIC, now))
            service = SimpleNamespace(mode=TradingMode.DEMO,
                                      submit=Mock(side_effect=AssertionError("unexpected order")))
            engine = AutoTrader(service, store, clock=lambda: now)
            engine._armed.set()
            engine.prototype_lots_enabled = True
            position = Position(Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
                                Decimal(1), Decimal(1), Decimal("100"), Decimal("100"),
                                Decimal("100"), Decimal(0), Decimal(0))
            targets = engine.holding_exit_targets(position)
            snapshot = SimpleNamespace(quote=SimpleNamespace(price=Decimal("100")))
            with patch.object(engine, "_execute", return_value=True) as execute:
                engine._lot_holdings_exits(item, position, snapshot, targets, set())
            self.assertEqual(execute.call_args.args[1].kind, TriggerKind.TIME_EXIT)
            service.submit.assert_not_called()

    def test_confirmed_minute_lot_reaches_fake_broker_sell_and_fill(self):
        session = session_on(Market.DOMESTIC, NOW.date())
        fill = session.opened + timedelta(minutes=30)
        due = fill + timedelta(minutes=15)
        with TemporaryDirectory() as directory:
            store, item, _, lot = self._filled_lot(
                directory, MINUTE_TRANSFER_IDS[0], fill)
            service = FakeTradingService()
            service.submit = Mock(wraps=service.submit)
            service.positions = (Position(
                Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
                Decimal(1), Decimal(1), Decimal("100"), Decimal("100"),
                Decimal("100"), Decimal(0), Decimal(0)),)
            engine = AutoTrader(service, store, clock=lambda: due)
            engine.prototype_lots_enabled = True
            engine.enable_holdings_exits = True
            engine._armed.set()
            position = service.positions[0]
            targets = engine.holding_exit_targets(position)
            quote = Quote(Market.DOMESTIC, "005930", "offline", "KRX",
                          Decimal("100"), "KRW")
            snapshot = MarketSnapshot(
                quote, DailyHistory(Market.DOMESTIC, "005930", "KRX", "KRW", 0, ()), due)
            engine._lot_holdings_exits(item, position, snapshot, targets, set())
            service.submit.assert_called_once()
            request = service.submit.call_args.args[0]
            self.assertEqual((request.market, request.side, request.symbol,
                              request.exchange, request.quantity, request.price),
                             (Market.DOMESTIC, OrderSide.SELL, "005930", "KRX", 1,
                              Decimal("100")))
            accepted = [row for row in store.order_history(limit=None, watch_id=item.id)
                        if row["side"] == "sell"]
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["status"], "accepted")
            self.assertEqual(accepted[0]["prototype_lot_id"], lot["lot_id"])
            store.record_execution(accepted[0]["rule_id"], filled_quantity=Decimal(1),
                                   remaining_quantity=Decimal(0), fill_price=Decimal("100"),
                                   observed_at=due + timedelta(minutes=1))
            store.finish(accepted[0]["rule_id"], "filled", "offline sell fill")
            closed = store.prototype_inventory(item.id, broker_quantity=Decimal(0),
                                               broker_sellable=Decimal(0))
            self.assertTrue(closed["reconciled"])
            self.assertEqual(closed["lots"][0]["quantity_remaining"], 0)

    def test_confirmed_demo_lot_can_sell_one_share_above_order_cap(self):
        session = session_on(Market.DOMESTIC, NOW.date())
        fill = session.opened + timedelta(minutes=30)
        now = fill + timedelta(minutes=1)
        with TemporaryDirectory() as directory:
            store, item, _, lot = self._filled_lot(
                directory, MINUTE_TRANSFER_IDS[0], fill)
            service = FakeTradingService()
            service.prices = [Decimal("104")]
            service.positions = (Position(
                Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
                Decimal(1), Decimal(1), Decimal("100"), Decimal("103"),
                Decimal("103"), Decimal("3"), Decimal("3")),)
            engine = AutoTrader(service, store, clock=lambda: now)
            engine.prototype_lots_enabled = True
            engine.enable_holdings_exits = True
            engine.holding_caps[Market.DOMESTIC] = Decimal("99")
            engine._armed.set()
            position = service.positions[0]
            snapshot = MarketSnapshot(
                Quote(Market.DOMESTIC, "005930", "offline", "KRX",
                      Decimal("103"), "KRW"),
                DailyHistory(Market.DOMESTIC, "005930", "KRX", "KRW", 0, ()), now)
            engine._lot_holdings_exits(
                item, position, snapshot, engine.holding_exit_targets(position), set())
            self.assertEqual(len(service.submitted), 1, store.events())
            request = service.submitted[0]
            self.assertEqual((request.side, request.quantity, request.price),
                             (OrderSide.SELL, 1, Decimal("104")))
            sale, = [row for row in store.order_history(limit=None) if row["side"] == "sell"]
            self.assertEqual((sale["status"], sale["prototype_lot_id"]),
                             ("accepted", lot["lot_id"]))
            self.assertEqual(engine.holding_caps[Market.DOMESTIC], Decimal("99"))

    def test_unconfirmed_manual_holding_cannot_use_one_share_cap_exception(self):
        session = session_on(Market.DOMESTIC, NOW.date())
        now = session.opened + timedelta(minutes=30)
        with TemporaryDirectory() as directory:
            store = WatchStore(Path(directory) / "manual-ledger.sqlite3")
            service = FakeTradingService()
            service.prices = [Decimal("103")]
            service.positions = (Position(
                Market.DOMESTIC, "005930", "offline", "KRX", "KRW",
                Decimal(1), Decimal(1), Decimal("100"), Decimal("103"),
                Decimal("103"), Decimal("3"), Decimal("3")),)
            engine = AutoTrader(service, store, clock=lambda: now)
            engine.prototype_lots_enabled = True
            engine.enable_holdings_exits = True
            engine.holding_caps[Market.DOMESTIC] = Decimal("99")
            engine._armed.set()
            engine._holdings_pass(set())
            self.assertEqual(service.submitted, [])
            self.assertEqual(store.order_history(limit=None), ())


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class MinuteGuiRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selected_models_route_to_shared_minute_feed_and_hold_only_research(self):
        with TemporaryDirectory() as directory, patch(
                "dockdack.gui_service.KiwoomConfig.from_env",
                side_effect=AssertionError("unexpected credentials")) as credentials:
            root = Path(directory)
            store = WatchStore(root / "ledger.sqlite3")
            service = FakeTradingService()
            service.broker = Mock(return_value=SimpleNamespace(mode=TradingMode.DEMO))
            item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "offline")
            store.save_item(item)
            window = V00Window(service, store, builtin=False)
            try:
                for timer in window.findChildren(QTimer):
                    timer.stop()
                selected = (*MINUTE_TRANSFER_IDS[:2], MINUTE_HEDGE_IDS[0])
                for model_id in selected:
                    checkbox = window.external_model_checks[model_id]
                    checkbox.blockSignals(True)
                    checkbox.setChecked(True)
                    checkbox.blockSignals(False)
                window.configure_external()
                transfer_feeds = [window._prototype_feeds[model_id]
                                  for model_id in selected[:2]]
                hedge_feed = window._prototype_feeds[selected[2]]
                self.assertTrue(all(isinstance(feed, DailyProxyMinuteFeed)
                                    for feed in transfer_feeds))
                self.assertIs(transfer_feeds[0].minute_feed,
                              transfer_feeds[1].minute_feed)
                self.assertIsInstance(hedge_feed, MinuteResearchHoldFeed)
                self.assertEqual(set(window.engine.external_sources),
                                 {window.engine.external_policy.source_id}
                                 | {MINUTE_RESEARCH_BY_ID[model_id].source_id
                                    for model_id in selected})
                self.assertEqual(set(window.engine.source_validators),
                                 {MINUTE_RESEARCH_BY_ID[model_id].source_id
                                  for model_id in selected})
                self.assertFalse(window.engine.orders_enabled)
                self.assertFalse(window.monitoring)
                self.assertEqual(service.submitted, [])
                self.assertEqual(service.quote_calls, 0)
                credentials.assert_not_called()

                # GUI verdicts must be tied to the quote used for this one
                # completed minute decision, including the compact summary.
                window.engine.clock = lambda: NOW
                window.snapshots[item.id] = MarketSnapshot(
                    Quote(Market.DOMESTIC, "005930", "offline", "KRX",
                          Decimal("100"), "KRW"),
                    service.history(item.instrument, 31), NOW)
                window.fresh_ids.add(item.id)
                window._items_by_id = {item.id: item}
                window._last_queried_watch_id = item.id
                transfer_feeds[0]._ready = True
                transfer_feeds[0].diagnostics[item.id] = {
                    "reason": "MINUTE_BUY_CANDIDATE", "score": .8,
                    "_display_quote_fetched_at": NOW.isoformat(),
                    "_display_price": "100"}
                self.assertEqual(window._minute_buy_display(selected[0], item),
                                 "매수 후보 · 점수 0.800")
                self.assertEqual(window._minute_buy_display(selected[1], item),
                                 "5분봉 판단 대기")
                self.assertEqual(window._minute_buy_display(selected[2], item),
                                 "연구 전용 · 주문 보류")
                window._update_latest_model_summary()
                self.assertIn("5분봉 판단 대기 1개", window.latest_model_summary.text())
                transfer_feeds[1]._ready = True
                transfer_feeds[1].diagnostics[item.id] = {
                    "reason": "MINUTE_THRESHOLD_NOT_MET", "score": .2,
                    "_display_quote_fetched_at": NOW.isoformat(),
                    "_display_price": "100"}
                window._update_latest_model_summary()
                self.assertIn("5분봉 매수 후보 1/2", window.latest_model_summary.text())
                self.assertIn("인버스 연구 주문 보류", window.latest_model_summary.text())
                window.engine.clock = lambda: NOW + timedelta(seconds=16)
                self.assertEqual(window._minute_buy_display(selected[0], item),
                                 "5분봉 판단 대기")
                self.assertEqual(service.submitted, [])
            finally:
                window.close()
                for name in ("pool", "inspection_pool", "activity_pool"):
                    pool = getattr(window, name, None)
                    if pool is not None:
                        pool.waitForDone(5000)
                window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
