from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.models import Market, OrderSide
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, atomic_json, export_charts, ingest_signals, read_signal_file
from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, WatchStore
from test_autotrade import FakeTradingService, NOW, position


class SignalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.store = WatchStore(self.path / "watch.sqlite3")
        self.service = FakeTradingService()
        self.item = WatchItem(self.service.resolve("005930"), "삼성전자")
        self.store.save_item(self.item)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.snapshot(self.item)
        self.export = export_charts(self.store, self.path / "charts.json", now=NOW)
        self.policy = ExternalPolicy("external-model", 10, Decimal(1000), Decimal(1000))
        self.engine.external_policy = self.policy
        self.engine.external_only = True

    def payload(self, action="buy", **changes):
        signal = dict(signal_id="decision-1", export_id=self.export["export_id"],
                      market="domestic", symbol="005930", exchange="KRX", action=action,
                      generated_at=NOW.isoformat(), expires_at=(NOW + timedelta(minutes=2)).isoformat())
        if action != "hold":
            signal.update(quantity=1, max_notional="500")
        signal.update(changes)
        return dict(schema_version=1, source_id="external-model", signals=[signal])

    def ingest(self, payload=None, **kwargs):
        return ingest_signals(self.store, payload or self.payload(), self.policy, now=kwargs.pop("now", NOW), **kwargs)

    def arm(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def test_export_is_versioned_decimal_strings_with_rank_and_bar_metadata(self):
        data = json.loads((self.path / "charts.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["scope"], "watchlist")
        self.assertTrue(data["completed_bars_persisted"])
        self.assertEqual(data["intraday_history_refresh_seconds"], 300)
        row = data["stocks"][0]
        self.assertEqual(row["history_fetched_at"], NOW.isoformat())
        self.assertEqual(row["price"], "100")
        self.assertEqual(row["available_days"], 30)
        self.assertTrue(row["complete"])
        self.assertFalse(row["quote_stale"])
        self.assertEqual(row["bars"][0]["volume"], "1234")
        self.assertFalse(row["bars"][-1]["is_current_day"])
        self.assertEqual(list(self.path.glob("*.tmp")), [])

    def test_missing_failed_and_old_data_cannot_be_signal_members(self):
        self.store.save_item(WatchItem(self.service.resolve("AAPL")))
        data = export_charts(self.store, self.path / "partial.json", now=NOW, errors={self.item.id: "API failed"})
        self.assertEqual([r["status"] for r in data["stocks"]], ["error", "missing"])
        with self.assertRaises(ValueError):
            self.ingest(self.payload(export_id=data["export_id"]))
        old = export_charts(self.store, self.path / "old.json", now=NOW + timedelta(days=2))
        self.assertEqual(old["stocks"][0]["status"], "error")

    def test_ingestion_is_data_only_and_same_id_is_durable_across_restart(self):
        self.assertEqual(self.ingest()["queued"], 1)
        self.assertEqual(self.service.submitted, [])
        self.assertFalse(self.engine.orders_enabled)
        self.store = WatchStore(self.store.path)
        self.assertEqual(self.ingest()["duplicates"], 1)
        self.assertEqual(len(self.store.rules()), 1)

    def test_changed_payload_with_same_id_is_rejected(self):
        self.ingest()
        with self.assertRaises(ValueError):
            self.ingest(self.payload(quantity=2))
        self.assertEqual(len(self.store.rules()), 1)

    def test_bad_batch_rolls_back_every_signal(self):
        us = WatchItem(self.service.resolve("AAPL"))
        self.store.save_item(us)
        second = self.payload(symbol="AAPL", market="us", exchange="ND", signal_id="us-1")["signals"][0]
        payload = self.payload()
        payload["signals"].append(second)  # No US snapshot in this export.
        with self.assertRaises(ValueError):
            self.ingest(payload)
        self.assertEqual(self.store.rules(), ())

    def test_invalid_schema_values_fail_closed(self):
        cases = [dict(quantity=True), dict(quantity=0), dict(quantity=11), dict(max_notional=500),
                 dict(max_notional="NaN"), dict(max_notional="1001"), dict(max_notional="-1"),
                 dict(exchange="ND"), dict(symbol="000660"), dict(export_id="unknown"),
                 dict(action="execute"), dict(generated_at=NOW.replace(tzinfo=None).isoformat()),
                 dict(expires_at=(NOW + timedelta(minutes=11)).isoformat()),
                 dict(generated_at=(NOW + timedelta(seconds=10)).isoformat())]
        for change in cases:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                self.ingest(self.payload(**change))
        self.assertEqual(self.store.rules(), ())

    def test_unknown_source_and_zero_market_cap_block_orders_but_allow_hold(self):
        self.policy = ExternalPolicy("external-model", 1, Decimal(0), Decimal(0))
        with self.assertRaises(ValueError):
            self.ingest()
        self.assertEqual(self.ingest(self.payload("hold"))["hold"], 1)
        data = self.payload()
        data["source_id"] = "untrusted"
        with self.assertRaises(ValueError):
            self.ingest(data)

    def test_expired_and_out_of_order_signals_do_not_create_rules(self):
        data = self.payload(generated_at=(NOW - timedelta(minutes=3)).isoformat(), expires_at=(NOW - timedelta(minutes=1)).isoformat())
        # An older valid export is necessary to describe this older decision.
        with self.store.connection() as db:
            db.execute("UPDATE chart_exports SET created_at=?", ((NOW - timedelta(minutes=4)).isoformat(),))
        self.assertEqual(self.ingest(data)["expired"], 1)
        self.assertEqual(self.store.rules(), ())
        self.assertEqual(self.ingest(self.payload("hold", signal_id="new"))["hold"], 1)
        self.assertEqual(self.ingest(self.payload(signal_id="older", generated_at=(NOW - timedelta(seconds=1)).isoformat()))["expired"], 1)

    def test_new_hold_supersedes_ready_order_without_cancelling_sent_orders(self):
        self.ingest()
        later = NOW + timedelta(seconds=1)
        self.ingest(self.payload("hold", signal_id="hold-2", generated_at=later.isoformat()), now=later)
        self.assertEqual(self.store.rules()[0].status, "superseded")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])

    def test_external_buy_and_sell_use_fresh_limit_once(self):
        for action in ("buy", "sell"):
            with self.subTest(action=action):
                if action == "sell":
                    self.store.mark_reviewed(self.store.rules()[0].id, "CHECKED_ORDER_HISTORY")
                    self.service.positions = (position(),)
                later = NOW + timedelta(seconds=(action == "sell"))
                self.ingest(self.payload(action, signal_id=action, generated_at=later.isoformat()), now=later)
                self.arm()
                self.engine.poll()
                self.engine.poll()
                self.assertEqual(self.service.submitted[-1].side, OrderSide(action))
                self.assertEqual(self.service.submitted[-1].price, Decimal(100))
                self.assertEqual(self.service.submitted[-1].order_type, "0")
        self.assertEqual(len(self.service.submitted), 2)

    def test_expiration_during_preflight_never_sends(self):
        self.ingest()
        self.arm()
        self.service.on_account = lambda: setattr(self.engine, "clock", lambda: NOW + timedelta(minutes=3))
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.rules()[0].status, "expired")

    def test_policy_rechecked_and_other_mode_rules_never_run(self):
        self.ingest()
        self.engine.external_policy = ExternalPolicy("external-model", 1, Decimal(99), Decimal(0))
        self.arm()
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        manual = TriggerRule.create(self.item, "price_ge", "buy", 1, Decimal(1000), Decimal(90))
        self.store.add_rule(manual)
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.engine.external_only = False
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.rules()[0].status, "ready")

    def test_file_reader_atomic_duplicates_and_hold_during_preflight(self):
        path = self.path / "signals.json"
        self.engine.external_reader = SignalFileReader(self.store, path, self.policy, lambda: NOW)
        self.assertIsNone(self.engine.external_reader())
        atomic_json(path, self.payload())
        self.arm()
        def publish_hold():
            atomic_json(path, self.payload("hold", signal_id="hold-2", generated_at=(NOW + timedelta(seconds=1)).isoformat()))
        self.service.on_account = publish_hold
        self.engine.poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.rules()[0].status, "superseded")
        self.assertIsNone(self.engine.external_reader())

    def test_malformed_file_disarms_previously_queued_orders(self):
        path = self.path / "signals.json"
        self.ingest()
        self.engine.external_reader = SignalFileReader(self.store, path, self.policy, lambda: NOW)
        path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        self.arm()
        self.engine.poll()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])
        with self.assertRaises(ValueError):
            read_signal_file(path)

    def test_new_signal_is_prioritized_between_watchlist_requests(self):
        items = [WatchItem(self.service.resolve(symbol)) for symbol in ("000660", "035420")]
        for item in items:
            self.store.save_item(item)
            self.engine.snapshot(item)
        self.export = export_charts(self.store, self.path / "charts.json", now=NOW)
        visited = []
        def progress(data):
            visited.append(data[0])
            if len(visited) == 1:
                self.ingest(self.payload(symbol="035420"))
        self.engine.poll(progress=progress)
        self.assertEqual(visited, [self.item.id, items[1].id, items[0].id])

    def test_unknown_submission_disarms_and_never_retries_signal(self):
        self.ingest()
        self.service.submit_error = TimeoutError("timeout")
        self.arm()
        self.engine.poll()
        self.assertEqual(self.store.rules()[0].status, "unknown")
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.ingest()["duplicates"], 1)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_v1_migration_preserves_watchlist_and_rules(self):
        self.ingest()
        with self.store.connection() as db:
            db.execute("PRAGMA user_version=1")
        migrated = WatchStore(self.store.path)
        self.assertEqual(migrated.items(), self.store.items())
        self.assertEqual(migrated.rules(), self.store.rules())
        with migrated.connection() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_external_model_adapter_is_hold_by_default_and_retry_is_stable(self):
        from examples.external_signal_producer import build_signals
        kwargs = dict(source_id="external-model", decision_id="run-1", generated_at=NOW)
        hold = build_signals(self.export, {}, **kwargs)
        self.assertEqual(hold["signals"][0]["action"], "hold")
        self.assertEqual(self.ingest(hold)["hold"], 1)
        self.assertEqual(build_signals(self.export, {}, **kwargs), hold)
        buy = build_signals(self.export, {self.item.id: {"action": "buy", "quantity": 1, "max_notional": "500"}},
                            source_id="external-model", decision_id="run-2", generated_at=NOW+timedelta(seconds=1))
        self.assertEqual(self.ingest(buy, now=NOW+timedelta(seconds=1))["queued"], 1)
        self.assertEqual(self.service.submitted, [])

    def test_export_cannot_overwrite_database(self):
        with self.assertRaises(ValueError):
            export_charts(self.store, self.store.path, now=NOW)
        self.assertEqual(self.store.items(), (self.item,))

    def test_signal_file_rejects_nan_oversize_and_unknown_fields(self):
        path = self.path / "signals.json"
        for text in ('{"schema_version":NaN}', ' ' * 2_000_001):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_signal_file(path)
        payload = self.payload()
        payload["code"] = "not executable"
        with self.assertRaises(ValueError):
            self.ingest(payload)

    def test_two_process_equivalents_share_one_durable_signal(self):
        from concurrent.futures import ThreadPoolExecutor
        def receive(_):
            return ingest_signals(WatchStore(self.store.path), self.payload(), self.policy, now=NOW)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(receive, range(2)))
        self.assertEqual(sum(r["queued"] for r in results), 1)
        self.assertEqual(sum(r["duplicates"] for r in results), 1)
        self.assertEqual(len(self.store.rules()), 1)


if __name__ == "__main__":
    unittest.main()
