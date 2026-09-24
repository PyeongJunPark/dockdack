"""Concurrent external prototypes against temporary ledgers and fake brokers only."""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dockdack.mark1_trigger import Mark1TriggerBridge
from dockdack.autotrade import AutoTrader
from dockdack.exceptions import BrokerAPIError
from dockdack.models import Market, OpenOrder
from dockdack.signal_bridge import (
    ExternalPolicy, SignalFileReader, atomic_json, export_charts,
    prototype_order_label,
)
from dockdack.watchlist import WatchItem, WatchStore
from test_autotrade import position
from test_mark1_1_trigger import new_predictor
import test_mark1_paper_execution as paper_tests


MARK1 = "mark1-prototype-demo-trigger"
MARK11 = "mark1-1-prototype-demo-trigger"
FAMILIES = {
    MARK1: ("mark1-prototype", "mark1 prototype"),
    MARK11: ("mark1-1-prototype", "mark1.1 prototype"),
}


class ConcurrentPrototypeFeedTests(unittest.TestCase):
    setUp = paper_tests.Mark1PaperExecutionTests.setUp

    def configure(self, *, primary=True):
        self.policies = {
            source: ExternalPolicy(source, 1, Decimal("500"), Decimal("1000"))
            for source in FAMILIES
        }
        self.files = {source: self.path / (source + ".json") for source in FAMILIES}
        self.readers = {
            source: SignalFileReader(self.store, self.files[source], policy, lambda: self.now)
            for source, policy in self.policies.items()
        }
        self.seen = {source: [] for source in FAMILIES}

        def validator(source):
            def check(item, rule, fresh, actual_limit_price, *, stage):
                record = self.store.external_for_rule(rule.id)
                if record["source_id"] != source:
                    raise ValueError("wrong model execution callback")
                self.seen[source].append((rule.id, stage))
            return check

        self.engine.isolated_symbol_errors = True
        self.engine.external_policy = self.policies[MARK1] if primary else None
        self.engine.external_reader = self.readers[MARK1] if primary else None
        extra = (MARK11,) if primary else (MARK1, MARK11)
        self.engine.configure_external_sources([(self.policies[source], self.readers[source]) for source in extra])
        self.engine.configure_source_validators({source: validator(source) for source in FAMILIES})

    def payload(self, source, *, symbol="005930", action="buy", suffix="first", notional="500"):
        strategy, title = FAMILIES[source]
        row = {"signal_id": f"{strategy}:{symbol}:{suffix}", "export_id": self.chart["export_id"],
               "market": "domestic", "symbol": symbol, "exchange": "KRX", "action": action,
               "generated_at": self.now.isoformat(),
               "expires_at": (self.now + timedelta(minutes=2)).isoformat()}
        if action == "buy":
            row.update(quantity=1, max_notional=notional, strategy_id=strategy,
                       model_title=title, model_version="test-fixture", model_manifest_sha256="a" * 64)
        return {"schema_version": 1, "trading_mode": "demo", "source_id": source, "signals": [row]}

    def publish(self, source, **kwargs):
        atomic_json(self.files[source], self.payload(source, **kwargs))

    def rules_for(self, source):
        return tuple(rule for rule in self.store.rules(include_inactive=True)
                     if self.store.external_for_rule(rule.id)["source_id"] == source)

    def arm_poll(self):
        self.engine.enable_orders("DEMO_AUTOTRADE")
        return self.engine.poll()

    def test_both_connected_does_not_arm_or_send(self):
        self.configure(primary=False)
        for source in FAMILIES:
            self.publish(source)
        self.engine.poll()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(len(self.store.rules(statuses=("ready",))), 2)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts(), ())

    def test_same_symbol_two_buy_feeds_submit_once_and_keep_winning_model(self):
        self.configure(primary=False)
        for source in FAMILIES:
            self.publish(source)
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        accepted = [rule for rule in self.store.rules() if rule.status == "accepted"]
        self.assertEqual(len(accepted), 1)
        record = self.store.external_for_rule(accepted[0].id)
        winner = record["source_id"]
        other = MARK11 if winner == MARK1 else MARK1
        self.assertEqual(self.seen[winner], [(accepted[0].id, "preflight"), (accepted[0].id, "final_send")])
        self.assertEqual(self.seen[other], [])
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], winner)
        self.assertEqual(prototype_order_label(self.store.order_history()[0]), FAMILIES[winner][1])
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1, "The other model must not buy while first order is pending")

    def test_secondary_only_buy_uses_secondary_validator_and_bracket(self):
        self.configure()
        self.publish(MARK1, action="hold")
        self.publish(MARK11)
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.seen[MARK1], [])
        self.assertEqual([stage for _, stage in self.seen[MARK11]], ["preflight", "final_send"])
        targets = self.store.exit_targets(self.item.id)
        self.assertEqual((targets["take_profit_price"], targets["stop_loss_price"]),
                         (Decimal("100.5"), Decimal("99.6")))

    def test_one_source_hold_does_not_supersede_other_source_buy(self):
        self.configure()
        self.publish(MARK1)
        self.publish(MARK11, action="hold")
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], MARK1)

    def test_failed_primary_reader_does_not_disarm_healthy_secondary(self):
        self.configure()
        self.engine.external_reader = Mock(side_effect=ValueError("primary model unavailable"))
        self.publish(MARK11)
        self.arm_poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertIn(MARK1, self.engine.external_source_errors)
        self.assertNotIn(MARK11, self.engine.external_source_errors)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], MARK11)

    def test_failed_secondary_reader_does_not_disarm_healthy_primary(self):
        self.configure()
        self.engine.configure_external_sources([(self.policies[MARK11], Mock(side_effect=ValueError("secondary unavailable")))])
        self.publish(MARK1)
        self.arm_poll()
        self.assertTrue(self.engine.orders_enabled)
        self.assertIn(MARK11, self.engine.external_source_errors)
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], MARK1)

    def test_error_quarantines_previously_received_source_but_not_other(self):
        self.configure()
        for source in FAMILIES:
            self.publish(source)
        self.engine._read_external()
        self.engine.external_reader = Mock(side_effect=ValueError("failed after publishing"))
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.store.exit_targets(self.item.id)["source"], MARK11)
        self.assertEqual(self.seen[MARK1], [])

    def test_removed_secondary_policy_blocks_its_already_received_rule(self):
        self.configure()
        self.publish(MARK11)
        self.engine._read_external()
        rule = self.rules_for(MARK11)[0]
        self.engine.configure_external_sources([])
        self.engine.configure_source_validators({MARK1: lambda *args, **kwargs: None})
        with self.assertRaises(ValueError):
            self.engine._validate_rule(rule)
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])

    def test_disabled_primary_cannot_fall_through_to_other_model_policy(self):
        self.configure()
        self.publish(MARK1)
        self.engine._read_external()
        rule = self.rules_for(MARK1)[0]
        self.engine.external_policy = None
        self.engine.external_reader = None
        self.engine.configure_source_validators({MARK11: lambda *args, **kwargs: None})
        with self.assertRaises(ValueError):
            self.engine._validate_rule(rule)
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])

    def test_secondary_uses_own_tighter_policy_not_primary_allowance(self):
        self.configure()
        self.publish(MARK11)
        self.engine._read_external()
        tight = replace(self.policies[MARK11], max_krw=Decimal("99"))
        self.engine.configure_external_sources([(tight, lambda: None)])
        rule = self.rules_for(MARK11)[0]
        self.assertEqual(self.engine._policy_for(rule), tight)
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.seen[MARK11], [])

    def test_changing_connections_or_validators_while_armed_is_rejected(self):
        self.configure(primary=False)
        self.engine.enable_orders("DEMO_AUTOTRADE")
        with self.assertRaises(ValueError):
            self.engine.configure_external_sources([])
        with self.assertRaises(ValueError):
            self.engine.configure_source_validators({})
        self.assertEqual(set(self.engine.external_sources), set(FAMILIES))
        self.assertEqual(set(self.engine.source_validators), set(FAMILIES))

    def test_each_connected_model_requires_execution_validator_before_arm(self):
        self.configure(primary=False)
        self.engine.configure_source_validators({MARK1: lambda *args, **kwargs: None})
        with self.assertRaises(ValueError):
            self.engine.enable_orders("DEMO_AUTOTRADE")
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual(self.service.submitted, [])

    def test_wrong_model_callback_cannot_authorize_secondary_buy(self):
        self.configure()
        self.publish(MARK11)
        model = new_predictor()
        wrong = Mark1TriggerBridge(SimpleNamespace(service=self.service, store=self.store, engine=self.engine),
                                   predictors={"domestic": model})
        self.engine.configure_source_validators({MARK1: lambda *args, **kwargs: None,
                                                 MARK11: wrong.validate_execution})
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])
        model.predict.assert_not_called()

    def test_disabling_new_model_does_not_change_owned_holding_exit(self):
        self.configure()
        self.publish(MARK11)
        self.arm_poll()
        self.engine.disarm()
        self.engine.configure_external_sources([])
        self.engine.configure_source_validators({MARK1: lambda *args, **kwargs: None})
        targets = self.engine.holding_exit_targets(replace(position(), average_price=Decimal("110")))
        self.assertEqual(targets["model_title"], "mark1.1 prototype")
        self.assertEqual((targets["take_profit_price"], targets["stop_loss_price"]),
                         (Decimal("110.550"), Decimal("109.560")))

    def test_reentrant_poll_cannot_execute_competing_source(self):
        self.configure(primary=False)
        for source in FAMILIES:
            self.publish(source)
        nested = []
        self.service.before_guard = lambda: nested.append(self.engine.poll())
        self.arm_poll()
        self.assertEqual(nested, [{}])
        self.assertEqual(len(self.service.submitted), 1)

    def test_second_symbol_rechecks_cash_after_first_model_order(self):
        self.configure(primary=False)
        other = WatchItem(self.service.resolve("000660"), "다른 테스트 종목", 31)
        self.store.save_item(other)
        self.engine.snapshot(other)
        self.chart = export_charts(self.store, self.path / "both-charts.json", now=self.now)
        self.publish(MARK1)
        self.publish(MARK11, symbol="000660")
        self.service.available = Decimal("150")
        self.service.on_submit = lambda: setattr(self.service, "available", Decimal("49"))
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(len(self.store.attempts()), 1)

    def test_secondary_disabled_during_final_pacing_is_not_sent(self):
        self.configure()
        self.publish(MARK11)
        self.service.before_guard = self.engine.external_sources.clear
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.attempts()[0]["status"], "not_sent")


class VirtualPrototypeLotEngineTests(unittest.TestCase):
    """Explicit lot mode permits independent model BUYs in a shared instrument."""
    configure = ConcurrentPrototypeFeedTests.configure
    payload = ConcurrentPrototypeFeedTests.payload
    publish = ConcurrentPrototypeFeedTests.publish
    rules_for = ConcurrentPrototypeFeedTests.rules_for
    arm_poll = ConcurrentPrototypeFeedTests.arm_poll

    def setUp(self):
        paper_tests.Mark1PaperExecutionTests.setUp(self)
        self.configure(primary=False)
        self.engine.prototype_lots_enabled = True
        submit = self.service.submit

        def numbered(request):
            result = submit(request)
            return replace(result, order_number=f"{len(self.service.submitted):06d}")
        self.service.submit = numbered

    def complete_buy(self, source, price="100"):
        rule = next(rule for rule in self.rules_for(source) if rule.status == "accepted")
        self.store.record_execution(rule.id, filled_quantity=Decimal(1), remaining_quantity=Decimal(0),
                                    fill_price=Decimal(price), observed_at=self.now)
        self.store.finish(rule.id, "filled", "fake confirmed fill")
        return rule

    def hold_both(self, *, prices=("100", "100")):
        self.publish(MARK1)
        self.publish(MARK11)
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 2)
        rules = self.complete_buy(MARK1, prices[0]), self.complete_buy(MARK11, prices[1])
        self.service.positions = (replace(position(2, 2), average_price=sum(map(Decimal, prices)) / 2),)
        self.engine.enable_holdings_exits = True
        return rules

    def sells(self):
        return [row for row in self.store.order_history(limit=None) if row["side"] == "sell"]

    def test_two_models_can_buy_same_symbol_before_either_fills(self):
        self.publish(MARK1)
        self.publish(MARK11)
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual({row["external_source_id"] for row in self.store.order_history()}, set(FAMILIES))
        self.assertEqual(len(self.seen[MARK1]), 2)
        self.assertEqual(len(self.seen[MARK11]), 2)
        self.assertEqual(self.store.prototype_inventory(self.item.id)["expected_quantity"], 0)
        self.assertIsNone(self.store.exit_targets(self.item.id), "Acknowledgement must not overwrite a symbol-wide bracket")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)

    def test_same_model_pending_signal_is_not_an_additional_lot(self):
        self.publish(MARK1)
        self.arm_poll()
        self.now += timedelta(seconds=1)
        self.publish(MARK1, suffix="another")
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_other_model_can_buy_already_held_symbol_but_own_model_cannot(self):
        self.publish(MARK1)
        self.arm_poll()
        self.complete_buy(MARK1)
        self.service.positions = (position(),)
        self.now += timedelta(seconds=1)
        self.publish(MARK1, suffix="another")
        self.publish(MARK11)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)
        self.assertEqual(self.store.order_history()[0]["external_source_id"], MARK11)

    def test_unmanaged_broker_shares_are_not_assigned_to_new_model(self):
        self.publish(MARK1)
        self.service.positions = (position(),)
        self.arm_poll()
        self.assertEqual(self.service.submitted, [])

    def test_pending_buy_cash_is_reserved_even_when_broker_cash_is_stale(self):
        self.publish(MARK1)
        self.publish(MARK11)
        self.service.available = Decimal("150")
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertEqual(self.service.available, Decimal("150"))

    def test_new_model_profit_sells_only_its_own_confirmed_lot(self):
        old_buy, new_buy = self.hold_both()
        self.service.prices = [Decimal("100.6")]
        self.engine.poll()
        sells = self.sells()
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0]["quantity"], 1)
        allocation = self.store.prototype_sell_allocation(sells[0]["rule_id"])
        self.assertEqual(allocation["lot_id"], new_buy.id)
        self.assertEqual(allocation["model_title"], "mark1.1 prototype")
        self.engine.poll()
        self.assertEqual(len(self.sells()), 1, "Pending sell reserves that model's shares")

    def test_new_model_stop_does_not_sell_old_models_wider_stop(self):
        old_buy, new_buy = self.hold_both()
        self.service.prices = [Decimal("99.5")]
        self.engine.poll()
        self.assertEqual(len(self.sells()), 1)
        self.assertEqual(self.store.prototype_sell_allocation(self.sells()[0]["rule_id"])["lot_id"], new_buy.id)

    def test_both_models_hit_can_sell_two_separate_lots_without_overselling(self):
        old_buy, new_buy = self.hold_both()
        self.service.prices = [Decimal("101.1")]
        self.engine.poll()
        sells = self.sells()
        self.assertEqual(len(sells), 2)
        self.assertEqual(sum(row["quantity"] for row in sells), 2)
        self.assertEqual({self.store.prototype_sell_allocation(row["rule_id"])["lot_id"] for row in sells},
                         {old_buy.id, new_buy.id})
        self.engine.poll()
        self.assertEqual(len(self.sells()), 2)

    def test_lot_targets_use_individual_fills_not_broker_blended_average(self):
        old_buy, new_buy = self.hold_both(prices=("100", "110"))
        summary = self.engine.holding_exit_targets(self.service.positions[0])
        self.assertTrue(summary["reconciled"])
        by_id = {lot["lot_id"]: lot for lot in summary["lots"]}
        self.assertEqual(by_id[old_buy.id]["take_profit_price"], Decimal("101"))
        self.assertEqual(by_id[new_buy.id]["take_profit_price"], Decimal("110.55"))
        self.assertEqual(by_id[new_buy.id]["average_price"], Decimal("110"))
        self.assertIsNone(summary["take_profit_price"])

    def test_disabling_both_feeds_keeps_independent_owned_exits(self):
        old_buy, new_buy = self.hold_both()
        self.engine.disarm()
        self.engine.configure_external_sources([])
        self.engine.configure_source_validators({})
        # Exits are a GUI holding responsibility, not permission from a BUY feed.
        self.engine.external_only = False
        self.service.prices = [Decimal("100.6")]
        self.arm_poll()
        self.assertEqual(len(self.sells()), 1)
        self.assertEqual(self.store.prototype_sell_allocation(self.sells()[0]["rule_id"])["lot_id"], new_buy.id)

    def test_manual_or_missing_broker_quantity_blocks_all_lot_sells(self):
        self.hold_both()
        self.service.positions = (position(1, 1),)
        self.service.prices = [Decimal("101.1")]
        self.engine.poll()
        self.assertEqual(self.sells(), [])
        summary = self.engine.holding_exit_targets(self.service.positions[0])
        self.assertFalse(summary["reconciled"])
        self.assertTrue(all(lot["take_profit_price"] is None for lot in summary["lots"]))

    def test_completed_sell_releases_only_its_confirmed_quantity(self):
        old_buy, new_buy = self.hold_both()
        self.service.prices = [Decimal("100.6")]
        self.engine.poll()
        sale = self.sells()[0]
        # Unfilled acknowledgement leaves both shares in confirmed inventory.
        self.assertEqual(self.store.prototype_inventory(self.item.id)["expected_quantity"], 2)
        self.store.record_execution(sale["rule_id"], filled_quantity=Decimal(1), remaining_quantity=Decimal(0),
                                    fill_price=Decimal("100.6"), observed_at=self.now)
        self.store.finish(sale["rule_id"], "filled", "fake confirmed sale")
        self.service.positions = (position(),)
        inventory = self.store.prototype_inventory(self.item.id, broker_quantity=Decimal(1), broker_sellable=Decimal(1))
        self.assertTrue(inventory["reconciled"])
        self.assertEqual({lot["lot_id"]: lot["quantity_remaining"] for lot in inventory["lots"]},
                         {old_buy.id: Decimal(1), new_buy.id: Decimal(0)})
        self.engine.poll()
        self.assertEqual(len(self.sells()), 1)

    def test_partial_fill_and_cancel_reserve_only_the_unsold_remainder(self):
        source = MARK11
        policy = replace(self.policies[source], max_quantity=2)
        self.policies[source] = policy
        self.readers[source].policy = policy
        self.engine.configure_external_sources([(self.policies[key], self.readers[key]) for key in FAMILIES])
        payload = self.payload(source)
        payload["signals"][0]["quantity"] = 2
        atomic_json(self.files[source], payload)
        self.arm_poll()
        buy = self.rules_for(source)[0]
        self.store.record_fill_recovery(buy.id, status="enriched", message="fake weighted fills", checked_at=self.now,
            source_api="offline-fixture", price_basis="weighted_fills", filled_quantity=Decimal(2),
            remaining_quantity=Decimal(0), fill_price=Decimal(100),
            price_basis_quantity=Decimal(2), price_basis_price=Decimal(100))
        self.store.finish(buy.id, "filled", "fake confirmed buy")
        self.service.positions = (position(2, 2),)
        self.service.prices = [Decimal("100.6")]
        self.engine.enable_holdings_exits = True
        self.engine.poll()
        sale = self.sells()[0]
        self.assertEqual(sale["quantity"], 2)
        self.store.record_execution(sale["rule_id"], filled_quantity=Decimal(1), remaining_quantity=Decimal(1),
                                    fill_price=Decimal("100.6"), observed_at=self.now)
        self.service.positions = (position(1, 0),)
        lot = self.store.prototype_lots(self.item.id)[0]
        self.assertEqual((lot["quantity_remaining"], lot["quantity_reserved_sell"], lot["available_quantity"]), (1, 1, 0))
        self.engine.poll()
        self.assertEqual(len(self.sells()), 1)
        self.store.record_execution(sale["rule_id"], filled_quantity=Decimal(1), remaining_quantity=Decimal(0),
                                    fill_price=Decimal("100.6"), observed_at=self.now)
        self.store.finish(sale["rule_id"], "cancelled", "fake remaining quantity cancelled")
        self.service.positions = (position(1, 1),)
        self.engine.poll()
        self.assertEqual([row["quantity"] for row in self.sells()], [2, 1])
        lot = self.store.prototype_lots(self.item.id)[0]
        self.assertEqual((lot["quantity_remaining"], lot["quantity_reserved_sell"], lot["available_quantity"]), (1, 1, 0))

    def test_known_other_model_open_buy_is_allowed_but_mismatched_order_is_not(self):
        self.publish(MARK1)
        self.arm_poll()
        accepted = self.store.order_history()[0]
        self.service.open_orders = (OpenOrder(Market.DOMESTIC, accepted["order_number"], "005930", "fixture", "KRX",
            "buy", "accepted", Decimal(1), Decimal(0), Decimal(1), Decimal(100)),)
        self.now += timedelta(seconds=1)
        self.publish(MARK11)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 2)

    def test_other_model_order_number_does_not_allow_wrong_side_open_order(self):
        self.publish(MARK1)
        self.arm_poll()
        accepted = self.store.order_history()[0]
        self.service.open_orders = (OpenOrder(Market.DOMESTIC, accepted["order_number"], "005930", "fixture", "KRX",
            "sell", "accepted", Decimal(1), Decimal(0), Decimal(1), Decimal(100)),)
        self.now += timedelta(seconds=1)
        self.publish(MARK11)
        self.engine.poll()
        self.assertEqual(len(self.service.submitted), 1)

    def test_unrelated_manual_holding_keeps_legacy_targets(self):
        targets = self.engine.holding_exit_targets(position())
        self.assertNotIn("lots", targets)
        self.assertEqual(targets["take_profit_price"], Decimal("101"))
        self.assertEqual(targets["stop_loss_price"], Decimal("99.2"))

    def test_old_model_ownership_marker_without_fills_cannot_use_broker_average(self):
        self.store.set_exit_targets(self.item.id, Decimal(101), Decimal("99.1"), source=MARK1,
                                    rule_id="missing-original-buy", now=self.now)
        targets = self.engine.holding_exit_targets(position())
        self.assertIn("lots", targets)
        self.assertFalse(targets["reconciled"])
        self.assertIsNone(targets["take_profit_price"])

    def test_us_rejected_lot_sell_retry_keeps_original_allocation(self):
        self.store = WatchStore(self.path / "us-lots.sqlite3")
        self.item = WatchItem(self.service.resolve("AAPL"), "미국 테스트", 31)
        self.store.save_item(self.item)
        self.now = self.now.replace(hour=14)
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.engine.external_only = True
        self.engine.snapshot(self.item)
        self.chart = export_charts(self.store, self.path / "us-charts.json", now=self.now)
        self.configure(primary=False)
        self.engine.prototype_lots_enabled = True
        payload = self.payload(MARK1, symbol="AAPL")
        payload["signals"][0].update(market="us", exchange="ND")
        atomic_json(self.files[MARK1], payload)
        self.arm_poll()
        self.assertEqual(len(self.service.submitted), 1)
        buy = self.complete_buy(MARK1)
        self.service.positions = (replace(position(), market=Market.US, symbol="AAPL", exchange="ND", currency="USD"),)
        self.service.prices = [Decimal("101.1")]
        self.engine.enable_holdings_exits = True
        def reject_first_sell():
            self.service.submit_error = (BrokerAPIError("fake explicit rejection", status_code=400, return_code="1")
                                         if len(self.service.submitted) == 2 else None)
        self.service.on_submit = reject_first_sell
        self.engine.poll()
        sells = self.sells()
        self.assertEqual([row["status"] for row in sells], ["rejected", "accepted"])
        self.assertEqual({self.store.prototype_sell_allocation(row["rule_id"])["lot_id"] for row in sells}, {buy.id})
        lot = self.store.prototype_lots(self.item.id)[0]
        self.assertEqual((lot["quantity_remaining"], lot["quantity_reserved_sell"], lot["available_quantity"]), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
