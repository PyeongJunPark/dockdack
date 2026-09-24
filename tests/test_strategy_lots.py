"""Offline lot replay/reservation tests; each ledger is temporary and account-bound."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

from dockdack.gui_service import Instrument
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, WatchStore


NOW = datetime(2026, 9, 24, 1, tzinfo=timezone.utc)
OLD = "mark1-prototype"
NEW = "mark1-1-prototype"


class StrategyLotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "ledger.sqlite3"
        self.store = WatchStore(self.path)
        self.item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "fake", 31)
        self.store.save_item(self.item)
        self.sequence = 0

    def tearDown(self):
        self.directory.cleanup()

    def rule(self, strategy=OLD, quantity=1, *, side=OrderSide.BUY):
        self.sequence += 1
        rule = TriggerRule(f"lot-test-{self.sequence}", self.item.id, TriggerKind.EXTERNAL,
                           side, quantity, D("100000"), period=20 + self.sequence)
        self.store.add_rule(rule)
        if strategy is not None and side is OrderSide.BUY:
            payload = {"signal_id": f"{strategy}:{self.sequence}", "action": "buy",
                       "strategy_id": strategy, "market": "domestic", "exchange": "KRX", "symbol": "005930"}
            with self.store.connection() as db:
                db.execute("INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (strategy + "-demo-trigger", payload["signal_id"], json.dumps(payload), rule.id,
                            self.item.id, NOW.isoformat(), (NOW + timedelta(hours=1)).isoformat(),
                            "fake-export", NOW.isoformat(), "buy", "ready"))
        return rule

    def accept(self, rule, *, prototype=True):
        self.assertTrue(self.store.claim(rule, D("100"), NOW + timedelta(seconds=self.sequence),
                                         prototype_lots=prototype))
        self.store.finish(rule.id, "accepted", "fake ack", f"ORDER{self.sequence}")

    def fill(self, rule, quantity=None, average="100", *, complete=True, verified=True):
        quantity = D(rule.quantity if quantity is None else quantity)
        price = D(average) if average is not None else None
        remaining = D(rule.quantity) - quantity
        if verified and rule.quantity > 1 and price is not None:
            self.store.record_fill_recovery(rule.id, status="enriched", message="fake average",
                checked_at=NOW + timedelta(minutes=1), price_basis="broker_average",
                filled_quantity=quantity, remaining_quantity=remaining, fill_price=price,
                price_basis_quantity=quantity, price_basis_price=price)
        else:
            self.store.record_execution(rule.id, filled_quantity=quantity, remaining_quantity=remaining,
                                        fill_price=price, observed_at=NOW + timedelta(minutes=1))
        if complete:
            self.store.finish(rule.id, "filled" if quantity == rule.quantity else "cancelled", "fake final")

    def buy(self, strategy=OLD, quantity=1, average="100"):
        rule = self.rule(strategy, quantity)
        self.accept(rule)
        self.fill(rule, average=average)
        return rule

    def sell(self, buy, quantity=1):
        rule = self.rule(None, quantity, side=OrderSide.SELL)
        self.store.reserve_prototype_sell(rule.id, buy.id, quantity, now=NOW)
        return rule

    def test_acknowledgement_does_not_create_shares_or_entry_price(self):
        rule = self.rule()
        self.accept(rule)
        inventory = self.store.prototype_inventory(self.item.id, D(0), D(0))
        lot, = inventory["lots"]
        self.assertTrue(inventory["reconciled"])
        self.assertTrue(inventory["has_prototype_history"])
        self.assertEqual(lot["quantity_remaining"], 0)
        self.assertIsNone(lot["average_price"])
        self.assertEqual(len(self.store.prototype_pending_buys(self.item.id, OLD)), 1)

    def test_models_share_symbol_but_keep_separate_fill_prices_and_barriers(self):
        self.buy(OLD, average="100")
        self.buy(NEW, average="110")
        inventory = self.store.prototype_inventory(self.item.id, D(2), D(2))
        first, second = inventory["lots"]
        self.assertTrue(inventory["reconciled"])
        self.assertEqual((first["take_profit_price"], first["stop_loss_price"]), (D(101), D("99.1")))
        self.assertEqual((second["take_profit_price"], second["stop_loss_price"]), (D("110.55"), D("109.560")))
        self.assertEqual(first["buy_order_number"], "ORDER1")
        self.assertEqual((first["mode"], first["storage_scope"]), ("demo", "demo"))

    def test_same_strategy_holding_blocks_another_buy_but_other_strategy_allowed(self):
        self.buy()
        same = self.rule(OLD)
        self.assertFalse(self.store.claim(same, D(100), NOW, prototype_lots=True))
        other = self.rule(NEW)
        self.assertTrue(self.store.claim(other, D(100), NOW, prototype_lots=True))

    def test_accepted_other_model_can_coexist_but_same_model_pending_cannot(self):
        first = self.rule(OLD)
        self.accept(first)
        self.accept(self.rule(NEW))
        third = self.rule(OLD)
        self.assertFalse(self.store.claim(third, D(100), NOW, prototype_lots=True))

    def test_unknown_or_submitting_other_model_blocks(self):
        first = self.rule(OLD)
        self.assertTrue(self.store.claim(first, D(100), NOW, prototype_lots=True))
        other = self.rule(NEW)
        self.assertFalse(self.store.claim(other, D(100), NOW, prototype_lots=True))
        self.store.finish(first.id, "unknown", "fake ambiguity")
        self.assertFalse(self.store.claim(other, D(100), NOW, prototype_lots=True))

    def test_legacy_claim_still_has_per_symbol_pending_lock(self):
        self.accept(self.rule(OLD))
        other = self.rule(NEW)
        self.assertFalse(self.store.claim(other, D(100), NOW))

    def test_partial_buy_is_visible_but_waits_for_terminal_buy_before_exit(self):
        buy = self.rule(OLD, 3)
        self.accept(buy)
        self.fill(buy, 1, complete=False)
        lot, = self.store.prototype_lots(self.item.id)
        self.assertEqual((lot["quantity_remaining"], lot["available_quantity"]), (D(1), D(0)))
        sell = self.rule(None, side=OrderSide.SELL)
        with self.assertRaises(ValueError):
            self.store.reserve_prototype_sell(sell.id, buy.id, 1)
        self.store.finish(buy.id, "cancelled", "unfilled rest cancelled")
        self.store.reserve_prototype_sell(sell.id, buy.id, 1)

    def test_partial_and_final_weighted_buy_updates_are_not_double_counted(self):
        buy = self.rule(OLD, 3)
        self.accept(buy)
        self.fill(buy, 1, "100", complete=False)
        self.fill(buy, 3, "110")
        for _ in range(3):
            lot, = self.store.prototype_lots(self.item.id)
            self.assertEqual((lot["quantity_remaining"], lot["average_price"]), (D(3), D(110)))

    def test_unverified_multishare_price_is_not_weighted_average(self):
        buy = self.rule(OLD, 2)
        self.accept(buy)
        self.fill(buy, average="120", verified=False)
        inventory = self.store.prototype_inventory(self.item.id, D(2), D(2))
        self.assertFalse(inventory["reconciled"])
        self.assertIsNone(inventory["lots"][0]["average_price"])

    def test_sell_one_model_does_not_consume_other_model_and_restarts_are_idempotent(self):
        old = self.buy(OLD, average="100")
        new = self.buy(NEW, average="120")
        sell = self.sell(new)
        self.accept(sell)
        self.fill(sell, average="121")
        self.store = WatchStore(self.path)
        for _ in range(3):
            lots = {lot["lot_id"]: lot for lot in self.store.prototype_lots(self.item.id)}
            self.assertEqual((lots[old.id]["quantity_remaining"], lots[new.id]["quantity_remaining"]), (D(1), D(0)))
        row = next(row for row in self.store.order_history(None) if row["rule_id"] == sell.id)
        self.assertEqual(row["prototype_lot_id"], new.id)
        self.assertEqual(row["prototype_strategy_id"], NEW)
        self.assertEqual(row["external_source_id"], NEW + "-demo-trigger")

    def test_partial_sell_reserves_only_unfilled_quantity_and_cancel_releases(self):
        buy = self.buy(quantity=3)
        sell = self.sell(buy, 2)
        self.accept(sell)
        self.fill(sell, 1, complete=False)
        lot, = self.store.prototype_lots(self.item.id)
        self.assertEqual((lot["quantity_remaining"], lot["quantity_reserved_sell"], lot["available_quantity"]),
                         (D(2), D(1), D(1)))
        self.store.finish(sell.id, "cancelled", "fake remainder cancel")
        lot, = self.store.prototype_lots(self.item.id)
        self.assertEqual((lot["quantity_reserved_sell"], lot["available_quantity"]), (D(0), D(2)))

    def test_allocation_replay_is_immutable_and_overlapping_reservation_blocked(self):
        buy = self.buy()
        sell = self.sell(buy)
        self.store.reserve_prototype_sell(sell.id, buy.id, 1)
        with self.assertRaises(ValueError):
            self.store.reserve_prototype_sell(sell.id, buy.id, 2)
        other = self.rule(None, side=OrderSide.SELL)
        with self.assertRaises(ValueError):
            self.store.reserve_prototype_sell(other.id, buy.id, 1)
        self.store.pause_rule(sell.id)
        self.store.reserve_prototype_sell(other.id, buy.id, 1)

    def test_accepted_sells_for_different_models_can_coexist_without_oversell(self):
        old, new = self.buy(OLD), self.buy(NEW)
        self.accept(self.sell(old))
        self.accept(self.sell(new))
        inventory = self.store.prototype_inventory(self.item.id, D(2), D(0))
        self.assertTrue(inventory["reconciled"])
        self.assertEqual(sum(lot["available_quantity"] for lot in inventory["lots"]), 0)

    def test_manual_residual_or_missing_broker_shares_fail_closed(self):
        self.buy()
        for quantity in (D(0), D(2)):
            self.assertFalse(self.store.prototype_inventory(self.item.id, quantity, quantity)["reconciled"])

    def test_unallocated_manual_sell_blocks_assignment_even_when_broker_total_matches(self):
        self.buy()
        sell = self.rule(None, side=OrderSide.SELL)
        self.accept(sell, prototype=False)
        self.fill(sell)
        inventory = self.store.prototype_inventory(self.item.id, D(1), D(1))
        self.assertFalse(inventory["reconciled"])
        self.assertIn("배분되지", " ".join(inventory["issues"]))

    def test_missing_historical_fill_snapshot_is_not_invented(self):
        buy = self.rule()
        self.accept(buy)
        self.store.finish(buy.id, "filled", "legacy status only")
        inventory = self.store.prototype_inventory(self.item.id, D(0), D(0))
        self.assertFalse(inventory["reconciled"])
        self.assertEqual(inventory["expected_quantity"], 0)

    def test_account_scope_and_real_mode_cannot_mix(self):
        self.buy()
        with self.assertRaises(ValueError):
            WatchStore(self.path, mode=TradingMode.REAL, storage_scope="other-account")
        with self.assertRaises(ValueError):
            WatchStore(self.path, storage_scope="other-demo-account")

    def test_cumulative_quantity_regression_is_rejected(self):
        buy = self.rule(quantity=2)
        self.accept(buy)
        self.fill(buy, 2)
        with self.assertRaises(ValueError):
            self.store.record_execution(buy.id, filled_quantity=D(1), remaining_quantity=D(1),
                                        fill_price=D(100), observed_at=NOW + timedelta(hours=1))

    def test_closed_lot_does_not_block_same_strategy_new_cycle(self):
        buy = self.buy()
        sell = self.sell(buy)
        self.accept(sell)
        self.fill(sell)
        self.assertTrue(self.store.prototype_inventory(self.item.id, D(0), D(0))["reconciled"])
        self.accept(self.rule(OLD))

    def test_late_buy_basis_change_after_sell_allocation_blocks_reconciliation(self):
        buy = self.buy(quantity=2)
        self.sell(buy, 1)
        self.store.record_fill_recovery(buy.id, status="enriched", message="late correction", checked_at=NOW,
            price_basis="broker_average", filled_quantity=D(2), remaining_quantity=D(0), fill_price=D(110),
            price_basis_quantity=D(2), price_basis_price=D(110))
        self.assertFalse(self.store.prototype_inventory(self.item.id, D(2), D(2))["reconciled"])

    def test_rejected_sell_retry_inherits_original_lot_and_model(self):
        buy = self.buy(NEW)
        sell = self.sell(buy)
        self.assertTrue(self.store.claim(sell, D(101), NOW, prototype_lots=True))
        self.store.finish(sell.id, "rejected", "fake explicit rejection")
        retry = self.store.retry_rule(sell)
        allocation = self.store.prototype_sell_allocation(retry.id)
        self.assertEqual((allocation["lot_id"], allocation["strategy_id"]), (buy.id, NEW))
        self.assertEqual(self.store.prototype_rule_source(retry.id), NEW + "-demo-trigger")
        self.accept(retry)
        self.fill(retry)
        inventory = self.store.prototype_inventory(self.item.id, D(0), D(0))
        self.assertTrue(inventory["reconciled"])
        self.assertEqual(inventory["lots"][0]["quantity_sold"], 1)

    def test_rejected_buy_retry_uses_original_signal_and_new_fill_lot(self):
        buy = self.rule(NEW)
        self.assertTrue(self.store.claim(buy, D(100), NOW, prototype_lots=True))
        self.store.finish(buy.id, "rejected", "fake rejection")
        retry = self.store.retry_rule(buy)
        self.accept(retry)
        self.fill(retry)
        inventory = self.store.prototype_inventory(self.item.id, D(1), D(1))
        self.assertTrue(inventory["reconciled"])
        filled = [lot for lot in inventory["lots"] if lot["quantity_remaining"] > 0]
        self.assertEqual([(lot["lot_id"], lot["strategy_id"]) for lot in filled], [(retry.id, NEW)])

    def test_forged_source_does_not_inherit_another_models_shares(self):
        buy = self.buy(NEW)
        with self.store.connection() as db:
            db.execute("UPDATE external_signals SET source_id=? WHERE rule_id=?",
                       (OLD + "-demo-trigger", buy.id))
        inventory = self.store.prototype_inventory(self.item.id, D(1), D(1))
        self.assertFalse(inventory["reconciled"])
        self.assertEqual(inventory["expected_quantity"], 0)

    def test_missing_sell_execution_is_not_assumed_full_quantity(self):
        buy = self.buy()
        sell = self.sell(buy)
        self.accept(sell)
        self.store.finish(sell.id, "filled", "legacy acknowledgement only")
        inventory = self.store.prototype_inventory(self.item.id, D(0), D(0))
        self.assertFalse(inventory["reconciled"])
        self.assertEqual(inventory["expected_quantity"], 1)

    def test_excess_broker_sellable_quantity_is_invalid(self):
        self.buy()
        self.assertFalse(self.store.prototype_inventory(self.item.id, D(1), D(2))["reconciled"])

    def test_unrelated_manual_position_has_no_prototype_history(self):
        inventory = self.store.prototype_inventory(self.item.id)
        self.assertFalse(inventory["has_prototype_history"])
        self.assertEqual(inventory["lots"], ())

    def test_manual_review_does_not_manufacture_missing_execution_evidence(self):
        buy = self.rule()
        self.accept(buy)
        self.store.mark_reviewed(buy.id, "CHECKED_ORDER_HISTORY")
        inventory = self.store.prototype_inventory(self.item.id, D(0), D(0))
        self.assertFalse(inventory["reconciled"])

    def test_false_full_fill_status_with_partial_snapshot_blocks(self):
        buy = self.rule(quantity=2)
        self.accept(buy)
        self.fill(buy, 1, complete=False)
        self.store.finish(buy.id, "filled", "inconsistent imported status")
        self.assertFalse(self.store.prototype_inventory(self.item.id, D(1), D(1))["reconciled"])


if __name__ == "__main__":
    unittest.main()
