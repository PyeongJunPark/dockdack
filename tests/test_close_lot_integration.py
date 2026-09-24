"""Normal-engine close accounting end to end; fake service and temporary DB only."""
from datetime import date, timedelta
from decimal import Decimal as D
import unittest
from unittest.mock import patch

from dockdack.autotrade import AutoTrader
from dockdack.gui_service import Instrument
from dockdack.models import Market, OrderExecution, OrderSide
from dockdack.market_schedule import session_on
from dockdack.performance import realized_performance
from dockdack.strategy_lots import project_prototype_inventory
from dockdack.watchlist import WatchItem, WatchStore
import test_strategy_lots as lots
import test_lstm30_close as closing


class CloseLotIntegrationTests(unittest.TestCase):
    rule = lots.StrategyLotTests.rule
    accept = lots.StrategyLotTests.accept
    fill = lots.StrategyLotTests.fill
    buy = lots.StrategyLotTests.buy

    def setUp(self):
        lots.StrategyLotTests.setUp(self)
        self.addCleanup(self.directory.cleanup)
        self.session = session_on(Market.DOMESTIC, date(2026, 9, 14))
        stamp = self.session.opened + timedelta(minutes=1)
        self.addCleanup(patch.stopall)
        patch.object(lots, "NOW", stamp).start()
        patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden")).start()
        self.now = self.session.closed - timedelta(minutes=5)
        self.service = closing.CloseService()
        self.service.prices = [D(120)]

    def configure_engine(self):
        self.engine = AutoTrader(self.service, self.store, clock=lambda: self.now)
        self.engine.prototype_lots_enabled = True
        self.engine.enable_holdings_exits = True
        self.engine.configure_close_liquidation(enabled=True)
        self.engine.enable_orders("DEMO_AUTOTRADE")

    def seed(self, *, tracked_manual=True, sellable=6, unknown_cost=False, mixed_timezones=False):
        if tracked_manual:
            manual = self.rule(None, 1)
            self.accept(manual, prototype=False)
            self.fill(manual, average="90")
        first = self.buy(lots.OLD, 2, "100")
        if unknown_cost:
            second = self.rule(lots.NEW, 3)
            self.accept(second)
            self.fill(second, average="110", verified=False)
        else:
            second = self.buy(lots.NEW, 3, "110")
        if mixed_timezones:
            with self.store.connection() as db:
                db.execute("UPDATE attempts SET started_at=? WHERE rule_id=?", ("2026-09-14T10:00:00+09:00", first.id))
                db.execute("UPDATE attempts SET started_at=? WHERE rule_id=?", ("2026-09-14T02:00:00+00:00", second.id))
        self.service.positions_by_market[Market.DOMESTIC] = (
            closing.held(Market.DOMESTIC, "005930", quantity=6, sellable=sellable),)
        self.configure_engine()
        self.engine.maintenance_checkpoint()
        self.assertEqual(len(self.service.submitted), 1, self.engine.close_liquidation_status())
        close_row, = [row for row in self.store.order_history(limit=None) if row["rule_id"].startswith("close-")]
        self.first, self.second, self.close_row = first, second, close_row
        return close_row

    def allocations(self):
        with self.store.connection() as db:
            return tuple(dict(row) for row in db.execute(
                "SELECT * FROM close_lot_allocations ORDER BY CAST(fill_offset AS NUMERIC)"))

    def reconcile(self, filled, *, remaining=None, cancelled=False, verified_price=False):
        ordered = self.close_row["quantity"]
        remaining = ordered - filled if remaining is None else remaining
        self.service.fills = (OrderExecution(
            order_number=self.close_row["order_number"], symbol="005930", side="sell",
            status="cancelled" if cancelled else "filled" if filled == ordered else "partial",
            order_quantity=D(ordered), filled_quantity=D(filled), remaining_quantity=D(remaining),
            order_price=D(120), fill_price=D(120), order_time="15:25:01"),)
        quantity_left = 6 - filled
        self.service.positions_by_market[Market.DOMESTIC] = (
            (closing.held(Market.DOMESTIC, "005930", quantity=quantity_left),) if quantity_left else ())
        self.now += timedelta(seconds=6)
        self.engine.maintenance_checkpoint()
        if verified_price:
            # An independently corroborated fake weighted-fill average, not
            # the preceding OrderExecution's last/unit execution price.
            self.store.record_fill_recovery(self.close_row["rule_id"], status="enriched",
                message="fake verified unique execution events", checked_at=self.now,
                source_api="kt00007+kt00009", price_basis="weighted_fills",
                filled_quantity=D(filled), remaining_quantity=D(remaining), fill_price=D(120),
                price_basis_quantity=D(filled), price_basis_price=D(120))

    def inventory(self):
        return self.store.prototype_inventory(self.item.id)

    def metric(self):
        return realized_performance(self.store.order_history(limit=None))["by_rule_id"][self.close_row["rule_id"]]

    def test_claim_captures_two_model_lots_offsets_and_reserves_without_manual_assignment(self):
        row = self.seed()
        self.assertEqual(row["quantity"], 6)
        self.assertEqual([(a["lot_id"], a["quantity"], a["fill_offset"], a["buy_filled_quantity"], a["buy_average_price"])
                          for a in self.allocations()],
                         [(self.first.id, "2", "0", "2", "100"), (self.second.id, "3", "2", "3", "110")])
        inventory = self.inventory()
        self.assertEqual(inventory["issues"], ())
        self.assertEqual(inventory["expected_quantity"], D(5))
        self.assertEqual(sum(lot["quantity_reserved_sell"] for lot in inventory["lots"]), D(5))
        self.assertEqual(sum(lot["available_quantity"] for lot in inventory["lots"]), D(0))
        self.assertEqual(len(row["close_allocations"]), 2)

    def test_partial_fill_consumes_model_lots_before_older_manual_buy_and_average_is_required(self):
        self.seed()
        self.reconcile(3)
        inventory = self.inventory()
        first, second = sorted(inventory["lots"], key=lambda lot: lot["buy_started_at"])
        self.assertEqual(inventory["issues"], ())
        self.assertEqual((first["quantity_sold"], second["quantity_sold"]), (D(2), D(1)))
        self.assertEqual((first["quantity_reserved_sell"], second["quantity_reserved_sell"]), (D(0), D(2)))
        self.assertEqual(self.metric()["status"], "unknown")
        self.reconcile(3, verified_price=True)
        metric = self.metric()
        self.assertEqual((metric["status"], metric["cost_basis"], metric["realized_profit"]), ("known", D(310), D(50)))
        self.assertEqual(len(self.service.submitted), 1)

    def test_full_fill_replay_restart_and_performance_account_for_model_and_manual_shares(self):
        self.seed()
        self.reconcile(3, verified_price=True)
        self.reconcile(6, verified_price=True)
        self.assertEqual(self.inventory()["expected_quantity"], D(0))
        self.assertEqual(self.inventory()["issues"], ())
        metric = self.metric()
        self.assertEqual((metric["status"], metric["cost_basis"], metric["realized_profit"]), ("known", D(620), D(100)))
        before = self.store.order_history(limit=None)
        self.engine.disarm()
        self.store = WatchStore(self.path)
        self.configure_engine()
        self.engine.maintenance_checkpoint()
        self.assertEqual(self.store.order_history(limit=None), before)
        self.assertTrue(self.store.prototype_inventory(self.item.id, D(0), D(0))["reconciled"])
        self.assertEqual(len(self.service.submitted), 1)

    def test_partial_cancel_releases_reservations_and_restart_never_resubmits_close(self):
        self.seed()
        self.reconcile(3, remaining=0, cancelled=True, verified_price=True)
        inventory = self.inventory()
        self.assertEqual(inventory["issues"], ())
        self.assertEqual(inventory["expected_quantity"], D(2))
        self.assertEqual(sum(lot["available_quantity"] for lot in inventory["lots"]), D(2))
        self.assertEqual(sum(lot["quantity_reserved_sell"] for lot in inventory["lots"]), D(0))
        self.engine.disarm()
        self.store = WatchStore(self.path)
        self.configure_engine()
        self.engine.maintenance_checkpoint()
        self.assertEqual(len(self.service.submitted), 1)
        self.assertTrue(any(row["reason"] == "ALREADY_ATTEMPTED:cancelled"
                            for row in self.engine.close_liquidation_status()["unsold"]))

    def test_only_sellable_four_shares_close_two_plus_two_not_whole_second_lot(self):
        self.seed(sellable=4)
        self.assertEqual([(a["quantity"], a["fill_offset"]) for a in self.allocations()], [("2", "0"), ("2", "2")])
        self.reconcile(4, verified_price=True)
        inventory = self.inventory()
        self.assertEqual(inventory["issues"], ())
        self.assertEqual(inventory["expected_quantity"], D(1))
        self.assertEqual(sum(lot["available_quantity"] for lot in inventory["lots"]), D(1))
        self.assertEqual(self.metric()["cost_basis"], D(420))

    def test_untracked_manual_residual_makes_full_profit_unknown_not_lot_replay(self):
        self.seed(tracked_manual=False)
        self.reconcile(6, verified_price=True)
        self.assertEqual(self.inventory()["issues"], ())
        self.assertEqual(self.inventory()["expected_quantity"], D(0))
        self.assertEqual(self.metric()["reason_code"], "missing_buy_history")
        self.assertIsNone(self.metric()["realized_profit"])

    def test_missing_buy_cost_does_not_block_authorized_account_close_or_fabricate_profit(self):
        self.seed(unknown_cost=True)
        self.assertIsNone(self.allocations()[1]["buy_average_price"])
        self.reconcile(6, verified_price=True)
        self.assertEqual(self.inventory()["issues"], ())
        self.assertEqual(self.inventory()["expected_quantity"], D(0))
        self.assertEqual(self.metric()["reason_code"], "missing_buy_price")

    def test_oldest_first_allocation_uses_absolute_time_not_iso_string_order(self):
        self.seed(mixed_timezones=True)
        self.assertEqual([row["lot_id"] for row in self.allocations()], [self.first.id, self.second.id])
        self.reconcile(3, verified_price=True)
        self.assertEqual(self.metric()["cost_basis"], D(310))

    def test_filtered_ledger_matches_unfiltered_projection_and_limit_is_applied_after_filter(self):
        self.seed()
        original = self.item
        self.item = WatchItem(Instrument(Market.US, "AAPL", "ND"))
        self.store.save_item(self.item)
        other = self.rule(None, 1)
        self.accept(other, prototype=False)
        self.fill(other, average="70")
        self.item = original
        rows = self.store.order_history(limit=None)
        filters = ({"watch_id": self.item.id}, {"market": Market.US}, {"side": OrderSide.SELL},
                   {"watch_id": self.item.id, "side": OrderSide.BUY, "statuses": ("filled",)},
                   {"market": Market.DOMESTIC, "statuses": ("accepted",)}, {"statuses": ()})
        for selected in filters:
            with self.subTest(selected=selected):
                expected = [row for row in rows if all(
                    row[key] == getattr(value, "value", value) if key != "statuses" else row["status"] in value
                    for key, value in selected.items())]
                self.assertEqual(self.store.order_history(limit=None, **selected), tuple(expected))
                self.assertEqual(self.store.order_history(limit=1, **selected), tuple(expected[-1:]))
        with self.store.connection() as db:
            all_allocations = self.store._prototype_allocations(db)
            selected = self.store._prototype_allocations(db, watch_id=self.item.id)
            self.assertEqual(selected, tuple(row for row in all_allocations if row["watch_id"] == self.item.id))
            self.assertEqual(self.store._prototype_allocations(db, rule_id=self.close_row["rule_id"]),
                             tuple(row for row in all_allocations if row["rule_id"] == self.close_row["rule_id"]))

    def test_unallocated_sell_after_buy_in_absolute_time_cannot_hide_behind_timezone_text(self):
        first = self.buy(lots.OLD, 2, "100")
        buy_row = {**self.store.order_history(limit=None)[0], "started_at": "2026-09-14T10:00:00+09:00"}
        sell_row = {**buy_row, "rule_id": "external-sell", "side": "sell", "quantity": 1,
                    "filled_quantity": "1", "started_at": "2026-09-14T02:00:00+00:00"}
        result = project_prototype_inventory([sell_row, buy_row], [])
        self.assertTrue(any("배분되지 않은 매도" in issue for issue in result["issues"]))
        self.assertEqual(result["lots"][0]["lot_id"], first.id)

    def test_unallocated_sell_before_buy_in_absolute_time_is_not_misattributed(self):
        self.buy(lots.OLD, 2, "100")
        buy_row = {**self.store.order_history(limit=None)[0], "started_at": "2026-09-14T02:00:00+00:00"}
        sell_row = {**buy_row, "rule_id": "older-manual-sell", "side": "sell", "quantity": 1,
                    "filled_quantity": "1", "started_at": "2026-09-14T10:00:00+09:00"}
        self.assertEqual(project_prototype_inventory([sell_row, buy_row], [])["issues"], ())

    def test_naive_malformed_missing_and_before_buy_allocation_times_fail_closed(self):
        self.seed()
        rows = self.store.order_history(limit=None)
        with self.store.connection() as db:
            allocations = self.store._prototype_allocations(db)
        for stamp in (None, "", "not-a-time", "2026-09-14T09:00:00", "2026-09-14"):
            for target in (self.first.id, self.close_row["rule_id"]):
                with self.subTest(stamp=stamp, target=target):
                    changed = [{**row, "started_at": stamp} if row["rule_id"] == target else row for row in rows]
                    self.assertTrue(project_prototype_inventory(changed, allocations)["issues"])
        before_buy = [{**row, "started_at": "2026-09-13T23:00:00+00:00"}
                      if row["rule_id"] == self.close_row["rule_id"] else row for row in rows]
        self.assertTrue(any("원매수보다 이른" in issue for issue in project_prototype_inventory(before_buy, allocations)["issues"]))


if __name__ == "__main__":
    unittest.main()
