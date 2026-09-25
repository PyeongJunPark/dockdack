"""Offline cumulative, model-origin execution-return regression tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from dockdack.model_performance import model_realized_performance
from dockdack.models import TradingMode
from dockdack.trading.performance import realized_performance


START = datetime(2026, 9, 24, tzinfo=timezone.utc)
OLD, NEW, DEEP = "mark1-prototype", "mark1-1-prototype", "mark1-2-prototype"


def order(number, side, quantity=1, price="100", *, model=None, market="domestic", **overrides):
    exchange, symbol, currency = (("KRX", "005930", "KRW") if market == "domestic" else ("NAS", "AAPL", "USD"))
    watch = f"{market}:{exchange}:{symbol}"
    record = {
        "rule_id": str(number), "watch_id": watch,
        "started_at": (START + timedelta(seconds=number)).isoformat(),
        "status": "filled", "market": market, "exchange": exchange,
        "symbol": symbol, "currency": currency, "side": side,
        "quantity": quantity, "filled_quantity": str(quantity), "fill_price": price,
        "remaining_quantity": "0", "reference_price": "9999999",
    }
    if model:
        signal = f"{model}:{number}"
        payload = {"signal_id": signal, "strategy_id": model, "market": market,
                   "exchange": exchange, "symbol": symbol, "action": "buy", "trading_mode": "demo"}
        record.update(external_payload=json.dumps(payload), external_source_id=model + "-demo-trigger",
                      external_signal_id=signal, external_watch_id=watch, external_decision="buy")
    record.update(overrides)
    record.setdefault("recovery_price_basis", "weighted_fills")
    record.setdefault("price_basis_quantity", record["filled_quantity"])
    record.setdefault("price_basis_price", record["fill_price"])
    return record


def result(records, mode="demo"):
    return model_realized_performance(records, mode=mode)


def row(report, model=OLD, market="domestic", currency="KRW"):
    return next(value for value in report["rows"] if
                (value["strategy_id"], value["market"], value["currency"]) == (model, market, currency))


class ModelPerformanceTests(unittest.TestCase):
    def test_empty_six_rows_are_no_sales_not_zero_return(self):
        report = result([], TradingMode.DEMO)
        self.assertEqual(report["mode"], "demo")
        self.assertEqual(len(report["rows"]), 6)
        for item in report["rows"]:
            self.assertEqual(item["status"], "no_sales")
            self.assertIsNone(item["return_pct"])
            self.assertIsNone(item["realized_profit"])
        self.assertTrue(report["complete"])
        self.assertIsNone(report["coverage"]["first_order_at"])

    def test_open_positions_and_unfilled_order_are_not_sales(self):
        report = result([order(1, "buy", model=OLD), order(2, "sell", status="accepted", filled_quantity="0", remaining_quantity="1")])
        self.assertEqual(row(report)["status"], "no_sales")
        self.assertEqual(report["coverage"]["executed_sell_count"], 0)

    def test_returns_weight_sold_cost_instead_of_average_percentages(self):
        report = result([order(1, "buy", model=OLD), order(2, "sell", price="110"),
                         order(3, "buy", price="1000", model=OLD), order(4, "sell", price="900")])
        model = row(report)
        self.assertEqual(model["known_cost_basis"], D("1100"))
        self.assertEqual(model["realized_profit"], D("-90"))
        self.assertEqual(model["return_pct"], D("-90") / D("1100") * 100)
        self.assertNotEqual(model["return_pct"], 0)
        self.assertEqual(model["known_sell_count"], 2)
        self.assertTrue(report["gross"])
        self.assertIn("수수료·세금 제외", report["description"])

    def test_two_models_same_symbol_explicit_sale_uses_origin_not_fifo(self):
        records = [order(1, "buy", model=OLD), order(2, "buy", price="120", model=NEW),
                   order(3, "sell", price="125", prototype_lot_id="2", prototype_buy_rule_id="2")]
        report = result(records)
        self.assertEqual(row(report)["status"], "no_sales")
        self.assertEqual(row(report, NEW)["realized_profit"], D("5"))
        self.assertEqual(row(report, NEW)["known_cost_basis"], D("120"))

    def test_three_models_same_symbol_keep_independent_sale_attribution(self):
        records = [order(1, "buy", model=OLD), order(2, "buy", price="120", model=NEW),
                   order(3, "buy", price="80", model=DEEP),
                   order(4, "sell", price="110", prototype_lot_id="3", prototype_buy_rule_id="3"),
                   order(5, "sell", price="130", prototype_lot_id="2", prototype_buy_rule_id="2"),
                   order(6, "sell", price="90", prototype_lot_id="1", prototype_buy_rule_id="1")]
        report = result(records)
        self.assertEqual(row(report, OLD)["realized_profit"], D("-10"))
        self.assertEqual(row(report, NEW)["realized_profit"], D("10"))
        self.assertEqual(row(report, DEEP)["realized_profit"], D("30"))
        self.assertEqual(row(report, DEEP)["known_cost_basis"], D("80"))
        self.assertEqual(row(report, DEEP)["return_pct"], D("37.5"))
        self.assertEqual(report["coverage"]["executed_sell_count"], 3)
        self.assertTrue(report["complete"])

    def test_sell_friendly_model_cannot_reassign_origin(self):
        records = [order(1, "buy", model=OLD), order(2, "sell", price="110", model=NEW,
                    prototype_strategy_id=NEW, prototype_model_title="wrong")]
        report = result(records)
        self.assertEqual(row(report)["realized_profit"], 10)
        self.assertEqual(row(report, NEW)["status"], "no_sales")

    def test_manual_fifo_sale_spans_models_without_double_counting(self):
        records = [order(1, "buy", 2, model=OLD), order(2, "buy", 2, "120", model=NEW),
                   order(3, "sell", 3, "130")]
        report = result(records)
        self.assertEqual(row(report)["realized_profit"], 60)
        self.assertEqual(row(report, NEW)["realized_profit"], 10)
        self.assertEqual(row(report)["known_sell_count"], 1)
        self.assertEqual(row(report, NEW)["known_sell_count"], 1)
        self.assertEqual(report["coverage"]["executed_sell_count"], 1)
        self.assertEqual(sum(value["known_cost_basis"] or D(0) for value in report["rows"]), 320)

    def test_close_allocations_follow_cumulative_fill_offsets(self):
        records = [order(1, "buy", 2, model=OLD), order(2, "buy", 2, "120", model=NEW),
                   order(3, "sell", 4, "130", rule_id="close-test", status="accepted", filled_quantity="3", remaining_quantity="1",
                         close_allocations=({"lot_id": "2", "quantity": "2", "fill_offset": "0"},
                                            {"lot_id": "1", "quantity": "2", "fill_offset": "2"}))]
        report = result(records)
        self.assertEqual(row(report)["known_quantity"], 1)
        self.assertEqual(row(report)["realized_profit"], 30)
        self.assertEqual(row(report, NEW)["known_quantity"], 2)
        self.assertEqual(row(report, NEW)["realized_profit"], 20)

    def test_partially_filled_then_cancelled_sell_counts_only_executed_shares(self):
        report = result([order(1, "buy", 4, model=OLD), order(2, "sell", 4, "105", status="cancelled", filled_quantity="2")])
        self.assertEqual(row(report)["known_quantity"], 2)
        self.assertEqual(row(report)["known_cost_basis"], 200)
        self.assertEqual(row(report)["return_pct"], 5)

    def test_cumulative_replay_is_idempotent_and_does_not_modify_input(self):
        records = [order(1, "buy", 3, model=OLD), order(2, "sell", 3, "110", status="accepted", filled_quantity="1", remaining_quantity="2")]
        original = copy.deepcopy(records)
        first = result(records)
        self.assertEqual(first, result(iter(copy.deepcopy(records))))
        self.assertEqual(records, original)
        records[1].update(status="filled", filled_quantity="3", remaining_quantity="0", price_basis_quantity="3")
        self.assertEqual(row(result(records))["realized_profit"], 30)
        self.assertEqual(row(first)["realized_profit"], 10)

    def test_missing_fill_price_does_not_use_reference_quote(self):
        report = result([order(1, "buy", model=OLD), order(2, "sell", price=None)])
        model = row(report)
        self.assertEqual(model["status"], "incomplete")
        self.assertEqual(model["unknown_sell_count"], 1)
        self.assertIsNone(model["return_pct"])
        self.assertIsNone(model["known_return_pct"])
        self.assertFalse(report["complete"])

    def test_unverified_multishare_average_is_unknown(self):
        report = result([order(1, "buy", 2, model=OLD, recovery_price_basis="last_fill"), order(2, "sell", 2, "110")])
        self.assertIsNone(row(report)["realized_profit"])
        self.assertEqual(row(report)["unknown_quantity"], 2)

    def test_stale_average_evidence_does_not_get_reused(self):
        report = result([order(1, "buy", 2, model=OLD), order(2, "sell", 2, "110", price_basis_quantity="1")])
        self.assertEqual(row(report)["status"], "incomplete")
        self.assertIsNone(row(report)["known_realized_profit"])

    def test_unknown_sale_keeps_known_subset_but_hides_whole_return(self):
        report = result([order(1, "buy", 2, model=OLD), order(2, "sell", price="110"), order(3, "sell", price=None)])
        model = row(report)
        self.assertEqual(model["known_return_pct"], 10)
        self.assertEqual(model["known_realized_profit"], 10)
        self.assertIsNone(model["realized_profit"])
        self.assertIsNone(model["return_pct"])
        self.assertEqual(model["unknown_sell_count"], 1)

    def test_one_sale_with_unknown_buy_price_keeps_all_its_allocations_unknown(self):
        records = [order(1, "buy", model=OLD), order(2, "buy", price=None, model=NEW), order(3, "sell", 2, "110")]
        report = result(records)
        for model in (OLD, NEW):
            self.assertEqual(row(report, model)["status"], "incomplete")
            self.assertIsNone(row(report, model)["known_return_pct"])

    def test_missing_buy_history_is_visible_unassigned_not_fabricated_model(self):
        report = result([order(1, "sell", price="110", model=OLD, prototype_buy_rule_id="absent")])
        self.assertEqual(row(report)["status"], "no_sales")
        self.assertEqual(row(report, "unassigned")["unknown_sell_count"], 1)
        self.assertEqual(report["coverage"]["unassigned_sell_count"], 1)
        self.assertFalse(report["complete"])

    def test_unassigned_manual_purchase_can_have_known_math_not_model_credit(self):
        report = result([order(1, "buy"), order(2, "sell", price="110")])
        unknown = row(report, "unassigned")
        self.assertEqual(unknown["realized_profit"], 10)
        self.assertFalse(unknown["attribution_complete"])
        self.assertFalse(report["complete"])
        self.assertEqual(row(report)["status"], "no_sales")

    def test_mismatched_source_metadata_cannot_claim_model(self):
        report = result([order(1, "buy", model=OLD, external_source_id=NEW), order(2, "sell", price="110")])
        self.assertEqual(row(report)["status"], "no_sales")
        self.assertEqual(row(report, "unassigned")["known_sell_count"], 1)
        self.assertIn("invalid_model_origin", report["warnings"][0]["reason_codes"])

    def test_malformed_signal_payload_is_explicit_warning(self):
        report = result([order(1, "buy", model=OLD, external_payload="not-json"), order(2, "sell", price="110")])
        self.assertFalse(report["complete"])
        self.assertEqual(row(report, "unassigned")["known_realized_profit"], 10)

    def test_krw_usd_not_mixed_and_market_totals_reconcile(self):
        records = [order(1, "buy", price="10000", model=OLD), order(2, "sell", price="10100"),
                   order(3, "buy", price="100", model=OLD, market="us"), order(4, "sell", price="90", market="us")]
        report = result(records)
        self.assertEqual(row(report)["return_pct"], 1)
        self.assertEqual(row(report, OLD, "us", "USD")["return_pct"], -10)
        performance = realized_performance(records)
        for market, currency in (("domestic", "KRW"), ("us", "USD")):
            subtotal = sum(item["known_realized_profit"] or D(0) for item in report["rows"] if item["market"] == market)
            self.assertEqual(subtotal, performance["summaries"][(market, currency)]["known_realized_profit"])

    def test_demo_prototype_provenance_is_not_relabelled_real(self):
        for model in (OLD, NEW, DEEP):
            with self.subTest(model=model):
                report = result([order(1, "buy", model=model), order(2, "sell", price="110")], "real")
                self.assertEqual(report["mode"], "real")
                self.assertEqual(row(report, model)["status"], "no_sales")
                self.assertFalse(report["complete"])
                self.assertIn("model_environment_mismatch", report["warnings"][0]["reason_codes"])

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            result([], "backtest")

    def test_duplicate_snapshot_does_not_duplicate_profit(self):
        buy = order(1, "buy", model=OLD)
        report = result([buy, copy.deepcopy(buy), order(2, "sell", price="110")])
        self.assertFalse(report["complete"])
        self.assertIsNone(row(report)["realized_profit"])
        self.assertEqual(report["coverage"]["executed_sell_count"], 1)

    def test_coverage_includes_durable_history_boundaries(self):
        report = result([order(1, "buy", model=OLD), order(9, "sell", price="110")])
        self.assertEqual(report["coverage"]["ledger_order_count"], 2)
        self.assertEqual(report["coverage"]["first_order_at"], (START + timedelta(seconds=1)).isoformat())
        self.assertEqual(report["coverage"]["last_order_at"], (START + timedelta(seconds=9)).isoformat())

    def test_per_buy_breakdown_is_additive_without_changing_totals(self):
        records = [order(1, "buy", 2, model=OLD), order(2, "buy", 2, "120", model=NEW), order(3, "sell", 3, "130")]
        metric = realized_performance(records)["by_rule_id"]["3"]
        self.assertEqual(tuple(allocation["buy_rule_id"] for allocation in metric["allocations"]), ("1", "2"))
        self.assertEqual(sum(allocation["cost_basis"] for allocation in metric["allocations"]), metric["cost_basis"])
        self.assertEqual(sum(allocation["realized_profit"] for allocation in metric["allocations"]), metric["realized_profit"])

    def test_collector_can_reuse_exact_same_fifo_projection(self):
        records = [order(1, "buy", model=OLD), order(2, "sell", price="110")]
        performance = realized_performance(records)
        expected = result(records)
        with patch("dockdack.model_performance.realized_performance", side_effect=AssertionError("replayed twice")):
            actual = model_realized_performance(records, mode="demo", performance=performance)
        self.assertEqual(actual, expected)
        self.assertEqual(row(actual)["executed_sell_count"], 1)

    def test_durable_account_ledger_reopen_reproduces_model_return(self):
        from dockdack.gui_service import Instrument
        from dockdack.models import Market, OrderSide
        from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, WatchStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "account.sqlite3"
            store = WatchStore(path)
            item = WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "offline", 31)
            store.save_item(item)
            buy = TriggerRule("integration-buy", item.id, TriggerKind.EXTERNAL, OrderSide.BUY, 1, D("100000"))
            sell = TriggerRule("integration-sell", item.id, TriggerKind.EXTERNAL, OrderSide.SELL, 1, D("100000"))
            store.add_rule(buy)
            store.add_rule(sell)
            signal = order(1, "buy", model=OLD)
            with store.connection() as db:
                db.execute("INSERT INTO external_signals VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                    signal["external_source_id"], signal["external_signal_id"], signal["external_payload"],
                    buy.id, item.id, START.isoformat(), (START + timedelta(hours=1)).isoformat(),
                    "offline-export", START.isoformat(), "buy", "ready"))
            self.assertTrue(store.claim(buy, D("100"), START, prototype_lots=True))
            store.finish(buy.id, "accepted", "offline acknowledgement", "OFFLINE-BUY")
            store.record_execution(buy.id, filled_quantity=D(1), remaining_quantity=D(0), fill_price=D(100), observed_at=START)
            store.finish(buy.id, "filled", "offline fill", "OFFLINE-BUY")
            store.reserve_prototype_sell(sell.id, buy.id, 1, now=START + timedelta(seconds=1))
            self.assertTrue(store.claim(sell, D("110"), START + timedelta(seconds=2), prototype_lots=True))
            store.finish(sell.id, "accepted", "offline acknowledgement", "OFFLINE-SELL")
            store.record_execution(sell.id, filled_quantity=D(1), remaining_quantity=D(0), fill_price=D(110), observed_at=START + timedelta(seconds=3))
            store.finish(sell.id, "filled", "offline fill", "OFFLINE-SELL")
            before = result(store.order_history(limit=None), store.mode)
            reopened = WatchStore(path)
            after = result(reopened.order_history(limit=None), reopened.mode)
            self.assertEqual(before, after)
            self.assertEqual(row(after)["realized_profit"], 10)
            self.assertEqual(row(after)["known_sell_count"], 1)

    def test_invalid_close_mapping_is_unknown_not_optimistic_fifo(self):
        report = result([order(1, "buy", model=OLD), order(2, "sell", price="110", rule_id="close-test",
                         close_allocations=({"lot_id": "1", "quantity": "1", "fill_offset": "1"},))])
        self.assertFalse(report["complete"])
        self.assertEqual(row(report)["status"], "incomplete")
        self.assertIsNone(row(report)["known_realized_profit"])

    def test_invalid_quantity_remains_visible_without_fabricated_shares(self):
        report = result([order(1, "buy", model=OLD), order(2, "sell", filled_quantity="2")])
        self.assertFalse(report["complete"])
        self.assertEqual(row(report, "unassigned")["unknown_sell_count"], 1)
        self.assertEqual(report["coverage"]["incomplete_sell_count"], 1)

    def test_reviewed_partial_buys_and_sells_keep_confirmed_execution_profit(self):
        records = [order(1, "buy", 5, "100", model=OLD, status="reviewed", filled_quantity="3", remaining_quantity="2"),
                   order(2, "sell", 3, "110", status="reviewed", filled_quantity="2", remaining_quantity="1")]
        report = result(records)
        self.assertEqual(row(report)["known_quantity"], 2)
        self.assertEqual(row(report)["known_cost_basis"], 200)
        self.assertEqual(row(report)["realized_profit"], 20)
        self.assertEqual(row(report)["return_pct"], 10)
        self.assertEqual(row(report)["executed_sell_count"], 1)

    def test_manual_review_never_infers_fill_from_requested_quantity_or_status(self):
        reviewed_buy = order(1, "buy", 2, model=OLD, status="reviewed", filled_quantity=None, remaining_quantity=None)
        reviewed_sell = order(2, "sell", 2, "110", status="reviewed", filled_quantity=None, remaining_quantity=None)
        report = result([reviewed_buy, reviewed_sell])
        self.assertEqual(report["coverage"]["executed_sell_count"], 0)
        self.assertEqual(row(report)["status"], "no_sales")
        self.assertIsNone(row(report)["return_pct"])
        # A subsequent actually executed sale must not borrow the reviewed
        # purchase's requested (but never confirmed) quantity.
        report = result([reviewed_buy, order(3, "sell", price="110")])
        self.assertEqual(row(report, "unassigned")["unknown_sell_count"], 1)

    def test_reviewed_buy_without_bound_average_remains_unknown_cost(self):
        report = result([order(1, "buy", 3, "100", model=OLD, status="reviewed", filled_quantity="2", remaining_quantity="1", recovery_price_basis="last_fill"),
                         order(2, "sell", price="110")])
        self.assertEqual(row(report)["unknown_sell_count"], 1)
        self.assertIsNone(row(report)["return_pct"])

    def test_reviewed_sell_without_bound_average_remains_unknown_profit(self):
        report = result([order(1, "buy", 3, model=OLD),
                         order(2, "sell", 3, "110", status="reviewed", filled_quantity="2", remaining_quantity="1", price_basis_quantity="1")])
        self.assertEqual(row(report)["unknown_quantity"], 2)
        self.assertIsNone(row(report)["known_realized_profit"])

    def test_reviewed_and_cancelled_remainder_have_same_partial_realized_profit(self):
        records = [order(1, "buy", 3, model=OLD),
                   order(2, "sell", 3, "110", status="cancelled", filled_quantity="2", remaining_quantity="0")]
        before = result(records)
        records[1]["status"] = "reviewed"
        self.assertEqual(result(records), before)

    def test_reviewed_invalid_filled_quantity_fails_closed(self):
        for quantity in ("-1", "2", "NaN"):
            with self.subTest(quantity=quantity):
                report = result([order(1, "buy", model=OLD),
                                 order(2, "sell", status="reviewed", filled_quantity=quantity)])
                self.assertFalse(report["complete"])
                self.assertEqual(report["coverage"]["incomplete_sell_count"], 1)
                self.assertIsNone(row(report)["realized_profit"])

    def test_reviewed_explicit_zero_never_becomes_an_execution(self):
        report = result([order(1, "buy", model=OLD),
                         order(2, "sell", status="reviewed", filled_quantity="0", remaining_quantity="1")])
        self.assertEqual(report["coverage"]["executed_sell_count"], 0)
        self.assertEqual(row(report)["status"], "no_sales")


if __name__ == "__main__":
    unittest.main()
