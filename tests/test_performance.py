from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

from dockdack.performance import realized_performance


START = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)


def order(number, side, quantity=1, price="100", *, verified_average=True, **overrides):
    """Synthetic prices are verified cumulative averages unless testing otherwise."""
    record = {
        "rule_id": str(number), "watch_id": "domestic:KRX:005930",
        "started_at": (START + timedelta(seconds=number)).isoformat(),
        "status": "filled", "order_number": str(number), "market": "domestic",
        "exchange": "KRX", "symbol": "005930", "currency": "KRW", "side": side,
        "quantity": quantity, "filled_quantity": str(quantity), "fill_price": price,
        "remaining_quantity": "0", "observed_at": START.isoformat(),
    }
    record.update(overrides)
    if verified_average:
        record.setdefault("price_basis", "weighted_fills")
        record.setdefault("price_basis_quantity", record["filled_quantity"] if record["filled_quantity"] is not None else quantity)
        record.setdefault("price_basis_price", record["fill_price"])
    return record


class RealizedPerformanceTests(unittest.TestCase):
    def metric(self, records, rule_id):
        return realized_performance(records)["by_rule_id"][str(rule_id)]

    def test_known_profit_loss_and_return_use_execution_prices_only(self):
        result = realized_performance([
            order(1, "buy", 2, "100", reference_price="1"),
            order(2, "sell", 1, "110", reference_price="9999"),
            order(3, "sell", 1, "90"),
        ])
        self.assertEqual(result["by_rule_id"]["2"]["realized_profit"], D("10"))
        self.assertEqual(result["by_rule_id"]["2"]["return_pct"], D("10"))
        self.assertEqual(result["by_rule_id"]["3"]["realized_profit"], D("-10"))
        summary = result["summaries"][("domestic", "KRW")]
        self.assertEqual(summary["realized_profit"], D("0"))
        self.assertEqual(summary["return_pct"], D("0"))
        self.assertEqual(summary["known_sell_count"], 2)
        self.assertTrue(result["gross"])
        self.assertEqual(result["basis"], "local_fifo")
        self.assertIn("계좌 전체 수익률 아님", result["description"])

    def test_fifo_multiple_buys_and_partial_lot_consumption(self):
        result = realized_performance([
            order(1, "buy", 2, "100"), order(2, "buy", 3, "120"),
            order(3, "sell", 3, "130"), order(4, "sell", 2, "125"),
        ])["by_rule_id"]
        self.assertEqual(result["3"]["cost_basis"], D("320"))
        self.assertEqual(result["3"]["proceeds"], D("390"))
        self.assertEqual(result["3"]["realized_profit"], D("70"))
        self.assertEqual(result["3"]["return_pct"], D("21.875"))
        self.assertEqual(result["4"]["cost_basis"], D("240"))
        self.assertEqual(result["4"]["realized_profit"], D("10"))

    def test_partial_accepted_and_cancelled_quantities_are_counted(self):
        records = [order(1, "buy", 5, "100", status="accepted", filled_quantity="2", remaining_quantity="3"),
                   order(2, "sell", 3, "110", status="cancelled", filled_quantity="1", remaining_quantity="0"),
                   order(3, "sell", 4, "120", status="accepted", filled_quantity="1", remaining_quantity="3")]
        result = realized_performance(records)["by_rule_id"]
        self.assertEqual(result["2"]["filled_quantity"], D("1"))
        self.assertEqual(result["2"]["realized_profit"], D("10"))
        self.assertEqual(result["3"]["realized_profit"], D("20"))

    def test_unconfirmed_or_failed_states_are_not_execution_evidence(self):
        records = [order(i, "buy", price="1", status=status)
                   for i, status in enumerate(("submitting", "unknown", "rejected", "not_sent"), 1)]
        records += [order(5, "buy", status="accepted", filled_quantity=None),
                    order(6, "buy", price="100"), order(7, "sell", price="110")]
        result = realized_performance(records)
        for i in range(1, 6):
            self.assertEqual(result["by_rule_id"][str(i)]["status"], "not_applicable")
        self.assertEqual(result["by_rule_id"]["7"]["realized_profit"], D("10"))

    def test_legacy_filled_quantity_can_be_inferred_but_not_fill_price(self):
        result = realized_performance([
            order(1, "buy", 2, None, filled_quantity=None, reference_price="99"),
            order(2, "sell", 2, "110", filled_quantity=None),
        ])["by_rule_id"]
        self.assertTrue(result["1"]["quantity_inferred"])
        self.assertEqual(result["2"]["filled_quantity"], D("2"))
        self.assertEqual(result["2"]["status"], "unknown")
        self.assertEqual(result["2"]["reason_code"], "missing_buy_price")
        self.assertIsNone(result["2"]["cost_basis"])
        self.assertIsNone(result["2"]["realized_profit"])

    def test_unknown_buy_price_lots_are_consumed_before_known_lots(self):
        result = realized_performance([
            order(1, "buy", 2, None), order(2, "buy", 2, "100"),
            order(3, "sell", 1, "110"), order(4, "sell", 2, "120"),
            order(5, "sell", 1, "130"),
        ])["by_rule_id"]
        for identifier in ("3", "4"):
            self.assertEqual(result[identifier]["status"], "unknown")
            self.assertIsNone(result[identifier]["cost_basis"])
            self.assertIsNone(result[identifier]["realized_profit"])
        self.assertEqual(result["5"]["realized_profit"], D("30"))

    def test_missing_sell_price_still_consumes_quantity(self):
        result = realized_performance([
            order(1, "buy", 1, "100"), order(2, "buy", 1, "120"),
            order(3, "sell", 1, None, reference_price="999"), order(4, "sell", 1, "130"),
        ])["by_rule_id"]
        self.assertEqual(result["3"]["reason_code"], "missing_fill_price")
        self.assertEqual(result["3"]["cost_basis"], D("100"))
        self.assertIsNone(result["3"]["proceeds"])
        self.assertIsNone(result["3"]["realized_profit"])
        self.assertEqual(result["4"]["realized_profit"], D("10"))

    def test_partly_unmatched_sell_is_entirely_unknown_and_later_buy_not_retroactive(self):
        result = realized_performance([
            order(1, "buy", 1, "100"), order(2, "sell", 2, "110"),
            order(3, "buy", 1, "120"), order(4, "sell", 1, "130"),
        ])["by_rule_id"]
        sale = result["2"]
        self.assertEqual(sale["matched_quantity"], D("1"))
        self.assertEqual(sale["unmatched_quantity"], D("1"))
        self.assertEqual(sale["reason_code"], "missing_buy_history")
        self.assertIsNone(sale["realized_profit"])
        self.assertIsNone(sale["cost_basis"])
        self.assertEqual(result["4"]["realized_profit"], D("10"))

    def test_earlier_unmatched_sell_does_not_use_future_buy(self):
        result = self.metric([order(1, "sell", price="110"), order(2, "buy", price="100")], 1)
        self.assertEqual(result["reason_code"], "missing_buy_history")
        self.assertIsNone(result["realized_profit"])

    def test_full_history_is_required_not_last_display_page(self):
        records = [order(1, "buy", price="100"), order(2, "sell", price="110")]
        self.assertEqual(self.metric(records, 2)["status"], "known")
        self.assertEqual(self.metric(records[-1:], 2)["reason_code"], "missing_buy_history")

    def test_sorts_timestamps_and_preserves_input_tie_order(self):
        buy, sell = order(1, "buy", price="100"), order(2, "sell", price="110")
        self.assertEqual(self.metric([sell, buy], 2)["realized_profit"], D("10"))
        sell["started_at"] = buy["started_at"]
        self.assertEqual(self.metric([buy, sell], 2)["realized_profit"], D("10"))
        self.assertIsNone(self.metric([sell, buy], 2)["realized_profit"])

    def test_aware_timezones_are_compared_by_instant(self):
        buy = order(1, "buy", started_at="2026-09-15T09:00:00+09:00")
        sell = order(2, "sell", price="110", started_at="2026-09-15T00:00:01+00:00")
        self.assertEqual(self.metric([sell, buy], 2)["realized_profit"], D("10"))

    def test_market_exchange_currency_symbol_are_all_part_of_fifo_key(self):
        for changed in ({"market": "us"}, {"exchange": "NXT"}, {"currency": "USD"}, {"symbol": "000660"}):
            with self.subTest(changed=changed):
                result = self.metric([order(1, "buy"), order(2, "sell", price="110", **changed)], 2)
                self.assertEqual(result["reason_code"], "missing_buy_history")

    def test_market_summaries_never_add_won_to_dollars(self):
        us = {"market": "us", "exchange": "ND", "symbol": "AAPL", "currency": "USD"}
        result = realized_performance([order(1, "buy", price="10000"), order(2, "sell", price="10100"),
                                       order(3, "buy", price="200", **us), order(4, "sell", price="202", **us)])
        self.assertEqual(result["summaries"][("domestic", "KRW")]["realized_profit"], D("100"))
        self.assertEqual(result["summaries"][("us", "USD")]["realized_profit"], D("2"))

    def test_mixed_known_unknown_summary_is_explicit_partial_not_full_profit(self):
        result = realized_performance([
            order(1, "buy", price="100"), order(2, "sell", price="110"), order(3, "sell", price="120"),
        ])
        summary = result["summaries"][("domestic", "KRW")]
        self.assertEqual(summary["known_realized_profit"], D("10"))
        self.assertEqual(summary["known_sell_count"], 1)
        self.assertEqual(summary["unknown_sell_count"], 1)
        self.assertFalse(summary["complete"])
        self.assertFalse(result["complete"])
        self.assertIsNone(summary["realized_profit"])
        self.assertIsNone(summary["return_pct"])

    def test_entirely_unknown_summary_does_not_show_zero_profit(self):
        summary = realized_performance([order(1, "sell")])["summaries"][("domestic", "KRW")]
        self.assertIsNone(summary["known_realized_profit"])
        self.assertIsNone(summary["known_return_pct"])
        self.assertIsNone(summary["realized_profit"])
        self.assertEqual(summary["unknown_sell_count"], 1)

    def test_summary_return_is_cost_weighted_not_mean_of_returns(self):
        summary = realized_performance([
            order(1, "buy", price="100"), order(2, "sell", price="110"),
            order(3, "buy", price="900"), order(4, "sell", price="1080"),
        ])["summaries"][("domestic", "KRW")]
        self.assertEqual(summary["known_cost_basis"], D("1000"))
        self.assertEqual(summary["known_realized_profit"], D("190"))
        self.assertEqual(summary["known_return_pct"], D("19"))

    def test_fractional_quantity_and_decimal_price_remain_exact(self):
        us = {"market": "us", "exchange": "ND", "symbol": "AAPL", "currency": "USD"}
        result = self.metric([order(1, "buy", "0.3", "123.4567", **us),
                              order(2, "sell", "0.2", "123.5567", **us)], 2)
        self.assertEqual(result["cost_basis"], D("24.69134"))
        self.assertEqual(result["proceeds"], D("24.71134"))
        self.assertEqual(result["realized_profit"], D("0.02000"))

    def test_one_share_full_fill_needs_no_average_provenance(self):
        result = self.metric([order(1, "buy", verified_average=False),
                              order(2, "sell", price="110", verified_average=False)], 2)
        self.assertEqual(result["effective_fill_price"], D("110"))
        self.assertEqual(result["realized_profit"], D("10"))
        self.assertIsNone(result["price_reason_code"])
        self.assertEqual(result["price_reason"], "")

    def test_unverified_multi_share_purchase_remains_unknown_cost_lot(self):
        records = [order(1, "buy", 2, "100", verified_average=False),
                   order(2, "buy", 1, "120"), order(3, "sell", 2, "110"),
                   order(4, "sell", 1, "130")]
        result = realized_performance(records)["by_rule_id"]
        self.assertIsNone(result["1"]["effective_fill_price"])
        self.assertEqual(result["1"]["price_reason_code"], "unverified_average_price")
        self.assertIn("평균 체결가", result["1"]["price_reason"])
        self.assertEqual(result["3"]["reason_code"], "missing_buy_price")
        self.assertIsNone(result["3"]["cost_basis"])
        self.assertEqual(result["4"]["realized_profit"], D("10"))

    def test_unverified_multi_share_sale_does_not_use_unit_price_as_average(self):
        result = self.metric([order(1, "buy", 2, "100"),
                              order(2, "sell", 2, "110", verified_average=False)], 2)
        self.assertEqual(result["reason_code"], "unverified_average_price")
        self.assertEqual(result["cost_basis"], D("200"))
        self.assertIsNone(result["effective_fill_price"])
        self.assertIsNone(result["proceeds"])
        self.assertIsNone(result["realized_profit"])

    def test_partial_one_share_of_larger_order_also_requires_bound_provenance(self):
        result = self.metric([order(1, "buy"),
                              order(2, "sell", 2, "110", status="accepted", filled_quantity="1",
                                    remaining_quantity="1", verified_average=False)], 2)
        self.assertEqual(result["reason_code"], "unverified_average_price")
        self.assertIsNone(result["effective_fill_price"])

    def test_average_provenance_must_bind_both_current_quantity_and_price(self):
        for changed in ({"price_basis_quantity": "1"}, {"price_basis_price": "100"},
                        {"price_basis_quantity": None}, {"price_basis_price": None},
                        {"price_basis": "single_share"}, {"price_basis": "last_execution"},
                        {"price_basis_quantity": "NaN"}, {"price_basis_price": "Infinity"}):
            with self.subTest(changed=changed):
                result = self.metric([order(1, "buy", 2), order(2, "sell", 2, "110", **changed)], 2)
                self.assertEqual(result["reason_code"], "unverified_average_price")
                self.assertIsNone(result["effective_fill_price"])

    def test_each_allowlisted_average_basis_with_exact_binding_is_usable(self):
        for basis in ("broker_average", "weighted_fills"):
            with self.subTest(basis=basis):
                result = self.metric([order(1, "buy", 2), order(2, "sell", 2, "110", price_basis=basis)], 2)
                self.assertEqual(result["effective_fill_price"], D("110"))
                self.assertEqual(result["realized_profit"], D("20"))

    def test_recovery_basis_is_not_trusted_without_binding_to_execution_snapshot(self):
        sale = order(2, "sell", 2, "110", verified_average=False, recovery_price_basis="broker_average")
        self.assertIsNone(self.metric([order(1, "buy", 2), sale], 2)["effective_fill_price"])
        sale.update(price_basis_quantity="2", price_basis_price="110")
        self.assertEqual(self.metric([order(1, "buy", 2), sale], 2)["realized_profit"], D("20"))

    def test_unconfirmed_raw_price_is_not_an_effective_display_price(self):
        result = self.metric([order(1, "buy", status="accepted", filled_quantity=None)], 1)
        self.assertIsNone(result["effective_fill_price"])
        self.assertEqual(result["price_reason_code"], "not_filled")

    def test_duplicate_buy_or_sell_ids_fail_closed_without_double_counting(self):
        for duplicate_side in ("buy", "sell"):
            with self.subTest(side=duplicate_side):
                records = [order(1, "buy"), order(2, "sell", price="110")]
                records.append(copy.deepcopy(records[0 if duplicate_side == "buy" else 1]))
                result = realized_performance(records)
                self.assertEqual(result["by_rule_id"]["2"]["reason_code"], "duplicate_rule_id")
                self.assertIsNone(result["by_rule_id"]["2"]["realized_profit"])
                self.assertEqual(result["summaries"][("domestic", "KRW")]["unknown_sell_count"], 1)
                self.assertFalse(result["complete"])

    def test_duplicate_id_does_not_poison_unrelated_instrument(self):
        records = [order(1, "buy"), order(1, "buy"), order(2, "sell", price="110"),
                   order(3, "buy", symbol="000660"), order(4, "sell", price="120", symbol="000660")]
        result = realized_performance(records)["by_rule_id"]
        self.assertEqual(result["2"]["status"], "unknown")
        self.assertEqual(result["4"]["realized_profit"], D("20"))

    def test_invalid_execution_quantity_poisoned_basis(self):
        for quantity in ("NaN", "Infinity", "-1", "2", "0", True):
            with self.subTest(quantity=quantity):
                records = [order(1, "buy", filled_quantity=quantity), order(2, "buy"),
                           order(3, "sell", price="110")]
                metric = self.metric(records, 3)
                self.assertEqual(metric["reason_code"], "invalid_quantity")
                self.assertIsNone(metric["realized_profit"])

    def test_executed_quantity_requires_positive_requested_quantity_even_with_average_proof(self):
        for requested in (None, "0", "-1", "NaN", "Infinity", "bad", True):
            for side in ("buy", "sell"):
                with self.subTest(requested=requested, side=side):
                    malformed = order(2, side, quantity=requested, filled_quantity="1", status="accepted",
                                      price_basis="weighted_fills", price_basis_quantity="1", price_basis_price="100")
                    records = [order(1, "buy"), malformed, order(3, "sell", price="110")]
                    result = realized_performance(records)["by_rule_id"]
                    self.assertEqual(result["2"]["reason_code"], "invalid_quantity")
                    self.assertEqual(result["2"]["price_reason_code"], "invalid_quantity")
                    self.assertIsNone(result["2"]["effective_fill_price"])
                    self.assertEqual(result["3"]["reason_code"], "invalid_quantity")
                    self.assertIsNone(result["3"]["realized_profit"])

    def test_explicit_remaining_must_be_finite_nonnegative_and_within_requested_total(self):
        for remaining in ("1", "-1", "NaN", "Infinity", "bad", True):
            for status in ("filled", "accepted", "cancelled"):
                with self.subTest(remaining=remaining, status=status):
                    records = [order(1, "buy"),
                               order(2, "sell", price="110", status=status, remaining_quantity=remaining)]
                    result = self.metric(records, 2)
                    self.assertEqual(result["reason_code"], "invalid_quantity")
                    self.assertIsNone(result["effective_fill_price"])
                    self.assertIsNone(result["realized_profit"])

    def test_filled_status_requires_entire_requested_quantity_and_no_remaining(self):
        for executed, remaining in (("1", "0"), ("1", "1"), ("1", None), ("2", "1")):
            with self.subTest(executed=executed, remaining=remaining):
                result = self.metric([order(1, "buy", 2),
                                      order(2, "sell", 2, "110", filled_quantity=executed,
                                            remaining_quantity=remaining)], 2)
                self.assertEqual(result["reason_code"], "invalid_quantity")
                self.assertIsNone(result["realized_profit"])

    def test_legacy_inferred_full_fill_accepts_absent_remaining_but_not_contradiction(self):
        for remaining in (None, "0"):
            with self.subTest(remaining=remaining):
                result = self.metric([order(1, "buy", filled_quantity=None, remaining_quantity=remaining),
                                      order(2, "sell", price="110", filled_quantity=None,
                                            remaining_quantity=remaining)], 2)
                self.assertTrue(result["quantity_inferred"])
                self.assertEqual(result["realized_profit"], D("10"))
        for remaining in ("1", "NaN", "-1"):
            with self.subTest(remaining=remaining):
                result = self.metric([order(1, "buy"), order(2, "sell", filled_quantity=None,
                                                            remaining_quantity=remaining)], 2)
                self.assertTrue(result["quantity_inferred"])
                self.assertEqual(result["reason_code"], "invalid_quantity")
                self.assertIsNone(result["realized_profit"])

    def test_invalid_timestamp_or_identity_fail_closed(self):
        for timestamp in (None, "bad", "2026-09-15T00:00:00"):
            with self.subTest(timestamp=timestamp):
                metric = self.metric([order(1, "buy", started_at=timestamp), order(2, "sell", price="110")], 2)
                self.assertEqual(metric["reason_code"], "invalid_time")
        metric = self.metric([order(1, "buy", exchange=""), order(2, "sell", price="110", exchange="")], 2)
        self.assertEqual(metric["reason_code"], "invalid_instrument")

    def test_partly_identified_old_buy_cannot_disappear_from_otherwise_known_basis(self):
        records = [order(1, "buy", price="90", exchange=""), order(2, "buy", price="100"),
                   order(3, "sell", price="110"), order(4, "buy", price="200", symbol="000660"),
                   order(5, "sell", price="220", symbol="000660")]
        result = realized_performance(records)["by_rule_id"]
        self.assertEqual(result["3"]["reason_code"], "invalid_instrument")
        self.assertIsNone(result["3"]["realized_profit"])
        self.assertEqual(result["5"]["realized_profit"], D("20"))

    def test_zero_negative_and_nonfinite_prices_are_unknown_not_fallback_quotes(self):
        for price in (None, "0", "-1", "NaN", "Infinity", "", True):
            with self.subTest(price=price):
                metric = self.metric([order(1, "buy", price=price, reference_price="100"),
                                      order(2, "sell", price="110")], 2)
                self.assertEqual(metric["reason_code"], "missing_buy_price")
                self.assertIsNone(metric["cost_basis"])

    def test_empty_buy_only_and_input_are_not_mutated(self):
        self.assertEqual(realized_performance([])["summaries"], {})
        records = [order(1, "buy", price="100")]
        original = copy.deepcopy(records)
        result = realized_performance(iter(records))
        self.assertEqual(result["by_rule_id"]["1"]["status"], "not_applicable")
        self.assertEqual(result["summaries"], {})
        self.assertEqual(records, original)


if __name__ == "__main__":
    unittest.main()
