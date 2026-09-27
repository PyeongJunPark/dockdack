from __future__ import annotations

import copy
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D

from dockdack.trade_journal import daily_trade_journal, empty_day, order_day, period_trade_journal


START = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)


def order(number, side="buy", price="100", **overrides):
    record = {
        "rule_id": str(number), "watch_id": "domestic:KRX:005930",
        "started_at": (START + timedelta(seconds=number)).isoformat(),
        "status": "filled", "order_number": str(number), "market": "domestic",
        "exchange": "KRX", "symbol": "005930", "name": "삼성전자", "currency": "KRW", "side": side,
        "quantity": 1, "filled_quantity": "1", "fill_price": price,
        "remaining_quantity": "0", "observed_at": (START + timedelta(days=2)).isoformat(),
        "reference_price": "9999", "message": "test fixture",
    }
    record.update(overrides)
    return record


class DailyTradeJournalTests(unittest.TestCase):
    def day(self, records, day=date(2026, 9, 15), market="domestic"):
        return daily_trade_journal(records)["days"][(market, day)]

    def test_prior_day_buy_costs_current_day_sale_from_entire_history(self):
        rows = [order(1, started_at=(START - timedelta(days=1)).isoformat()), order(2, "sell", "110")]
        result = self.day(rows)
        self.assertEqual(result["buy_amount"], D(0))
        self.assertEqual(result["sell_amount"], D(110))
        self.assertEqual(result["realized_profit"], D(10))
        self.assertEqual(result["return_pct"], D(10))
        self.assertEqual(result["known_cost_basis"], D(100))

    def test_market_and_currency_are_never_aggregated(self):
        rows = [order(1), order(2, "sell", "110"),
                order(3, market="us", exchange="ND", symbol="AAPL", currency="USD"),
                order(4, "sell", "102", market="us", exchange="ND", symbol="AAPL", currency="USD")]
        result = daily_trade_journal(rows)
        kr = result["days"][("domestic", date(2026, 9, 15))]
        us = result["days"][("us", date(2026, 9, 14))]
        self.assertEqual(kr["realized_profit"], D(10))
        self.assertEqual(us["realized_profit"], D(2))
        self.assertEqual(len(result["days"]), 2)
        self.assertNotIn("total_profit", result)

    def test_accepted_unknown_and_rejected_do_not_create_cash_from_reference(self):
        result = self.day([order(i, status=status, filled_quantity=None, fill_price=None)
                           for i, status in enumerate(("accepted", "unknown", "submitting", "rejected", "not_sent"), 1)])
        self.assertEqual(result["buy_amount"], D(0))
        self.assertEqual(result["buy_count"], 0)
        self.assertEqual(result["pending_count"], 3)
        self.assertEqual(result["rejected_count"], 1)
        self.assertIsNone(result["return_pct"])

    def test_legacy_filled_is_not_reference_price_cash(self):
        result = self.day([order(1, fill_price=None, filled_quantity=None)])
        self.assertEqual(result["buy_count"], 1)
        self.assertIsNone(result["buy_amount"])
        self.assertEqual(result["unknown_buy_count"], 1)
        self.assertEqual(result["known_buy_amount"], D(0))

    def test_partial_cancelled_fill_uses_verified_average_only(self):
        row = order(1, status="cancelled", quantity=4, filled_quantity="2", fill_price="123")
        result = self.day([row])
        self.assertEqual(result["buy_quantity"], D(2))
        self.assertIsNone(result["buy_amount"])
        row.update(price_basis="weighted_fills", price_basis_quantity="2", price_basis_price="123")
        self.assertEqual(self.day([row])["buy_amount"], D(246))

    def test_profit_unknown_without_cost_but_sell_cash_known(self):
        result = self.day([order(1, "sell", "110")])
        self.assertEqual(result["sell_amount"], D(110))
        self.assertIsNone(result["realized_profit"])
        self.assertIsNone(result["return_pct"])
        self.assertEqual(result["unknown_profit_count"], 1)

    def test_known_subtotals_are_separate_from_incomplete_totals(self):
        result = self.day([order(1), order(2, "sell", "110"), order(3, "sell", None, symbol="000001")])
        self.assertIsNone(result["sell_amount"])
        self.assertEqual(result["known_sell_amount"], D(110))
        self.assertIsNone(result["realized_profit"])
        self.assertEqual(result["known_realized_profit"], D(10))
        self.assertEqual(result["known_return_pct"], D(10))
        self.assertEqual(result["unknown_profit_count"], 1)
        self.assertFalse(result["complete"])

    def test_weighted_cost_return_is_not_average_of_percentage_returns(self):
        rows = [order(1, price="100"), order(2, "sell", "110"),
                order(3, price="900"), order(4, "sell", "900")]
        result = self.day(rows)
        self.assertEqual(result["return_pct"], D(1))
        self.assertEqual(result["buy_amount"], D(1000))
        self.assertEqual(result["sell_amount"], D(1010))

    def test_order_date_not_observation_or_ambiguous_broker_fill_date(self):
        row = order(1, market="us", currency="USD", exchange="ND", symbol="AAPL",
                    recovery_order_date="2026-09-15", recovery_fill_time="030100")
        self.assertEqual(order_day(row), (date(2026, 9, 14), "20:00:01"))
        result = daily_trade_journal([row])
        self.assertIn(("us", date(2026, 9, 14)), result["days"])
        self.assertIn("정확한 체결일", result["date_description"])

    def test_dst_and_standard_time_use_new_york_not_fixed_offset(self):
        summer = order(1, market="us", started_at="2026-07-02T04:30:00+00:00")
        winter = order(2, market="us", started_at="2026-01-02T04:30:00+00:00")
        self.assertEqual(order_day(summer)[0], date(2026, 7, 2))
        self.assertEqual(order_day(winter)[0], date(2026, 1, 1))

    def test_invalid_date_not_assigned_to_today(self):
        for value in ("invalid", "2026-09-15T09:00:00", None):
            with self.subTest(value=value):
                result = daily_trade_journal([order(1, started_at=value)])
                self.assertFalse(result["days"])
                self.assertEqual(len(result["undated"]), 1)

    def test_duplicate_order_is_not_double_counted(self):
        row = order(1)
        result = self.day([row, row.copy()])
        self.assertEqual(result["order_count"], 1)
        self.assertEqual(result["invalid_count"], 1)
        self.assertIsNone(result["buy_amount"])
        self.assertEqual(result["known_buy_amount"], D(0))

    def test_invalid_executed_quantity_is_unknown_not_zero_cash(self):
        result = self.day([order(1, filled_quantity="NaN")])
        self.assertEqual(result["invalid_count"], 1)
        self.assertIsNone(result["buy_amount"])

    def test_wrong_currency_is_explicitly_excluded(self):
        result = daily_trade_journal([order(1, currency="USD")])
        self.assertFalse(result["days"])
        self.assertEqual(len(result["undated"]), 1)

    def test_zero_profit_is_known_but_no_sales_is_undefined_return(self):
        result = self.day([order(1), order(2, "sell", "100")])
        self.assertEqual(result["realized_profit"], D(0))
        self.assertEqual(result["return_pct"], D(0))
        empty = empty_day("domestic", date(2026, 9, 15))
        self.assertEqual(empty["buy_amount"], D(0))
        self.assertIsNone(empty["return_pct"])

    def test_calculation_never_mutates_original_ledger(self):
        rows = [order(1), order(2, "sell")]
        original = copy.deepcopy(rows)
        daily_trade_journal(rows)
        self.assertEqual(rows, original)

    def test_month_and_year_group_existing_market_local_days_without_mixing_currency(self):
        records = [
            order(1),
            order(2, "sell", "110", started_at=(START + timedelta(days=1)).isoformat()),
            order(3, price="900", started_at="2026-10-02T00:00:00+00:00"),
            order(4, "sell", "900", started_at="2026-10-03T00:00:00+00:00"),
            order(5, market="us", exchange="ND", symbol="AAPL", currency="USD"),
            order(6, market="us", exchange="ND", symbol="AAPL", currency="USD",
                  started_at="2026-10-01T00:15:00+00:00"),
        ]
        journal = daily_trade_journal(records)
        original = copy.deepcopy(journal["days"])
        september = period_trade_journal(journal, "domestic", date(2026, 9, 30), "month")
        october = period_trade_journal(journal, "domestic", date(2026, 10, 1), "month")
        year = period_trade_journal(journal, "domestic", date(2026, 1, 1), "year")
        us = period_trade_journal(journal, "us", date(2026, 9, 1), "month")
        self.assertEqual((september["buy_amount"], september["sell_amount"],
                          september["realized_profit"], september["return_pct"]),
                         (D(100), D(110), D(10), D(10)))
        self.assertEqual(october["realized_profit"], D(0))
        self.assertEqual((year["buy_amount"], year["sell_amount"], year["realized_profit"]),
                         (D(1000), D(1010), D(10)))
        self.assertEqual(year["return_pct"], D(1))
        self.assertEqual([row["rule_id"] for row in year["rows"]], ["1", "2", "3", "4"])
        # October 1 UTC is still September 30 in New York.
        self.assertEqual((us["currency"], us["buy_amount"], us["order_count"]), ("USD", D(200), 2))
        self.assertEqual(period_trade_journal(journal, "us", date(2026, 10, 1), "month")["order_count"], 0)
        self.assertEqual(journal["days"], original)

    def test_period_unknown_profit_and_cash_remain_unknown_with_known_subtotals(self):
        records = [order(1), order(2, "sell", "110"),
                   order(3, "sell", "105", symbol="000001",
                         started_at=(START + timedelta(days=1)).isoformat()),
                   order(4, "buy", price=None, filled_quantity=None,
                         started_at="2026-10-02T00:00:00+00:00")]
        journal = daily_trade_journal(records)
        september = period_trade_journal(journal, "domestic", date(2026, 9, 1), "month")
        year = period_trade_journal(journal, "domestic", date(2026, 12, 31), "year")
        self.assertEqual(september["known_realized_profit"], D(10))
        self.assertIsNone(september["realized_profit"])
        self.assertEqual(september["known_return_pct"], D(10))
        self.assertIsNone(september["return_pct"])
        self.assertEqual(september["unknown_profit_count"], 1)
        self.assertIsNone(year["buy_amount"])
        self.assertEqual(year["known_buy_amount"], D(100))
        self.assertEqual(year["unknown_buy_count"], 1)
        self.assertIsNone(year["realized_profit"])
        self.assertFalse(year["complete"])
        self.assertEqual(period_trade_journal(journal, "domestic", date(2026, 8, 1), "month")["order_count"], 0)

    def test_period_rejects_unknown_market_or_period(self):
        journal = daily_trade_journal(())
        with self.assertRaises(ValueError):
            period_trade_journal(journal, "unknown", date(2026, 9, 1), "month")
        with self.assertRaises(ValueError):
            period_trade_journal(journal, "domestic", date(2026, 9, 1), "week")


if __name__ == "__main__":
    unittest.main()
