from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from dockdack import BrokerAPIError, KiwoomBroker, Market
from dockdack.history import market_time, regular_session
from dockdack.http import KiwoomHTTPClient
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


def row(day, price="100", us=False):
    return {"dt": day, "cur_prc": price, "open_pric": "98", "high_pric": "102", "low_pric": "97",
            "acc_trde_qty" if us else "trde_qty": "1234"}


class HistoryTests(unittest.TestCase):
    def test_domestic_payload_sign_normalization_sort_and_trim(self):
        transport = QueueTransport(token_response(), FakeResponse({"stk_cd": "005930", "stk_dt_pole_chart_qry": [
            row("20260914", "-100"), row("20260911", "+99"), row("20260910", "98"),
        ]}))
        result = KiwoomBroker(config(), transport=transport).daily_history(
            "domestic", "005930", exchange="KRX", days=2, as_of=date(2026, 9, 14))
        self.assertEqual([bar.day for bar in result.bars], [date(2026, 9, 11), date(2026, 9, 14)])
        self.assertEqual(result.bars[-1].close, Decimal(100))
        self.assertTrue(result.complete)
        self.assertEqual(transport.calls[-1]["headers"]["api-id"], "ka10081")
        self.assertEqual(transport.calls[-1]["json"], {"stk_cd": "005930", "base_dt": "20260914", "upd_stkpc_tp": "1"})

    def test_us_payload_uses_usd_no_fx_conversion_and_start_date(self):
        transport = QueueTransport(token_response(), FakeResponse({"result_list": [row("20260911", "100.1234", True)]}))
        result = KiwoomBroker(config(), transport=transport).daily_history("us", "AAPL", exchange="ND", days=30, as_of=date(2026, 9, 14))
        self.assertEqual(result.currency, "USD")
        self.assertEqual(result.bars[0].close, Decimal("100.1234"))
        self.assertFalse(result.complete)
        self.assertEqual(transport.calls[-1]["headers"]["api-id"], "usa06012")
        self.assertEqual(transport.calls[-1]["json"], {"stex_tp": "ND", "stk_cd": "AAPL", "strt_dt": "20260914", "upd_stkpc_tp": "1", "exrt_appl_tp": "0"})

    def test_pagination_deduplicates_boundary_date(self):
        transport = QueueTransport(token_response(),
            FakeResponse({"stk_dt_pole_chart_qry": [row("20260914")]}, headers={"cont-yn": "Y", "next-key": "next"}),
            FakeResponse({"stk_dt_pole_chart_qry": [row("20260914"), row("20260911")]}))
        result = KiwoomBroker(config(), transport=transport).daily_history("domestic", "005930", exchange="KRX", days=2)
        self.assertEqual(len(result.bars), 2)
        self.assertEqual(transport.calls[-1]["headers"]["next-key"], "next")

    def test_invalid_days_fail_before_http(self):
        for days in (0, -1, 1.5, True, 1001):
            transport = QueueTransport()
            with self.assertRaises(ValueError):
                KiwoomBroker(config(), transport=transport).daily_history("domestic", "005930", exchange="KRX", days=days)
            self.assertEqual(transport.calls, [])

    def test_malformed_or_empty_chart_data_is_not_a_valid_history(self):
        bodies = [{}, {"stk_dt_pole_chart_qry": []}, {"stk_dt_pole_chart_qry": ["bad"]},
                  {"stk_dt_pole_chart_qry": [row("bad")]},
                  {"stk_dt_pole_chart_qry": [row("20260914", "NaN")]},
                  {"stk_dt_pole_chart_qry": [row("20260914", "0")]},
                  {"stk_cd": "000660", "stk_dt_pole_chart_qry": [row("20260914")]}]
        for body in bodies:
            with self.subTest(body=body):
                transport = QueueTransport(token_response(), FakeResponse(body))
                with self.assertRaises((ValueError, BrokerAPIError)):
                    KiwoomBroker(config(), transport=transport).daily_history("domestic", "005930", exchange="KRX")

    def test_conflicting_duplicate_dates_fail(self):
        transport = QueueTransport(token_response(), FakeResponse({"stk_dt_pole_chart_qry": [row("20260914"), row("20260914", "101")]}))
        with self.assertRaises(BrokerAPIError):
            KiwoomBroker(config(), transport=transport).daily_history("domestic", "005930", exchange="KRX")

    def test_unfinished_pagination_does_not_silently_return_partial_data(self):
        for headers in ({"cont-yn": "Y"}, {"cont-yn": "Y", "next-key": "more"}):
            transport = QueueTransport(token_response(), FakeResponse({}, headers=headers))
            client = KiwoomHTTPClient(config(), transport=transport)
            with self.assertRaises(BrokerAPIError):
                tuple(client.iter_pages(api_id="test", path="/test", max_pages=1))

    def test_exchange_time_zones_and_regular_hours(self):
        summer = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc)
        winter = datetime(2026, 12, 14, 14, 30, tzinfo=timezone.utc)
        self.assertEqual(market_time(Market.US, summer).hour, 9)
        self.assertEqual(market_time(Market.US, winter).hour, 9)
        self.assertTrue(regular_session(Market.US, summer))
        self.assertTrue(regular_session(Market.US, winter))
        self.assertFalse(regular_session(Market.US, datetime(2026, 9, 14, 20, tzinfo=timezone.utc)))
        self.assertFalse(regular_session(Market.DOMESTIC, datetime(2026, 9, 12, 1, tzinfo=timezone.utc)))
        self.assertTrue(regular_session(Market.DOMESTIC, datetime(2026, 9, 14, 0, tzinfo=timezone.utc)))
        self.assertFalse(regular_session(Market.DOMESTIC, datetime(2026, 9, 14, 6, 30, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
