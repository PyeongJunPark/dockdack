from __future__ import annotations

import unittest
from datetime import date, datetime
from decimal import Decimal

from dockdack import BrokerAPIError, KiwoomBroker, Market, OrderSide, TradingMode
from dockdack.gui_service import TradingService
from dockdack.http import _READ_ONLY_APIS
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


DAY = date(2026, 9, 15)


def domestic_row(**overrides):
    return {
        "ord_no": "0000050", "stk_cd": "A005930", "ord_qty": "0000000001",
        "ord_uv": "0000000000", "cntr_qty": "0000000001", "cntr_uv": "0000248500",
        "ord_remnq": "0000000000", "io_tp_nm": "현금매수", "acpt_tp": "접수",
        "ord_tm": "13:05:43", "ori_ord": "0000000", "dmst_stex_tp": "KRX",
        **overrides,
    }


def us_row(**overrides):
    return {
        "ord_no": "000000252", "stk_cd": "NVDA", "crnc_code": "USD", "stex_nm": "미국",
        "ord_qty": "000000000001", "cntr_qty": "000000000001", "ord_uv": "201.0200",
        "cntr_uv": "201.3147", "ord_remnq": "000000000000", "slby_tp_nm": "매도",
        "ord_stat_nm": "체결완료", "ord_time": "21:04:41", "cntr_time": "21:05:41",
        **overrides,
    }


class ExecutionHistoryTests(unittest.TestCase):
    def service(self, *responses, mode=TradingMode.DEMO):
        transport = QueueTransport(token_response(), *responses)
        broker = KiwoomBroker(config(mode), transport=transport)
        return TradingService(lambda _: broker), broker, transport

    def test_domestic_account_day_query_and_exact_single_share_price(self):
        service, _, transport = self.service(FakeResponse({"acnt_ord_cntr_prps_dtl": [domestic_row()]}))
        rows = service.execution_history(Market.DOMESTIC, DAY)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row.market, row.order_date, row.currency), (Market.DOMESTIC, DAY, "KRW"))
        self.assertEqual((row.order_number, row.symbol, row.exchange, row.side),
                         ("0000050", "005930", "KRX", OrderSide.BUY))
        self.assertEqual((row.order_quantity, row.filled_quantity, row.remaining_quantity),
                         (Decimal(1), Decimal(1), Decimal(0)))
        self.assertEqual((row.fill_price, row.reported_fill_price, row.price_basis),
                         (Decimal("248500"), Decimal("248500"), "single_share"))
        self.assertIsNone(row.order_price)  # A market order's 0 is not its execution price.
        self.assertEqual((row.order_time, row.fill_time, row.status), ("13:05:43", "", "접수"))
        call = transport.calls[-1]
        self.assertEqual(call["headers"]["api-id"], "kt00007")
        self.assertEqual(call["json"], {"ord_dt": "20260915", "qry_tp": "1", "stk_bond_tp": "1",
                         "sell_tp": "0", "stk_cd": "", "fr_ord_no": "", "dmst_stex_tp": "%"})
        self.assertEqual(len(transport.calls), 2)  # Authentication plus one account-wide read.

    def test_us_documented_response_typo_and_decimal_precision(self):
        for key in ("result_list", "result_lsit"):
            with self.subTest(key=key):
                service, _, transport = self.service(FakeResponse({key: [us_row()]}))
                row, = service.execution_history(Market.US, DAY)
                self.assertEqual((row.market, row.order_date, row.exchange, row.currency),
                                 (Market.US, DAY, "", "USD"))
                self.assertEqual((row.side, row.fill_price, row.order_price),
                                 (OrderSide.SELL, Decimal("201.3147"), Decimal("201.0200")))
                self.assertEqual((row.order_time, row.fill_time), ("21:04:41", "21:05:41"))
                call = transport.calls[-1]
                self.assertEqual(call["headers"]["api-id"], "ust21150")
                self.assertEqual(call["json"], {"ord_dt": "20260915", "query_tp": "1", "slby_tp": "0",
                                 "stex_tp": "", "stk_cd": "", "oppo_trde_tp": "%", "fr_ord_no": ""})

    def test_dated_history_does_not_substitute_today(self):
        service, _, transport = self.service(FakeResponse({"acnt_ord_cntr_prps_dtl": []}))
        self.assertEqual(service.execution_history(Market.DOMESTIC, date(2026, 9, 11)), ())
        self.assertEqual(transport.calls[-1]["json"]["ord_dt"], "20260911")

    def test_empty_valid_history_is_distinct_from_missing_list(self):
        for market, key in ((Market.DOMESTIC, "acnt_ord_cntr_prps_dtl"), (Market.US, "result_list")):
            service, _, _ = self.service(FakeResponse({key: []}))
            self.assertEqual(service.execution_history(market, DAY), ())
            for body in ({}, {key: None}, {key: {}}, {key: [None]}, {key: [["0000050"]]}):
                with self.subTest(market=market, body=body):
                    service, _, _ = self.service(FakeResponse(body))
                    with self.assertRaises(ValueError):
                        service.execution_history(market, DAY)

    def test_unknown_fill_price_is_not_replaced_by_order_price(self):
        for value in (None, "", " ", "0", "0.0000"):
            service, _, _ = self.service(FakeResponse({"result_list": [us_row(cntr_uv=value)]}))
            row, = service.execution_history(Market.US, DAY)
            self.assertEqual(row.order_price, Decimal("201.0200"))
            self.assertIsNone(row.fill_price)
            self.assertEqual(row.price_basis, "missing")

    def test_multishare_price_is_preserved_but_not_assumed_average(self):
        for filled, remaining in (("2", "0"), ("1", "1"), ("1", "0")):
            service, _, _ = self.service(FakeResponse({"result_list": [
                us_row(ord_qty="2", cntr_qty=filled, ord_remnq=remaining)]}))
            row, = service.execution_history(Market.US, DAY)
            self.assertEqual(row.reported_fill_price, Decimal("201.3147"))
            self.assertIsNone(row.fill_price)
            self.assertEqual(row.price_basis, "unverified_multi_share")

    def test_no_fill_does_not_publish_a_usable_price(self):
        service, _, _ = self.service(FakeResponse({"result_list": [us_row(cntr_qty="0", ord_remnq="1")]}))
        row, = service.execution_history(Market.US, DAY)
        self.assertIsNone(row.fill_price)
        self.assertEqual(row.price_basis, "not_filled")

    def test_malformed_identity_direction_currency_exchange_rejected(self):
        cases = [("ord_no", ""), ("ord_no", "0000000"), ("ord_no", 50), ("ord_no", "a50"),
                 ("stk_cd", "Z123456"), ("dmst_stex_tp", ""), ("dmst_stex_tp", "UNKNOWN"),
                 ("io_tp_nm", "접수"), ("io_tp_nm", "매수매도")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                service, _, _ = self.service(FakeResponse({"acnt_ord_cntr_prps_dtl": [domestic_row(**{field: value})]}))
                with self.assertRaises(ValueError):
                    service.execution_history(Market.DOMESTIC, DAY)
        service, _, _ = self.service(FakeResponse({"result_list": [us_row(crnc_code="KRW")]}))
        with self.assertRaisesRegex(ValueError, "통화"):
            service.execution_history(Market.US, DAY)

    def test_malformed_or_inconsistent_quantities_rejected(self):
        cases = [{field: value} for field in ("ord_qty", "cntr_qty", "ord_remnq")
                 for value in (None, "", "NaN", "Infinity", "-1", "0.5")]
        cases.extend(({"ord_qty": "0"}, {"cntr_qty": "2"}, {"ord_remnq": "1"}))
        for changes in cases:
            with self.subTest(changes=changes):
                service, _, _ = self.service(FakeResponse({"result_list": [us_row(**changes)]}))
                with self.assertRaises(ValueError):
                    service.execution_history(Market.US, DAY)

    def test_nonstock_account_instrument_prefixes_cannot_alias_stock(self):
        for symbol in ("Q123456", "J123456"):
            service, _, _ = self.service(FakeResponse({"acnt_ord_cntr_prps_dtl": [domestic_row(stk_cd=symbol)]}))
            row, = service.execution_history(Market.DOMESTIC, DAY)
            self.assertEqual(row.symbol, symbol)

    def test_nonfinite_malformed_and_negative_prices_rejected(self):
        for field in ("ord_uv", "cntr_uv"):
            for value in ("NaN", "Infinity", "bad", "-1"):
                with self.subTest(field=field, value=value):
                    service, _, _ = self.service(FakeResponse({"result_list": [us_row(**{field: value})]}))
                    with self.assertRaises(ValueError):
                        service.execution_history(Market.US, DAY)

    def test_pages_aggregate_and_identical_duplicates_deduplicate(self):
        first = FakeResponse({"result_list": [us_row()]}, headers={"cont-yn": "Y", "next-key": "page2"})
        second = FakeResponse({"result_list": [us_row(), us_row(ord_no="000000253")]})
        service, _, transport = self.service(first, second)
        self.assertEqual(len(service.execution_history(Market.US, DAY)), 2)
        self.assertEqual(transport.calls[-1]["headers"]["next-key"], "page2")
        self.assertEqual(transport.calls[-1]["json"]["ord_dt"], "20260915")

    def test_conflicting_same_order_is_not_double_counted(self):
        service, _, _ = self.service(FakeResponse({"result_list": [us_row(), us_row(cntr_uv="202")]}))
        with self.assertRaisesRegex(ValueError, "동일 주문번호"):
            service.execution_history(Market.US, DAY)

    def test_incomplete_pagination_never_returns_partial_history(self):
        response = FakeResponse({"result_list": [us_row()]}, headers={"cont-yn": "Y", "next-key": "more"})
        _, broker, _ = self.service(response)
        with self.assertRaisesRegex(BrokerAPIError, "최대 조회 페이지"):
            broker.list_execution_history(Market.US, DAY, max_pages=1)

    def test_invalid_dates_and_page_limits_fail_before_network(self):
        for day in ("20260915", datetime(2026, 9, 15), None):
            _, broker, transport = self.service()
            with self.assertRaises(ValueError):
                broker.list_execution_history(Market.US, day)
            self.assertEqual(transport.calls, [])
        for pages in (0, 101, True, 1.5):
            _, broker, transport = self.service()
            with self.assertRaises(ValueError):
                broker.list_execution_history(Market.US, DAY, max_pages=pages)
            self.assertEqual(transport.calls, [])

    def test_gui_service_still_refuses_live_broker(self):
        service, _, transport = self.service(mode=TradingMode.REAL)
        with self.assertRaisesRegex(ValueError, "모의투자 전용"):
            service.execution_history(Market.US, DAY)
        self.assertEqual(transport.calls, [])

    def test_new_history_reads_use_only_exact_reviewed_retry_routes(self):
        self.assertIn(("kt00007", "/api/dostk/acnt"), _READ_ONLY_APIS)
        self.assertIn(("ust21150", "/api/us/acnt"), _READ_ONLY_APIS)
        self.assertNotIn(("kt00007", "/api/dostk/ordr"), _READ_ONLY_APIS)
        self.assertNotIn(("ust21150", "/api/us/ordr"), _READ_ONLY_APIS)


if __name__ == "__main__":
    unittest.main()
