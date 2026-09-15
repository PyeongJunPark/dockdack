from datetime import date
import unittest

from dockdack.equity_universe import (
    EQUITY_MARKETS, EXCLUDED_MARKETS, domestic_common_symbols, is_domestic_common_row,
)
from dockdack.exceptions import BrokerAPIError
from dockdack.http import APIPage, KiwoomHTTPClient
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


def common(code="005930", name="삼성전자", market="0", **changes):
    row = dict(code=code, name=name, marketCode=market,
               marketName="거래소" if market == "0" else "코스닥",
               upName="전기전자", companyClassName="", upSizeName="대형주", state="정상")
    row.update(changes)
    return row


class MasterHTTP:
    def __init__(self, groups=None):
        self.groups = {"0": [common()], "10": [common("035900", "JYP Ent.", "10")]}
        self.groups.update(groups or {})
        self.calls = []
        self.fail_market = None

    def iter_pages(self, *, api_id, path, body, max_pages):
        self.calls.append((api_id, path, body["mrkt_tp"]))
        if body["mrkt_tp"] == self.fail_market:
            raise BrokerAPIError("master unavailable")
        yield APIPage({"list": self.groups.get(body["mrkt_tp"], [])}, False, None, None)


class DomesticEquityUniverseTests(unittest.TestCase):
    def test_positive_exchange_membership_and_common_shape(self):
        self.assertTrue(is_domestic_common_row(common()))
        self.assertTrue(is_domestic_common_row(common("0068Y0", "새일반기업", "10")))
        self.assertTrue(is_domestic_common_row(common("A005930")))
        self.assertTrue(is_domestic_common_row(common("138040", "메리츠금융지주", upName="기타금융")))
        self.assertFalse(is_domestic_common_row(common("005935", "삼성전자우")))
        self.assertFalse(is_domestic_common_row(common("00593K", "삼성전자2우B")))
        self.assertFalse(is_domestic_common_row(common(market="50")))

    def test_missing_unknown_or_malformed_metadata_fails_closed(self):
        for key in ("name", "marketCode", "marketName", "upName", "companyClassName"):
            row = common()
            del row[key]
            with self.subTest(key=key):
                self.assertFalse(is_domestic_common_row(row))
        for changes in ({"marketCode": "unknown"}, {"marketName": "unknown"},
                        {"marketName": "코스닥"}, {"upName": ""}, {"companyClassName": None},
                        {"companyClassName": "새로운분류"}, {"kind": "unknown"},
                        {"isEtf": "unknown"}, {"isEtf": None}, {"state": {}}):
            self.assertFalse(is_domestic_common_row(common(**changes)))
        for flag in (False, 0, "0", "N"):
            self.assertTrue(is_domestic_common_row(common(isEtf=flag)))

    def test_all_excluded_product_categories_override_positive_row(self):
        for market in EXCLUDED_MARKETS:
            with self.subTest(market=market):
                http = MasterHTTP({market: [common()]})
                self.assertEqual(domestic_common_symbols(http), frozenset({"035900"}))
        for market in EXCLUDED_MARKETS:
            self.assertFalse(is_domestic_common_row(common(market=market)))

    def test_etf_spac_reit_fund_and_preferred_guards(self):
        for field, value in (("isEtf", True), ("upName", "기업인수목적회사"),
                             ("companyClassName", "SPAC"), ("upName", "부동산투자회사"),
                             ("upName", "뮤추얼펀드"), ("name", "ETF ABC"),
                             ("name", "ACE 반도체 ETN"), ("name", "키움제9호스팩"),
                             ("name", "회사우B"), ("name", "회사2우B"),
                             ("name", "회사리츠"), ("state", "우선주")):
            with self.subTest(field=field, value=value):
                self.assertFalse(is_domestic_common_row(common(**{field: value})))

    def test_cache_daily_per_client_and_forced_refresh(self):
        http = MasterHTTP()
        first_day, next_day = date(2026, 9, 15), date(2026, 9, 16)
        first = domestic_common_symbols(http, as_of=first_day)
        calls_per_load = len(EQUITY_MARKETS) + len(EXCLUDED_MARKETS)
        self.assertEqual(len(http.calls), calls_per_load)
        self.assertIs(domestic_common_symbols(http, as_of=first_day), first)
        self.assertEqual(len(http.calls), calls_per_load)
        domestic_common_symbols(http, as_of=next_day)
        domestic_common_symbols(http, as_of=next_day, force=True)
        self.assertEqual(len(http.calls), calls_per_load * 3)
        other = MasterHTTP({"0": [common("000660", "SK하이닉스")]})
        self.assertNotIn("005930", domestic_common_symbols(other, as_of=next_day))

    def test_partial_failure_is_not_cached_or_replaced_with_previous_day(self):
        http = MasterHTTP()
        domestic_common_symbols(http, as_of=date(2026, 9, 15))
        http.fail_market = "8"
        for day in (date(2026, 9, 16), date(2026, 9, 16)):
            with self.assertRaises(BrokerAPIError):
                domestic_common_symbols(http, as_of=day)
        http.fail_market = None
        self.assertIn("005930", domestic_common_symbols(http, as_of=date(2026, 9, 16)))

    def test_failed_force_refresh_invalidates_old_same_day_cache(self):
        http = MasterHTTP()
        domestic_common_symbols(http)
        http.fail_market = "8"
        with self.assertRaises(BrokerAPIError):
            domestic_common_symbols(http, force=True)
        with self.assertRaises(BrokerAPIError):
            domestic_common_symbols(http)

    def test_ambiguous_duplicates_excluded_and_malformed_lists_rejected(self):
        http = MasterHTTP({"10": [common(market="10"), common("035900", "JYP Ent.", "10")]})
        self.assertNotIn("005930", domestic_common_symbols(http))
        for value in (None, {}, ["not a master row"], [{"code": 5930}], []):
            with self.subTest(value=value), self.assertRaises(BrokerAPIError):
                domestic_common_symbols(MasterHTTP({"0": value}))

    def test_real_http_pagination_must_complete_before_cache(self):
        responses = [token_response(), FakeResponse({"list": [common()]},
                     headers={"cont-yn": "Y", "next-key": "second"}),
                     FakeResponse({"list": [common("000660", "SK하이닉스")]}),
                     FakeResponse({"list": [common("035900", "JYP Ent.", "10")]})]
        responses.extend(FakeResponse({"list": []}) for _ in EXCLUDED_MARKETS)
        transport = QueueTransport(*responses)
        result = domestic_common_symbols(KiwoomHTTPClient(config(), transport=transport))
        self.assertEqual(result, frozenset({"005930", "000660", "035900"}))
        self.assertEqual(transport.calls[2]["headers"]["next-key"], "second")
        self.assertTrue(all(call["headers"].get("api-id") in (None, "ka10099") for call in transport.calls))

    def test_unfinished_generator_rejected(self):
        class BrokenMaster(MasterHTTP):
            def iter_pages(self, **kwargs):
                yield APIPage({"list": [common()]}, True, "more", "Y")
        with self.assertRaises(BrokerAPIError):
            domestic_common_symbols(BrokenMaster())

    def test_real_master_issuer_classes_and_product_rows_mixed_in_kospi(self):
        for issuer_class in ("", "벤처기업", "신성장기업", "외국기업", "우량기업", "중견기업"):
            self.assertTrue(is_domestic_common_row(common(market="10", companyClassName=issuer_class, kind="A")))
        http = MasterHTTP({"0": [common(), common("069500", "KODEX 200", "8", marketName="ETF", kind="A"),
                                       common("432320", "KB스타리츠", "6", marketName="리츠", kind="A")]})
        self.assertEqual(domestic_common_symbols(http), frozenset({"005930", "035900"}))

    def test_long_non_equity_identities_are_never_truncated_into_stocks(self):
        http = MasterHTTP({"5": [common("0036221D", "KG모빌리티 122WR", "5")],
                           "9": [common("7010003", "공모펀드", "9")],
                           "80": [common("M04020000", "금 99.99_1kg", "80")]})
        self.assertEqual(domestic_common_symbols(http), frozenset({"005930", "035900"}))
        with self.assertRaises(BrokerAPIError):
            domestic_common_symbols(MasterHTTP({"8": [{"code": None}]}))
        with self.assertRaises(BrokerAPIError):
            domestic_common_symbols(MasterHTTP({"8": [{"code": "bad-code!"}]}))

    def test_only_confirmed_successful_terminal_null_product_list_is_empty(self):
        class NullMaster(MasterHTTP):
            body = {"list": None, "return_code": 0}
            def iter_pages(self, **kwargs):
                if kwargs["body"]["mrkt_tp"] == "7":
                    yield APIPage(self.body, False, None, None)
                else:
                    yield from super().iter_pages(**kwargs)
        self.assertIn("005930", domestic_common_symbols(NullMaster()))
        for body in ({"list": None}, {"return_code": 0}, {"list": None, "return_code": 1}):
            http = NullMaster()
            http.body = body
            with self.assertRaises(BrokerAPIError):
                domestic_common_symbols(http)


if __name__ == "__main__":
    unittest.main()
