"""Offline regression tests for the broker/public common-share intersection."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from dockdack.equity_policy import _US_CACHE, _us_broker_common_candidates, common_equities
from dockdack.exceptions import BrokerAPIError
from dockdack.http import APIPage
from dockdack.models import Market


def row(symbol="AAPL", exchange="ND", flag="N", **changes):
    result = dict(stk_cd=symbol, stex_tp=exchange, isEtf=flag)
    result.update(changes)
    return result


class MasterHTTP:
    def __init__(self, stocks=None, etfs=None):
        self.groups = {
            "usa10099": [APIPage({"list": stocks if stocks is not None else [row()]}, False, None, None)],
            "usa10104": [APIPage({"list": etfs if etfs is not None else [row("SPY", "NA", "Y")]}, False, None, None)],
        }
        self.fail = None
        self.calls = []

    def iter_pages(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail == kwargs["api_id"]:
            raise BrokerAPIError("classification source unavailable")
        yield from self.groups[kwargs["api_id"]]


class EquityPolicyTests(unittest.TestCase):
    def setUp(self):
        _US_CACHE.clear()
        self.addCleanup(_US_CACHE.clear)
        self.public = patch("dockdack.us_equity_universe.us_common_symbols",
                            side_effect=lambda http, candidates: frozenset(candidates)).start()
        self.addCleanup(patch.stopall)

    def test_us_requires_explicit_non_etf_flag_before_public_classification(self):
        for flag in (None, "", "unknown", "Y", "y", False, 0, "0"):
            with self.subTest(flag=flag):
                http = MasterHTTP([row(), row("MYSTERY", flag=flag)])
                result = common_equities(http, Market.US, (("AAPL", "ND"), ("MYSTERY", "ND")))
                self.assertEqual(result, frozenset({("AAPL", "ND")}))
                self.assertEqual(self.public.call_args.args[1], (("AAPL", "ND"),))
        missing = row("MISSING")
        del missing["isEtf"]
        self.assertNotIn(("MISSING", "ND"), _us_broker_common_candidates(MasterHTTP([row(), missing])))

    def test_etf_membership_and_conflicting_duplicate_flags_override_stocks(self):
        http = MasterHTTP([row(), row("QQQ"), row("CONFLICT"), row("CONFLICT", flag="Y")],
                          [row("QQQ", flag="N")])
        # Even an erroneously non-ETF flag in the ETF-specific master is a veto.
        self.assertEqual(_us_broker_common_candidates(http), frozenset({("AAPL", "ND")}))

    def test_broker_and_public_sources_must_both_allow_exact_symbol_and_venue(self):
        http = MasterHTTP([row(), row("BRKb", "NY"), row("PUBLICDENY", "NA")])
        self.public.side_effect = lambda http, candidates: frozenset(k for k in candidates if k[0] != "PUBLICDENY")
        candidates = (("BRKb", "NY"), ("BRKB", "NY"), ("BRKb", "ND"),
                      ("AAPL", "ND"), ("PUBLICDENY", "NA"), ("NOTINBROKER", "ND"))
        self.assertEqual(common_equities(http, Market.US, candidates), frozenset({("BRKb", "NY"), ("AAPL", "ND")}))
        self.assertEqual(self.public.call_args.args[1], (("BRKb", "NY"), ("AAPL", "ND"), ("PUBLICDENY", "NA")))
        self.assertTrue(all(c["body"] == {"stex_tp": "%"} for c in http.calls))

    def test_empty_broker_candidate_intersection_does_not_call_public(self):
        self.assertEqual(common_equities(MasterHTTP(), Market.US, (("MISSING", "NY"),)), frozenset())
        self.public.assert_not_called()

    def test_complete_master_cached_once_per_client_and_new_york_date(self):
        http = MasterHTTP()
        with patch("dockdack.equity_policy.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
            first = _us_broker_common_candidates(http)
            self.assertIs(_us_broker_common_candidates(http), first)
            self.assertEqual(len(http.calls), 2)
            clock.now.return_value = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
            _us_broker_common_candidates(http)
            self.assertEqual(len(http.calls), 4)
        other = MasterHTTP([row("MSFT")])
        self.assertEqual(_us_broker_common_candidates(other), frozenset({("MSFT", "ND")}))

    def test_failure_during_second_source_never_caches_partial_allow_list(self):
        http = MasterHTTP()
        http.fail = "usa10104"
        for _ in range(2):
            with self.assertRaises(BrokerAPIError):
                common_equities(http, Market.US, (("AAPL", "ND"),))
            self.assertNotIn(http, _US_CACHE)
        self.public.assert_not_called()
        self.assertEqual(len(http.calls), 4)
        http.fail = None
        self.assertIn(("AAPL", "ND"), common_equities(http, Market.US, (("AAPL", "ND"),)))

    def test_new_day_failure_never_falls_back_to_yesterdays_broker_master(self):
        http = MasterHTTP()
        with patch("dockdack.equity_policy.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)
            _us_broker_common_candidates(http)
            clock.now.return_value = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
            http.fail = "usa10104"
            with self.assertRaises(BrokerAPIError):
                _us_broker_common_candidates(http)
            self.assertNotIn(http, _US_CACHE)

    def test_unfinished_empty_and_malformed_sources_are_not_cached(self):
        bad_pages = (
            APIPage({"list": [row()]}, True, "more", "Y"),
            APIPage({"list": []}, False, None, None),
            APIPage({"list": None}, False, None, None),
            APIPage({}, False, None, None),
            APIPage({"list": ["bad"]}, False, None, None),
            APIPage({"list": [row(stk_cd="")]}, False, None, None),
            APIPage({"list": [row(code="DIFFERENT")]}, False, None, None),
            APIPage({"list": [row(exchange="UNKNOWN")]}, False, None, None),
        )
        for page in bad_pages:
            with self.subTest(page=page):
                http = MasterHTTP()
                http.groups["usa10099"] = [page]
                with self.assertRaises(BrokerAPIError):
                    _us_broker_common_candidates(http)
                self.assertNotIn(http, _US_CACHE)

    def test_domestic_adapter_preserves_codes_and_requires_krx(self):
        http = object()
        with patch("dockdack.equity_policy.domestic_common_symbols", return_value=frozenset({"005930", "0068Y0"})) as master:
            result = common_equities(http, Market.DOMESTIC,
                                     iter((("005930", "KRX"), ("005930", "NY"), ("0068Y0", "KRX"), ("069500", "KRX"))))
        self.assertEqual(result, frozenset({("005930", "KRX"), ("0068Y0", "KRX")}))
        master.assert_called_once_with(http)
        self.public.assert_not_called()

    def test_public_or_domestic_classification_failure_propagates(self):
        self.public.side_effect = BrokerAPIError("public metadata unavailable")
        with self.assertRaises(BrokerAPIError):
            common_equities(MasterHTTP(), Market.US, (("AAPL", "ND"),))
        with patch("dockdack.equity_policy.domestic_common_symbols", side_effect=BrokerAPIError("domestic unavailable")):
            with self.assertRaises(BrokerAPIError):
                common_equities(object(), Market.DOMESTIC, (("005930", "KRX"),))


if __name__ == "__main__":
    unittest.main()
