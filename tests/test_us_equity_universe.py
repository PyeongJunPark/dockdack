import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from dockdack import us_equity_universe as us
from dockdack.exceptions import BrokerAPIError


def master_zip(exchange, rows):
    lines = []
    for symbol, name, security_type, dr, industry, etp in rows:
        columns = [""] * 24
        columns[2], columns[4], columns[7], columns[8] = exchange.upper(), symbol, name, security_type
        columns[17], columns[19], columns[22] = dr, industry, etp
        lines.append("\t".join(columns))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(exchange + "mst.cod", "\n".join(lines).encode("cp949"))
    return output.getvalue()


def directory_bytes(rows):
    header = "Symbol|Security Name|Listing Exchange|ETF|Test Issue|NextShares"
    return ("\n".join([header] + ["|".join(row) for row in rows]
                       + ["File Creation Time: 0915202606:03|||||"]) + "\n").encode()


def screener_bytes(rows):
    return json.dumps({"data": {"rows": rows}, "status": {"rCode": 200}}).encode()


class USEquityUniverseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "classification.json"
        for name, value in (("CACHE_PATH", self.cache), ("_MEMORY", None), ("_FAILURE", None)):
            patcher = patch.object(us, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.kis = {"name": "APPLE INC", "type": "2", "dr": "N", "industry": "730", "etp": ""}
        self.listed = {"name": "Apple Inc. Common Stock", "etf": "N", "test": "N", "next": "N"}
        self.screen = {"name": "Apple Inc. Common Stock", "industry": "Computer Manufacturing", "sector": "Technology"}

    def test_positive_common_and_ordinary_labels_required(self):
        self.assertTrue(us._eligible(self.kis, self.listed, self.screen))
        for label in ("Example Class A Ordinary Shares", "Example Class C Capital Stock", "Example Common Shares"):
            self.assertTrue(us._eligible(self.kis, dict(self.listed, name=label), dict(self.screen, name=label)))
        self.assertFalse(us._eligible(self.kis, dict(self.listed, name="Apple Inc."), self.screen))
        self.assertFalse(us._eligible(self.kis, self.listed, dict(self.screen, name="Apple Inc.")))

    def test_security_master_rejects_etp_dr_and_unknown_industry(self):
        for changed in ({"type": "3"}, {"type": "4"}, {"type": ""}, {"dr": "Y"}, {"dr": ""},
                        {"industry": "000"}, {"industry": ""}, {"industry": "UNKNOWN"}, {"etp": "004"}):
            with self.subTest(changed=changed):
                self.assertFalse(us._eligible(dict(self.kis, **changed), self.listed, self.screen))

    def test_nasdaq_excludes_funds_reits_spacs_preferred_and_test_products(self):
        names = ["PIMCO Dynamic Income Fund Common Stock", "Example Acquisition Corp Common Stock",
                 "Example Real Estate Investment Trust Common Stock", "Example Preferred Stock",
                 "Example Units each representing Common Stock", "Example Warrants for Common Stock",
                 "Example Rights for Common Shares", "Example American Depositary Common Shares",
                 "Example Notes Common Stock", "Example REIT Common Stock", "Example SPAC Common Stock"]
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(us._eligible(self.kis, dict(self.listed, name=name), self.screen))
                self.assertFalse(us._eligible(dict(self.kis, name=name), self.listed, self.screen))
        for flag in ("etf", "test", "next"):
            for value in ("Y", "", "0"):
                self.assertFalse(us._eligible(self.kis, dict(self.listed, **{flag: value}), self.screen))

    def test_industry_vehicles_and_missing_metadata_are_rejected(self):
        for industry in ("Blank Checks", "Real Estate Investment Trusts", "Closed-End Funds",
                         "Trusts Except Educational Religious and Charitable", "Finance: Consumer Services", ""):
            self.assertFalse(us._eligible(self.kis, self.listed, dict(self.screen, industry=industry)))
        self.assertFalse(us._eligible(self.kis, self.listed, dict(self.screen, sector="")))
        self.assertFalse(us._eligible({}, self.listed, self.screen))
        self.assertFalse(us._eligible(self.kis, {}, self.screen))
        self.assertFalse(us._eligible(self.kis, self.listed, {}))

    def test_explicit_case_sensitive_class_alias_only(self):
        self.assertEqual(us._public_symbol("BRKb"), "BRK.B")
        self.assertEqual(us._public_symbol("BRK/B"), "BRK.B")
        self.assertEqual(us._public_symbol("BRK.B"), "BRK.B")
        self.assertEqual(us._public_symbol("BRKB"), "BRKB")
        eligible = frozenset({("BRK.B", "NY"), ("AAPL", "ND")})
        with patch.object(us, "_universe", return_value=eligible):
            self.assertEqual(us.us_common_symbols(object(), (("BRKb", "NY"), ("BRKB", "NY"),
                                                             ("AAPL", "NY"), ("AAPL", "ND"))),
                             frozenset({("BRKb", "NY"), ("AAPL", "ND")}))

    def test_full_public_join_filters_known_counterexamples_without_broker_calls(self):
        public = {
            us.KIS_URL.format(exchange="nas"): master_zip("nas", [
                ("AAPL", "APPLE INC", "2", "N", "730", ""),
                ("ARCC", "ARES CAPITAL CORP", "2", "N", "640", "")]),
            us.KIS_URL.format(exchange="nys"): master_zip("nys", [
                ("BRK/B", "BERKSHIRE HATHAWAY", "2", "N", "640", ""),
                ("AAC", "ARES ACQUISITION CORP III", "2", "N", "000", ""),
                ("O", "REALTY INCOME", "2", "N", "630", ""),
                ("PDI", "PIMCO DYNAMIC INCOME FD", "3", "N", "000", "004")]),
            us.KIS_URL.format(exchange="ams"): master_zip("ams", [
                ("SPY", "SPDR", "3", "N", "000", "001")]),
            us.DIRECTORY_URL: directory_bytes([
                ("AAPL", "Apple Common Stock", "Q", "N", "N", "N"),
                ("ARCC", "Ares Capital Common Stock", "Q", "N", "N", "N"),
                ("BRK.B", "Berkshire Class B Common Stock", "N", "N", "N", "N"),
                ("AAC", "Ares Acquisition Class A Ordinary Shares", "N", "N", "N", "N"),
                ("O", "Realty Income Common Stock", "N", "N", "N", "N"),
                ("PDI", "PIMCO Dynamic Income Fund Common Stock", "N", "N", "N", "N")]),
            us.SCREENER_URL: screener_bytes([
                dict(symbol="AAPL", name="Apple Common Stock", industry="Computer Manufacturing", sector="Technology"),
                dict(symbol="ARCC", name="Ares Capital Common Stock", industry="Finance: Consumer Services", sector="Finance"),
                dict(symbol="BRK.B", name="Berkshire Class B Common Stock", industry="Property-Casualty Insurers", sector="Finance"),
                dict(symbol="AAC", name="Ares Acquisition Ordinary Shares", industry="Metal Fabrications", sector="Industrials"),
                dict(symbol="O", name="Realty Income Common Stock", industry="Real Estate Investment Trusts", sector="Real Estate"),
                dict(symbol="PDI", name="PIMCO Dynamic Income Fund Common Stock", industry="Trusts", sector="Finance")]),
        }
        with patch.object(us, "_download", side_effect=lambda url: public[url]) as download:
            self.assertEqual(us._fetch_universe(), frozenset({("AAPL", "ND"), ("BRK.B", "NY")}))
            self.assertEqual(download.call_count, 5)

    def test_cache_is_shared_daily_and_survives_process_restart(self):
        eligible = frozenset({("AAPL", "ND")})
        with patch.object(us, "_today", return_value="2026-09-15"), patch.object(us, "_fetch_universe", return_value=eligible) as fetch:
            self.assertTrue(us.is_us_common(None, "AAPL", "ND"))
            self.assertFalse(us.is_us_common(None, "SPY", "NA"))
            us._MEMORY = None
            self.assertTrue(us.is_us_common(None, "AAPL", "ND"))
            self.assertEqual(fetch.call_count, 1)
        with patch.object(us, "_today", return_value="2026-09-16"), patch.object(us, "_fetch_universe", return_value=eligible) as fetch:
            self.assertTrue(us.is_us_common(None, "AAPL", "ND"))
            self.assertEqual(fetch.call_count, 1)

    def test_provider_failure_never_uses_yesterdays_cache_and_backs_off(self):
        us._write_cache(self.cache, "2026-09-14", frozenset({("AAPL", "ND")}))
        with patch.object(us, "_today", return_value="2026-09-15"), patch.object(us, "_fetch_universe", side_effect=ValueError("offline")) as fetch:
            for _ in range(2):
                with self.assertRaises(BrokerAPIError):
                    us.is_us_common(None, "AAPL", "ND")
            self.assertEqual(fetch.call_count, 1)
        self.assertIsNone(us._MEMORY)

    def test_cache_corruption_and_duplicate_symbols_do_not_grant_eligibility(self):
        document = {"version": us._VERSION, "day": "2026-09-15", "eligible": [["AAPL", "ND"], ["AAPL", "ND"]]}
        self.cache.write_text(json.dumps(document), encoding="utf-8")
        self.assertIsNone(us._read_cache(self.cache, "2026-09-15"))
        self.cache.write_text("{bad", encoding="utf-8")
        self.assertIsNone(us._read_cache(self.cache, "2026-09-15"))

    def test_malformed_or_truncated_public_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            us._directory_rows(b"Symbol|Security Name|Listing Exchange|ETF|Test Issue|NextShares\nAAPL|Apple Common Stock|Q|N|N|N")
        with self.assertRaises(ValueError):
            us._kis_rows(master_zip("nys", [("AAPL", "APPLE", "2", "N", "730", "")]), "nas")
        for data in ({"status": {"rCode": 500}}, {"status": {"rCode": 200}, "data": {"rows": []}}):
            with self.assertRaises(ValueError):
                us._screener_rows(json.dumps(data).encode())
        duplicate = [dict(symbol="AAPL", name="Apple Common Stock", industry="Tech", sector="Tech"),
                     dict(symbol="AAPL", name="Wrong Common Stock", industry="Tech", sector="Tech")]
        with self.assertRaises(ValueError):
            us._screener_rows(screener_bytes(duplicate))

    def test_nested_screener_shape_is_supported(self):
        row = dict(symbol="AAPL", name="Apple Common Stock", industry="Tech", sector="Tech")
        document = {"status": {"rCode": 200}, "data": {"table": {"rows": [row]}}}
        self.assertEqual(us._screener_rows(json.dumps(document).encode())["AAPL"]["industry"], "Tech")

    def test_empty_candidates_do_not_fetch(self):
        with patch.object(us, "_universe") as fetch:
            self.assertEqual(us.us_common_symbols(None, ()), frozenset())
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
