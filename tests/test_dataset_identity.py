"""Offline catalog decisions never query a broker or normalize a bar identity."""

import copy
import json
import unittest
from unittest.mock import patch

from dockdack.dataset_identity import classify_instrument, resolve_catalog_key


def domestic(symbol="005930", name="삼성전자", category="0"):
    market_name = {"0": "거래소", "10": "코스닥", "8": "ETF", "6": "리츠"}.get(category, "기타")
    raw = {"code": symbol, "name": name, "marketCode": category, "marketName": market_name,
           "upName": "전기/전자", "companyClassName": "", "kind": "A"}
    return {"symbol": symbol, "exchange": "KRX", "name": name, "english_name": None,
            "listing_market": market_name, "catalog_market_code": category, "is_etf": None,
            "raw_json": json.dumps(raw, ensure_ascii=False)}


def us(symbol="AAPL", name="애플", english="APPLE INC", exchange="ND", is_etf=0):
    raw = {"stk_cd": symbol, "stex_tp": exchange, "stk_nm": name, "stk_enm": english,
           "mkgb": {"ND": "NASDAQ", "NY": "NYSE", "NA": "AMEX"}.get(exchange, "UNKNOWN"),
           "upgb": "컴퓨터 및 전자장비", "isEtf": "Y" if is_etf == 1 else "N"}
    return {"symbol": symbol, "exchange": exchange, "name": name, "english_name": english,
            "listing_market": exchange, "catalog_market_code": exchange, "is_etf": is_etf,
            "raw_json": json.dumps(raw, ensure_ascii=False)}


def raw_change(row, **changes):
    result = copy.deepcopy(row)
    raw = json.loads(result["raw_json"])
    raw.update(changes)
    result["raw_json"] = json.dumps(raw, ensure_ascii=False)
    return result


class DatasetClassificationTests(unittest.TestCase):
    def setUp(self):
        forbidden = patch("requests.sessions.Session.request", side_effect=AssertionError("network forbidden"))
        forbidden.start()
        self.addCleanup(forbidden.stop)

    def test_domestic_common_candidate_requires_supporting_metadata_and_preserves_input(self):
        for row in (domestic(), domestic("123410", "코리아에프티", "10"), domestic("0070X0", "회사", "10")):
            original = copy.deepcopy(row)
            status, reasons = classify_instrument(row, "domestic")
            self.assertEqual(status, "eligible")
            self.assertIn("DOMESTIC_COMPANY_CANDIDATE", reasons)
            self.assertIn("CURRENT_CATALOG_NOT_POINT_IN_TIME", reasons)
            self.assertEqual(row, original)

    def test_domestic_product_categories_are_explicit_exclusions_even_with_ordinary_issuer_name(self):
        for category in ("2", "3", "4", "5", "7", "8", "9", "60", "70", "80", "90"):
            with self.subTest(category=category):
                status, reasons = classify_instrument(domestic(category=category), "domestic")
                self.assertEqual(status, "excluded")
                self.assertTrue(reasons[0].startswith("CATALOG_CATEGORY_"))

    def test_domestic_unknown_otc_konex_and_reit_category_are_review_not_invented_products(self):
        for category in ("15", "30", "50", "6", "NEW"):
            status, reasons = classify_instrument(domestic(category=category), "domestic")
            self.assertEqual((status, reasons), ("review", ("DOMESTIC_CATEGORY_OUTSIDE_CANDIDATE_SCOPE",)))

    def test_domestic_preferred_and_spac_exclusions_are_labelled_heuristics(self):
        for row in (domestic("005935", "삼성전자우"), domestic("123450", "테스트스팩1호")):
            status, reasons = classify_instrument(row, "domestic")
            self.assertEqual(status, "excluded")
            self.assertIn("NAME_SUBTYPE_NOT_MASTER_VERIFIED", reasons)

    def test_domestic_noncommon_code_and_missing_industry_or_class_go_to_review(self):
        for row in (domestic("005935", "종목명"), raw_change(domestic(), upName=""),
                    raw_change(domestic(), companyClassName="unverified"), raw_change(domestic(), kind="J"),
                    raw_change(domestic(), marketName=[]), raw_change(domestic(), companyClassName=[])):
            self.assertEqual(classify_instrument(row, "domestic")[0], "review")

    def test_us_bare_issuer_is_annotated_candidate_not_verified_common_share(self):
        status, reasons = classify_instrument(us(), "us")
        self.assertEqual(status, "eligible")
        self.assertIn("NON_ETF_EQUITY_CANDIDATE", reasons)
        self.assertIn("SHARE_SUBTYPE_NOT_VERIFIED", reasons)
        self.assertIn("CURRENT_CATALOG_NOT_POINT_IN_TIME", reasons)

    def test_us_explicit_etf_and_conflicting_or_unknown_flags(self):
        self.assertEqual(classify_instrument(us(is_etf=1), "us"), ("excluded", ("EXPLICIT_ETF_FLAG",)))
        for row in (us(is_etf=None), us(is_etf="unknown"), raw_change(us(), isEtf="?"), raw_change(us(), isEtf="Y")):
            self.assertEqual(classify_instrument(row, "us")[0], "review")

    def test_us_adr_and_reit_are_retained_with_flags(self):
        adr = us("BABA", "알리바바(ADR)", "ALIBABA GROUP HOLDING LTD SPON ADS EACH", "NY")
        reit = raw_change(us("NLY", "애널리", "ANNALY CAPITAL MANAGEMENT INC", "NY"), upgb="REIT 및 부동산관리개발")
        for row, flag in ((adr, "ADR_CANDIDATE_RETAINED"), (reit, "REIT_CANDIDATE_RETAINED")):
            status, reasons = classify_instrument(row, "us")
            self.assertEqual(status, "eligible")
            self.assertIn(flag, reasons)

    def test_us_missing_industry_is_visible_but_not_guessed_to_be_non_equity(self):
        status, reasons = classify_instrument(raw_change(us(), upgb=""), "us")
        self.assertEqual(status, "eligible")
        self.assertIn("INDUSTRY_UNAVAILABLE", reasons)

    def test_product_names_exclude_only_with_heuristic_annotation(self):
        cases = (("SOUNDHOUND AI INC C/WTS 26/04/2027", "WARRANT"),
                 ("ISSUER 5.1% PREF", "PREFERRED"), ("ISSUER UNITS", "UNIT"),
                 ("ISSUER UNIT 1 COM &", "UNIT"), ("ADAMS DIVERSIFIED EQUITY FUND INC", "FUND"),
                 ("ARES ACQUISITION CORPORATION III", "SPAC"), ("ISSUER RIGHTS", "RIGHT"),
                 ("ISSUER ETN", "ETN"), ("ISSUER 5.1% SUB NOTES", "DEBT"))
        for english, subtype in cases:
            with self.subTest(english=english):
                status, reasons = classify_instrument(us(english=english), "us")
                self.assertEqual(status, "excluded")
                self.assertIn("HEURISTIC_NAME_" + subtype, reasons)
                self.assertIn("NAME_SUBTYPE_NOT_MASTER_VERIFIED", reasons)

    def test_issuer_words_do_not_become_unit_fund_or_trust_securities(self):
        for english in ("UNIT CORP", "UNITED AIRLINES INC", "NORTHERN TRUST CORP", "FUNDING COMPANY INC"):
            self.assertEqual(classify_instrument(us(english=english), "us")[0], "eligible")

    def test_exact_currency_market_and_case_are_validated_not_normalized(self):
        for row in ({**us(), "currency": "KRW"}, {**us(), "market": "domestic"},
                    {**us(), "exchange": "NASDAQ"}, {**us(), "exchange": []},
                    raw_change(us(), stk_cd="aapl"), raw_change(us(), stex_tp="NY"),
                    raw_change(us(), currency="KRW")):
            self.assertEqual(classify_instrument(row, "us")[0], "review")
        self.assertEqual(classify_instrument({**us(), "currency": "USD", "market": "us"}, "us")[0], "eligible")
        self.assertEqual(classify_instrument(us("BRKb", "버크셔 B", "BERKSHIRE HATHAWAY INC", "NY"), "us")[0], "eligible")

    def test_raw_or_name_conflicts_and_malformed_json_are_quarantined(self):
        for raw in (None, "[]", "null", "bad", '{"code":"a","code":"b"}', '{"price":NaN}'):
            self.assertEqual(classify_instrument({**us(), "raw_json": raw}, "us"), ("review", ("RAW_CATALOG_INVALID",)))
        for row in (raw_change(us(), stk_nm="different"), raw_change(us(), stk_enm="different"),
                    raw_change(us(), mkgb="NYSE"), {**us(), "name": ""}):
            self.assertEqual(classify_instrument(row, "us")[0], "review")
        self.assertEqual(classify_instrument({}, "other"), ("review", ("UNSUPPORTED_MARKET",)))
        self.assertEqual(classify_instrument(None, "us"), ("review", ("CATALOG_ROW_INVALID",)))


class CatalogKeyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {("NY", "BRKb"): us("BRKb", "버크셔 B", "BERKSHIRE HATHAWAY INC", "NY"),
                        ("NY", "BRKa"): us("BRKa", "버크셔 A", "BERKSHIRE HATHAWAY INC", "NY"),
                        ("ND", "AAPL"): us()}

    def test_exact_key_preserves_share_class_case(self):
        self.assertEqual(resolve_catalog_key("BRKb", "NY", self.catalog),
                         ("exact", (("NY", "BRKb"),), ("EXACT_CATALOG_KEY",)))

    def test_uppercase_only_orphan_returns_review_candidate_never_renames(self):
        original = copy.deepcopy(self.catalog)
        self.assertEqual(resolve_catalog_key("BRKB", "NY", self.catalog),
                         ("review", (("NY", "BRKb"),), ("CASE_ONLY_CANDIDATE_REQUIRES_VERIFICATION",)))
        self.assertEqual(self.catalog, original)

    def test_multiple_case_candidates_stay_review_and_exact_key_wins(self):
        self.catalog[("NY", "brkb")] = us("brkb", "another", "OTHER INC", "NY")
        self.assertEqual(resolve_catalog_key("BRKB", "NY", self.catalog)[0:2],
                         ("review", (("NY", "BRKb"), ("NY", "brkb"))))
        self.assertEqual(resolve_catalog_key("BRKb", "NY", self.catalog)[0], "exact")

    def test_exchange_mismatch_is_review_not_automatic_cross_exchange_join(self):
        self.assertEqual(resolve_catalog_key("AAPL", "NY", self.catalog),
                         ("review", (("ND", "AAPL"),), ("OTHER_EXCHANGE_CANDIDATE_REQUIRES_VERIFICATION",)))
        self.assertEqual(resolve_catalog_key("MISSING", "ND", self.catalog)[0], "missing")
        self.assertEqual(resolve_catalog_key("AAPL", "NP", self.catalog)[0], "review")

    def test_no_punctuation_or_whitespace_conversion_and_mapping_identity_must_match(self):
        self.assertEqual(resolve_catalog_key("BRK.B", "NY", self.catalog)[0], "missing")
        self.assertEqual(resolve_catalog_key(" BRKb", "NY", self.catalog)[0], "review")
        self.catalog[("NY", "BRKb")]["symbol"] = "BRKB"
        self.assertEqual(resolve_catalog_key("BRKb", "NY", self.catalog)[0], "review")


if __name__ == "__main__":
    unittest.main()
