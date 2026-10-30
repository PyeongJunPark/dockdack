"""The offline CLI verifies receipts and never touches a broker or account."""

import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dockdack.minute_hedge_research import PaperCosts
from dockdack.models import Market
from examples.run_minute_hedge_research import run


KST = ZoneInfo("Asia/Seoul")
COSTS = PaperCosts(1, 2, 500, 252 * 390)
DAYS = (date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3),
        date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8))


def make_rows(symbol, exchange, *, days=DAYS, count=40, shift=0, flat=False):
    rows = []
    for day in days:
        start = datetime(day.year, day.month, day.day, 9, 10, tzinfo=KST)
        for i in range(count):
            if exchange == "INDEX":
                close = 25000 + (i % 3) * 20
            elif symbol == "114800":
                close = 1000 - (i % 3)
            elif flat:
                close = 100
            else:
                close = (100 + i % 3 if i < 15 else 94 if i <= 18 else
                         97 if i == 19 else 99)
            rows.append({"market": "domestic", "exchange": exchange, "symbol": symbol,
                         "timestamp": (start + timedelta(minutes=(i + shift) * 5)).isoformat(),
                         "bar_minutes": 5, "open": str(close), "high": str(close + 1),
                         "low": str(close - 1), "close": str(close), "volume": "10000"})
    return rows


def write_collected(root, name, rows, *, index_code=None, coverage=True,
                    truncated=False, collected_at="2026-09-09T00:00:00+00:00"):
    path = root / f"{name}.jsonl"
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    path.write_bytes(payload)
    symbols = sorted({row["symbol"] for row in rows if row["exchange"] != "INDEX"})
    receipt = {
        "source": "kiwoom_rest_demo_minute_chart", "market": "domestic",
        "exchange": rows[0]["exchange"], "symbols": symbols, "index_code": index_code,
        "index_price_basis": ("ka20005_signed_abs_divided_by_100_points; demo_sample_verified_2026-09-29"
                              if index_code else None),
        "interval_minutes": 5, "collected_at_utc": collected_at,
        "regular_session_filter": True, "chart_truncated": truncated if coverage else None,
        "chart_pagination_metadata_known": coverage,
        "accepted_bars": len(rows), "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.with_suffix(".jsonl.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


class MinuteHedgeCLIResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stock = write_collected(self.root, "stock", make_rows("005930", "KRX"))
        self.index = write_collected(self.root, "index", make_rows("201", "INDEX"), index_code="201")
        self.etf = write_collected(self.root, "etf", make_rows("114800", "KRX"))
        self.common = dict(stock_file=self.stock, index_file=self.index,
                           stock_symbol="005930", index_code="201", market=Market.DOMESTIC,
                           costs=COSTS, notional_per_leg=1_000_000,
                           min_prior_volume=1_000, min_train_sessions=3,
                           min_train_trades=2)

    def test_verified_index_reference_report_and_explicit_abstentions(self):
        output = self.root / "report.json"
        report = run(**self.common, output=output)
        saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(saved["schema"], "dockdack.minute_hedge_research.v1")
        self.assertEqual(report["hedge_leg"], "synthetic_index_short")
        self.assertIn("divided by 100", report["benchmark_unit_note"])
        self.assertEqual(report["execution_delay_bars"], 3)
        self.assertIn("completed stock bar close", report["policy_controls"]["exit_signal_basis"])
        self.assertTrue(any("fractional-share" in item for item in report["limitations"]))
        self.assertTrue(any("not official ETF tracking error" in item for item in report["limitations"]))
        self.assertEqual(report["used_contiguous_sessions"], len(DAYS))
        self.assertEqual(len(report["candidates"]), 7)
        reasons = {candidate["variant"]: candidate["reason"] for candidate in report["candidates"]}
        self.assertIn("sector", reasons["sector_relative"])
        self.assertIn("turnover", reasons["turnover_vwap"])
        self.assertIn("previous", reasons["gap_relative"])
        self.assertNotIn("etf", report["inputs"])
        with self.assertRaises(FileExistsError):
            run(**self.common, output=output)

    def test_etf_requires_own_file_and_explicit_minus_one_confirmation(self):
        output = self.root / "etf-report.json"
        with self.assertRaisesRegex(ValueError, "-1x"):
            run(**self.common, output=output, inverse_etf_file=self.etf,
                inverse_etf_symbol="114800")
        self.assertFalse(output.exists())
        report = run(**self.common, output=output, inverse_etf_file=self.etf,
                     inverse_etf_symbol="114800", confirm_inverse_etf_minus_one=True)
        self.assertEqual(report["hedge_leg"], "long_inverse_etf")
        self.assertIn("etf", report["inputs"])
        self.assertEqual(report["inverse_etf_symbol"], "114800")

    def test_sha_tamper_rejected_before_report(self):
        self.stock.write_bytes(self.stock.read_bytes() + b"\n")
        output = self.root / "bad-report.json"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            run(**self.common, output=output)
        self.assertFalse(output.exists())

    def test_us_index_hedge_is_not_claimed(self):
        with self.assertRaisesRegex(ValueError, "domestic ka20005"):
            run(**{**self.common, "market": Market.US}, output=self.root / "us-report.json")

    def test_unverified_index_price_basis_is_rejected(self):
        receipt_path = self.index.with_suffix(".jsonl.receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["index_price_basis"] = "INDEX_X100"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index-point price basis"):
            run(**self.common, output=self.root / "wrong-basis-report.json")
        self.assertFalse((self.root / "wrong-basis-report.json").exists())

    def test_unverified_pagination_or_short_data_produces_abstain_report(self):
        stock = write_collected(self.root, "uncertain", make_rows("005930", "KRX"), coverage=False)
        report = run(**{**self.common, "stock_file": stock}, output=self.root / "uncertain-report.json")
        self.assertTrue(all(candidate["data_gate"] == "abstain" for candidate in report["candidates"]))
        self.assertTrue(report["data_quality_reasons"])
        short = write_collected(self.root, "short", make_rows("005930", "KRX", count=15))
        report = run(**{**self.common, "stock_file": short}, output=self.root / "short-report.json")
        self.assertEqual(report["used_contiguous_sessions"], 0)
        self.assertTrue(all(row["reason"] == "contiguous_run_too_short"
                            for row in report["session_audit"]))
        self.assertTrue(all(candidate["data_gate"] == "abstain" for candidate in report["candidates"]))

    def test_bounded_recent_slice_drops_only_oldest_page_edge_day(self):
        stock = write_collected(self.root, "bounded", make_rows("005930", "KRX"), truncated=True)
        report = run(**{**self.common, "stock_file": stock}, output=self.root / "bounded-report.json")
        self.assertEqual(report["used_contiguous_sessions"], len(DAYS) - 1)
        self.assertEqual(report["session_audit"][0]["reason"], "oldest_page_boundary_may_be_partial")
        self.assertFalse(report["data_quality_reasons"])
        self.assertTrue(report["data_quality_warnings"])

    def test_during_session_collection_drops_latest_unfinished_day(self):
        stock = write_collected(self.root, "intraday", make_rows("005930", "KRX"),
                                collected_at="2026-09-08T05:00:00+00:00")
        report = run(**{**self.common, "stock_file": stock}, output=self.root / "intraday-report.json")
        self.assertEqual(report["used_contiguous_sessions"], len(DAYS) - 1)
        self.assertEqual(report["session_audit"][-1]["reason"],
                         "collection_during_unfinished_market_session")

    def test_timestamp_gap_not_filled_and_wrong_index_identity_fails(self):
        rows = make_rows("005930", "KRX")
        rows = [row for row in rows if not row["timestamp"].endswith("10:25:00+09:00")]
        gap = write_collected(self.root, "gap", rows)
        report = run(**{**self.common, "stock_file": gap}, output=self.root / "gap-report.json")
        self.assertGreater(report["session_audit"][0]["source_gap_count"], 0)
        self.assertGreater(report["session_audit"][0]["discarded_aligned_bars"], 0)
        with self.assertRaisesRegex(ValueError, "index code"):
            run(**{**self.common, "index_code": "101"}, output=self.root / "wrong-index.json")


if __name__ == "__main__":
    unittest.main()
