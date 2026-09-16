"""Synthetic temporary SQLite integration tests; never open real datasets."""
from contextlib import closing, redirect_stdout
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from dockdack.clean_daily_dataset import BAR_COLUMNS, CATALOG_COLUMNS, build_database
from dockdack.dataset_quality import QualityPolicy


def sessions(count=220):
    result, day = [], date(2024, 1, 1)
    while len(result) < count:
        if day.weekday() < 5:
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def catalog_item(symbol="005930", *, market="domestic", etf=False):
    if market == "domestic":
        name, exchange = "테스트기업", "KRX"
        category, listing = ("8", "ETF") if etf else ("0", "거래소")
        raw = dict(code=symbol, name=name, marketCode=category, marketName=listing,
                   upName="전기/전자", companyClassName="", kind="A")
        english, is_etf = None, None
    else:
        name, exchange, category, listing = "테스트기업", "ND", "ND", "ND"
        english, is_etf = "EXAMPLE TECHNOLOGIES INC", int(etf)
        raw = dict(stk_cd=symbol, stk_nm=name, stk_enm=english, stex_tp=exchange,
                   mkgb="NASDAQ", upgb="소프트웨어", isEtf="Y" if etf else "N")
    return dict(symbol=symbol, exchange=exchange, name=name, english_name=english,
                listing_market=listing, catalog_market_code=category, is_etf=is_etf,
                raw_json=json.dumps(raw, ensure_ascii=False), discovered_at="2024-12-31T00:00:00Z")


def price_rows(calendar, count=61, *, symbol="005930", market="domestic"):
    return [dict(symbol=symbol, exchange="KRX" if market == "domestic" else "ND", trade_date=day,
                 open="100", high="110", low="90", close="100", volume=10000,
                 trade_value="1000" if market == "domestic" else "1000000", change=None,
                 change_rate=None, adjustment_type=None, adjustment_rate=None,
                 currency="KRW" if market == "domestic" else "USD", collected_at="2025-01-01T00:00:00Z")
            for day in calendar[:count]]


def create_source(path, rows, catalog, market):
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE instruments(
                symbol TEXT NOT NULL,exchange TEXT NOT NULL,name TEXT NOT NULL,
                english_name TEXT,listing_market TEXT,catalog_market_code TEXT,is_etf INTEGER,
                raw_json TEXT NOT NULL,discovered_at TEXT NOT NULL,PRIMARY KEY(symbol,exchange));
            CREATE TABLE daily_bars(
                symbol TEXT NOT NULL,exchange TEXT NOT NULL,trade_date TEXT NOT NULL,
                open TEXT,high TEXT,low TEXT,close TEXT,volume INTEGER,trade_value TEXT,
                change TEXT,change_rate TEXT,adjustment_type TEXT,adjustment_rate TEXT,
                currency TEXT,collected_at TEXT NOT NULL,PRIMARY KEY(symbol,exchange,trade_date));
        """)
        db.execute("INSERT INTO metadata VALUES('market',?)", (market,))
        db.executemany("INSERT INTO instruments VALUES(" + ",".join("?" for _ in CATALOG_COLUMNS) + ")",
                       [tuple(item[key] for key in CATALOG_COLUMNS) for item in catalog])
        db.executemany("INSERT INTO daily_bars VALUES(" + ",".join("?" for _ in BAR_COLUMNS) + ")",
                       [tuple(row[key] for key in BAR_COLUMNS) for row in rows])
        db.commit()


class CleanDailyDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.calendar = sessions()
        self.as_of = date.fromisoformat(self.calendar[-1]) + timedelta(days=1)
        self.serial = 0

    def source(self, rows, catalog=None, market="domestic"):
        self.serial += 1
        path = self.folder / f"source_{self.serial}.sqlite3"
        if catalog is None:
            catalog = [catalog_item("005930" if market == "domestic" else "AAPL", market=market)]
        create_source(path, rows, catalog, market)
        return path

    def run_build(self, source, *, market="domestic", output=None, as_of=None, session_dates=None):
        output = output or source.with_name(source.stem + "_clean.sqlite3")
        policy = QualityPolicy(min_median_turnover=1e9 if market == "domestic" else 1e6)
        with redirect_stdout(io.StringIO()):
            result = build_database(source, output, market=market, as_of=as_of or self.as_of,
                                    policy=policy, session_dates=self.calendar if session_dates is None else session_dates)
        return output, result

    def connect(self, path):
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        self.addCleanup(db.close)
        return db

    def assert_approved_windows(self, db, lookback=30):
        order = {day: index for index, day in enumerate(self.calendar)}
        for sample in db.execute("SELECT * FROM training_samples"):
            first, last, target = (sample[key] for key in ("input_start_date", "input_end_date", "target_date"))
            self.assertEqual(order[target] - order[first], lookback)
            self.assertEqual(order[target] - order[last], 1)
            included = db.execute("SELECT trade_date,segment_id FROM daily_bars WHERE symbol=? AND exchange=? "
                                  "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
                                  (sample["symbol"], sample["exchange"], first, target)).fetchall()
            self.assertEqual([row["trade_date"] for row in included], self.calendar[order[first]:order[target] + 1])
            self.assertEqual({row["segment_id"] for row in included}, {sample["segment_id"]})
        self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")

    def test_completed_build_preserves_source_bytes_and_records_expected_window(self):
        rows = price_rows(self.calendar)
        rows[60]["close"] = "101"
        source = self.source(rows)
        original = source.read_bytes()
        output, summary = self.run_build(source)
        self.assertEqual(source.read_bytes(), original)
        self.assertTrue(summary["source_unchanged"])
        self.assertEqual(summary["row_decisions"], {"not_in_approved_window": 30, "kept": 31})
        self.assertEqual(summary["totals"]["source_rows"], 61)
        self.assertEqual(summary["totals"]["samples"], 1)
        db = self.connect(output)
        sample, = db.execute("SELECT * FROM training_samples").fetchall()
        self.assertEqual((sample["input_start_date"], sample["input_end_date"], sample["target_date"]),
                         (self.calendar[30], self.calendar[59], self.calendar[60]))
        self.assertEqual(sample["target_up"], 1)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0], 31)
        metadata = {row[0]: json.loads(row[1]) for row in db.execute("SELECT * FROM metadata")}
        self.assertEqual(metadata["build_status"], "complete")
        self.assertTrue(metadata["requires_training_samples"])
        self.assertEqual(metadata["source_fingerprints"][source.name]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8")), summary)
        self.assertFalse(output.with_name(output.name + ".building").exists())
        self.assert_approved_windows(db)

    def test_overwrite_source_existing_output_building_and_summary_are_blocked(self):
        source = self.source(price_rows(self.calendar))
        before = source.read_bytes()
        with self.assertRaises(ValueError):
            self.run_build(source, output=source)
        output, _ = self.run_build(source)
        output_before = output.read_bytes()
        with self.assertRaises(ValueError):
            self.run_build(source, output=output)
        fresh = self.folder / "unfinished.sqlite3"
        building = fresh.with_name(fresh.name + ".building")
        building.write_bytes(b"unfinished existing output")
        with self.assertRaises(ValueError):
            self.run_build(source, output=fresh)
        another = self.folder / "has_summary.sqlite3"
        another.with_suffix(".summary.json").write_text("existing summary", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_build(source, output=another)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(output.read_bytes(), output_before)
        self.assertEqual(building.read_bytes(), b"unfinished existing output")
        self.assertEqual(another.with_suffix(".summary.json").read_text(encoding="utf-8"), "existing summary")

    def test_accounting_includes_retained_excluded_missing_catalog_and_catalog_without_bars(self):
        rows = price_rows(self.calendar)
        rows += price_rows(self.calendar, 5, symbol="069500")
        rows += price_rows(self.calendar, 3, symbol="999990")
        catalog = [catalog_item(), catalog_item("069500", etf=True), catalog_item("123450")]
        output, summary = self.run_build(self.source(rows, catalog))
        self.assertEqual(sum(summary["row_decisions"].values()), len(rows))
        self.assertEqual(summary["row_decisions"]["instrument_excluded"], 5)
        self.assertEqual(summary["row_decisions"]["instrument_review"], 3)
        db = self.connect(output)
        audit = {row["symbol"]: row for row in db.execute("SELECT * FROM instrument_audit")}
        self.assertEqual(audit["005930"]["status"], "retained")
        self.assertEqual(audit["005930"]["valid_rows"], 61)
        self.assertEqual(audit["005930"]["stored_rows"], 31)
        self.assertEqual(audit["069500"]["status"], "excluded")
        self.assertEqual(audit["999990"]["status"], "review")
        self.assertIn("CATALOG_KEY_NOT_FOUND", json.loads(audit["999990"]["reasons"]))
        self.assertEqual(audit["123450"]["status"], "no_exact_source_bars")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM source_instruments").fetchone()[0], 3)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM instruments").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM bar_audit WHERE decision='not_in_approved_window'").fetchone()[0], 30)
        self.assert_approved_windows(db)

    def test_case_only_us_identity_is_quarantined_never_renamed_or_combined(self):
        rows = price_rows(self.calendar, market="us", symbol="aapl")
        source = self.source(rows, [catalog_item("AAPL", market="us")], market="us")
        output, summary = self.run_build(source, market="us")
        self.assertEqual(summary["row_decisions"], {"instrument_review": 61})
        db = self.connect(output)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0], 0)
        audit = {row["symbol"]: row for row in db.execute("SELECT * FROM instrument_audit")}
        self.assertEqual(audit["aapl"]["status"], "review")
        self.assertEqual(json.loads(audit["aapl"]["candidate_keys"]), [["ND", "AAPL"]])
        self.assertIn("CASE_ONLY_CANDIDATE_REQUIRES_VERIFICATION", json.loads(audit["aapl"]["reasons"]))
        self.assertEqual(audit["AAPL"]["status"], "no_exact_source_bars")

    def test_missing_session_and_invalid_ohlc_do_not_bridge_or_skip_warmup(self):
        for bad_kind in ("missing", "invalid"):
            with self.subTest(bad_kind=bad_kind):
                rows = price_rows(self.calendar, 125)
                if bad_kind == "missing":
                    del rows[60]
                else:
                    rows[60]["close"] = "NaN"
                output, summary = self.run_build(self.source(rows))
                db = self.connect(output)
                sample_dates = [row[0] for row in db.execute("SELECT input_end_date FROM training_samples ORDER BY input_end_date")]
                self.assertEqual(sample_dates, self.calendar[120:124])
                self.assertEqual(sum(summary["row_decisions"].values()), len(rows))
                if bad_kind == "invalid":
                    invalid = db.execute("SELECT * FROM bar_audit WHERE decision='hard_invalid'").fetchall()
                    self.assertEqual(len(invalid), 1)
                    self.assertIn("INVALID_CLOSE", json.loads(invalid[0]["reasons"]))
                self.assert_approved_windows(db)

    def test_invalid_dates_and_non_session_rows_inside_source_order_are_quarantined(self):
        rows = price_rows(self.calendar)
        # Both strings sort between valid input rows without replacing a
        # scheduled session. Their presence must not crash or join a bad row.
        invalid_day = self.calendar[40] + "junk"
        friday = next(date.fromisoformat(day) for day in self.calendar[35:50]
                      if date.fromisoformat(day).weekday() == 4)
        saturday = (friday + timedelta(days=1)).isoformat()
        rows += [dict(rows[40], trade_date=invalid_day), dict(rows[40], trade_date=saturday)]
        output, summary = self.run_build(self.source(rows))
        self.assertEqual(summary["totals"]["samples"], 1)
        self.assertEqual(summary["row_decisions"]["hard_invalid"], 2)
        db = self.connect(output)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0], 31)
        invalid = {row["trade_date"]: json.loads(row["reasons"])
                   for row in db.execute("SELECT * FROM bar_audit WHERE decision='hard_invalid'")}
        self.assertIn("INVALID_DATE", invalid[invalid_day])
        self.assertIn("NON_SESSION_DATE", invalid[saturday])
        self.assert_approved_windows(db)

    def test_extreme_valid_movements_and_sparse_zero_volume_inputs_are_preserved(self):
        rows = price_rows(self.calendar)
        rows[40].update(open="100", low="50", high="200", close="150")
        rows[41].update(volume=0, trade_value="0")
        output, summary = self.run_build(self.source(rows))
        self.assertEqual(summary["totals"]["samples"], 1)
        db = self.connect(output)
        extreme = db.execute("SELECT * FROM daily_bars WHERE trade_date=?", (self.calendar[40],)).fetchone()
        self.assertEqual(extreme["close"], "150")
        self.assertIn("ABS_RETURN_GE_50PCT", json.loads(extreme["quality_flags"]))
        self.assertIn("HIGH_LOW_RATIO_GE_2", json.loads(extreme["quality_flags"]))
        zero = db.execute("SELECT * FROM daily_bars WHERE trade_date=?", (self.calendar[41],)).fetchone()
        self.assertEqual(zero["volume"], 0)
        self.assertIn("ZERO_VOLUME", json.loads(zero["quality_flags"]))
        self.assert_approved_windows(db)

    def test_liquidity_exclusions_leave_separate_segments_and_only_approved_windows(self):
        rows = price_rows(self.calendar, 200)
        for row in rows[90:130]:
            row["volume"] = 1
        output, summary = self.run_build(self.source(rows))
        db = self.connect(output)
        approved = [row[0] for row in db.execute("SELECT input_end_date FROM training_samples ORDER BY input_end_date")]
        self.assertEqual(approved, self.calendar[59:99] + self.calendar[140:199])
        self.assertEqual(summary["totals"]["valid_candidate_rows"], 200)
        self.assertEqual(summary["totals"]["samples"], 99)
        self.assertEqual(db.execute("SELECT COUNT(DISTINCT segment_id) FROM daily_bars").fetchone()[0], 2)
        for day in self.calendar[100:111]:
            self.assertIsNone(db.execute("SELECT 1 FROM daily_bars WHERE trade_date=?", (day,)).fetchone())
        self.assert_approved_windows(db)

    def test_target_turnover_and_positive_tiny_volume_do_not_select_outcomes(self):
        rows = price_rows(self.calendar)
        rows[60].update(open="400", high="800", low="100", close="500", volume=1, trade_value="0")
        output, summary = self.run_build(self.source(rows))
        self.assertEqual(summary["totals"]["samples"], 1)
        db = self.connect(output)
        self.assertEqual(db.execute("SELECT target_up FROM training_samples").fetchone()[0], 1)
        self.assert_approved_windows(db)

    def test_as_of_is_exclusive_and_no_incomplete_target_is_published(self):
        source = self.source(price_rows(self.calendar, 62))
        output, summary = self.run_build(source, as_of=date.fromisoformat(self.calendar[60]))
        db = self.connect(output)
        self.assertEqual(summary["totals"]["samples"], 0)
        self.assertEqual(summary["row_decisions"]["hard_invalid"], 2)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0], 0)
        reasons = [json.loads(row[0]) for row in db.execute("SELECT reasons FROM bar_audit WHERE decision='hard_invalid'")]
        self.assertTrue(all("INCOMPLETE_SESSION" in item for item in reasons))

    def test_bad_calendar_is_rejected_even_when_all_instruments_are_excluded(self):
        source = self.source(price_rows(self.calendar), [catalog_item(etf=True)])
        for calendar in ([], ["20240101"], ["2024-01-02", "2024-01-01"], ["2024-01-01"] * 2):
            with self.subTest(calendar=calendar), self.assertRaises(ValueError):
                output = self.folder / ("invalid_calendar_" + str(self.serial) + ".sqlite3")
                self.serial += 1
                self.run_build(source, output=output, session_dates=calendar)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
