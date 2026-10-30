"""Read-only, completed-session daily research export tests."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dockdack.models import Market
from examples.export_daily_research_bars import export_daily


class ExportDailyResearchBarsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.database = root / "daily.sqlite3"
        self.output = root / "research" / "daily.jsonl"
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TABLE daily_bars (
                symbol TEXT, exchange TEXT, trade_date TEXT, open TEXT,
                high TEXT, low TEXT, close TEXT, volume INTEGER,
                currency TEXT, quality_flags TEXT)""")
            for day, close, flags in (
                ("2024-06-03", "104", "[]"),
                ("2024-06-04", "105", "[]"),
                ("2024-06-05", "106", "[]"),
                ("2024-06-06", "107", "[]"),  # KRX holiday
                ("2024-06-07", "108", '["source_warning"]'),
            ):
                db.execute("INSERT INTO daily_bars VALUES(?,?,?,?,?,?,?,?,?,?)",
                           ("005930", "KRX", day, "100", "110", "99", close,
                            1000, "KRW", flags))
            db.commit()
        self.before = hashlib.sha256(self.database.read_bytes()).hexdigest()

    def _export(self):
        return export_daily(
            database=self.database, market=Market.DOMESTIC,
            exchange="KRX", symbols=("005930",),
            from_date=date(2024, 6, 3), through_date=date(2024, 6, 7),
            output=self.output,
            now=datetime(2024, 6, 6, 9, tzinfo=ZoneInfo("Asia/Seoul")),
        )

    def test_only_completed_clean_sessions_and_source_unchanged(self):
        info = self._export()
        self.assertEqual(info["source_rows"], 5)
        self.assertEqual(info["quality_flagged_skipped"], 1)
        self.assertEqual(info["closed_session_skipped"], 1)
        self.assertEqual(info["accepted_bars"], 3)
        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["bar_minutes"], 1440)
        self.assertEqual(rows[0]["timestamp"], "2024-06-03T15:30:00+09:00")
        self.assertEqual(rows[-1]["close"], "106")
        receipt = json.loads(self.output.with_suffix(".jsonl.receipt.json").read_text(
            encoding="utf-8"))
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(self.before, hashlib.sha256(self.database.read_bytes()).hexdigest())

    def test_never_overwrites_research_output(self):
        self._export()
        with self.assertRaises(FileExistsError):
            self._export()

    def test_rejects_duplicate_symbols_and_invalid_date_range(self):
        with self.assertRaises(ValueError):
            export_daily(database=self.database, market=Market.DOMESTIC,
                         exchange="KRX", symbols=("005930", "005930"),
                         from_date=date(2024, 6, 3), through_date=date(2024, 6, 7),
                         output=self.output)
        with self.assertRaises(ValueError):
            export_daily(database=self.database, market=Market.DOMESTIC,
                         exchange="KRX", symbols=("005930",),
                         from_date=date(2024, 6, 7), through_date=date(2024, 6, 3),
                         output=self.output)


if __name__ == "__main__":
    unittest.main()
