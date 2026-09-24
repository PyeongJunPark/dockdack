from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from dockdack.mark1_backtest_data import load_price_panel
from dockdack.clean_daily_dataset import create_schema, write_metadata


class Mark1BacktestDataTests(unittest.TestCase):
    def fixture(self, path):
        with closing(sqlite3.connect(path)) as connection, connection:
            create_schema(connection)
            write_metadata(connection, {
                "schema_version": "clean-daily-v1", "build_status": "complete",
                "requires_training_samples": True, "market": "domestic", "policy": {"lookback": 30},
                "as_of_exclusive": "2025-01-09", "source_fingerprints": {"raw": {"sha256": "fixture"}},
            })
            for ordinal, day in enumerate(("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08")):
                connection.execute("INSERT INTO sessions VALUES(?,?)", (day, ordinal))
            for symbol in ("AAA", "BBB"):
                connection.execute("INSERT INTO instruments VALUES(?,?,?,?,?,?,?,?,?)",
                                   (symbol, "KRX", symbol, symbol, "KRX", "0", 0, "{}", "2025-01-09"))
            for ordinal, day in enumerate(("2025-01-02", "2025-01-06", "2025-01-07")):
                connection.execute("INSERT INTO daily_bars VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("AAA", "KRX", day, "100.0000000001", "102.0000000001", "99.2", "100.0000000001",
                     0 if day == "2025-01-07" else 10000, None, None, None, None, None, "KRW",
                     "2025-01-09", ordinal + 1, ordinal + 1, "[]"))
        return [{"symbol_id": 7, "symbol": "AAA", "exchange": "KRX"},
                {"symbol_id": 13, "symbol": "BBB", "exchange": "KRX"}]

    def test_readonly_precise_all_bars_and_explicit_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            symbols = self.fixture(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            prices, sessions, metadata = load_price_panel(path, "domestic", symbols, "2025-01-02", "2025-01-08")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(sessions), 5)
            self.assertEqual(len(prices), 3)  # No approved targets in this fixture: still include all prices.
            self.assertAlmostEqual(prices[(7, sessions[0])][0], 100.0000000001, places=12)
            self.assertNotIn((7, sessions[1]), prices)
            self.assertNotIn((13, sessions[0]), prices)
            self.assertEqual(metadata["missing_symbol_sessions"], 7)
            self.assertEqual(metadata["internal_missing_symbol_sessions"], 1)
            self.assertEqual(metadata["zero_volume_keys"], [[7, sessions[3]]])
            self.assertEqual(metadata["symbol_coverage"][1]["bars"], 0)
            same, same_sessions, _ = load_price_panel(path, "domestic", symbols, sessions[0], sessions[-1])
            self.assertEqual(same, prices)
            self.assertEqual(same_sessions, sessions)

    def test_reject_bad_currency_and_ohlc(self):
        for change in ("currency='USD'", "high='99'", "open='nan'", "volume=-1",
                       "trade_date='2025-01-04'"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "clean.sqlite3"
                symbols = self.fixture(path)
                with closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute(f"UPDATE daily_bars SET {change} WHERE trade_date='2025-01-02'")
                with self.assertRaises(ValueError):
                    load_price_panel(path, "domestic", symbols, "2025-01-02", "2025-01-08")

    def test_reject_bad_mapping_market_and_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            symbols = self.fixture(path)
            for mapping in ([], symbols + [symbols[0]], [{"symbol_id": True, "symbol": "AAA", "exchange": "KRX"}],
                            [{"symbol_id": 0, "symbol": "MISSING", "exchange": "KRX"}]):
                with self.assertRaises(ValueError):
                    load_price_panel(path, "domestic", mapping, "2025-01-02", "2025-01-08")
            for start, end in ((True, "2025-01-08"), ("20250102", "2025-01-08"),
                               ("2025-01-02", "2025-01-09"), ("2025-01-01", "2025-01-08"),
                               ("2025-01-08", "2025-01-02"), ("2025-01-04", "2025-01-05")):
                with self.assertRaises(ValueError):
                    load_price_panel(path, "domestic", symbols, start, end)
            with self.assertRaises(ValueError):
                load_price_panel(path, "us", symbols, "2025-01-02", "2025-01-08")

    def test_incomplete_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.sqlite3"
            symbols = self.fixture(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE metadata SET value='\"building\"' WHERE key='build_status'")
            with self.assertRaises(ValueError):
                load_price_panel(path, "domestic", symbols, "2025-01-02", "2025-01-08")


if __name__ == "__main__":
    unittest.main()
