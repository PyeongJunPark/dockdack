"""Synthetic, broker-free contracts for the Mark1 target/horizon path bank."""
from __future__ import annotations

from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from dockdack.mark1_target_horizon_data import load_horizon_bank
from dockdack.research_artifacts import sha256_file


EPOCH = date(1970, 1, 1)


def _sessions(count=45):
    days, day = [], date(2024, 1, 2)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


class HorizonBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.database = self.workspace / "domestic_clean.sqlite3"
        self.days = _sessions()
        self.bounds = dict(zip(
            ("train", "tune", "calibration", "selection", "test"),
            (self.days[index].isoformat() for index in (36, 38, 40, 42, 44))))

    def _fixture(self, *, overrides=None, missing=(), sample_days=(30,),
                 splits=None, bounds=None):
        overrides = overrides or {}
        bars = []
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript("""
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE instruments(symbol TEXT,exchange TEXT);
                CREATE TABLE sessions(session_date TEXT PRIMARY KEY,ordinal INTEGER UNIQUE);
                CREATE TABLE daily_bars(symbol TEXT,exchange TEXT,trade_date TEXT,
                    open TEXT,high TEXT,low TEXT,close TEXT,volume INTEGER,
                    segment_id INTEGER,currency TEXT);
                CREATE TABLE training_samples(symbol TEXT,exchange TEXT,target_date TEXT);
            """)
            db.executemany("INSERT INTO metadata VALUES (?,?)", (
                ("market", '"domestic"'), ("build_status", '"complete"'),
                ("as_of_exclusive", '"2024-04-01"')))
            db.execute("INSERT INTO instruments VALUES ('000001','KRX')")
            for index, day in enumerate(self.days):
                db.execute("INSERT INTO sessions VALUES (?,?)", (day.isoformat(), index))
                o, h, l, c, volume, segment = overrides.get(index, (100., 102., 99., 100., 1000, 1))
                bars.append((o, h, l, c, volume))
                if index not in missing:
                    db.execute("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("000001", "KRX", day.isoformat(),
                                str(o), str(h), str(l), str(c), volume, segment, "KRW"))
        lookup = {value: index for index, value in enumerate(sample_days)}
        if splits is None:
            splits = {"train": np.asarray(list(range(len(sample_days))), dtype=np.int64)}
        dataset = SimpleNamespace(
            bars=np.asarray(bars, dtype=np.float32),
            starts=np.asarray([index - 30 for index in sample_days], dtype=np.int64),
            target_dates=np.asarray([(self.days[index] - EPOCH).days for index in sample_days],
                                    dtype=np.int32),
            target_ohlc=np.asarray([bars[index][:4] for index in sample_days], dtype=np.float64),
            symbol_ids=np.zeros(len(sample_days), dtype=np.int32),
            splits=splits,
            manifest={"market": "domestic", "symbols": [
                {"symbol_id": 0, "symbol": "000001", "exchange": "KRX"}],
                "period_ends": list((bounds or self.bounds).values())[:4]},
        )
        source = {"version": 2, "database_path": str(self.database),
                  "database_sha256": sha256_file(self.database), "market": "domestic"}
        return dataset, source, lookup

    def _load(self, dataset, source, *, max_horizon=3, bounds=None):
        return load_horizon_bank(dataset, source, self.workspace, "domestic",
                                 max_horizon=max_horizon,
                                 split_end_dates=bounds or self.bounds)

    def test_first_touch_uses_entry_day_as_day_one(self):
        dataset, source, _ = self._fixture(overrides={31: (100., 104., 99., 100., 1000, 1),
                                                      32: (100., 106., 99., 100., 1000, 1)})
        bank = self._load(dataset, source)
        outcome = bank.outcomes(target_pct=3, horizon=3, lookback=20)
        self.assertTrue(outcome.eligible[0])
        self.assertTrue(outcome.hit[0])
        self.assertEqual(int(outcome.exit_session[0]), 2)
        self.assertEqual(int(outcome.exit_date[0]), (self.days[31] - EPOCH).days)
        self.assertEqual(float(outcome.exit_price[0]), 103.)
        self.assertAlmostEqual(float(outcome.gross_return[0]), .03)
        self.assertEqual(bank.histories(dataset, lookback=20, indices=np.array([0])).shape,
                         (1, 20, 5))
        self.assertEqual(int(bank.entry_ordinals[0]), 30)
        self.assertEqual(int(bank.symbol_ids[0]), 0)
        self.assertEqual(str(bank.split_names[0]), "train")
        self.assertFalse(Path(str(self.database) + "-wal").exists())

    def test_entry_day_take_profit_and_one_day_timeout(self):
        dataset, source, _ = self._fixture(overrides={30: (100., 103.5, 99., 101., 1000, 1)})
        bank = self._load(dataset, source)
        hit = bank.outcomes(target_pct=3, horizon=1)
        self.assertTrue(hit.hit[0])
        self.assertEqual(int(hit.exit_session[0]), 1)
        self.assertEqual(float(hit.exit_price[0]), 103.)
        timeout = bank.outcomes(target_pct=4, horizon=1)
        self.assertFalse(timeout.hit[0])
        self.assertEqual(float(timeout.exit_price[0]), 101.)

    def test_no_touch_sells_at_horizon_close_and_cost_is_separate(self):
        dataset, source, _ = self._fixture(overrides={32: (100., 102., 98., 98., 1000, 1)})
        bank = self._load(dataset, source)
        outcome = bank.outcomes(target_pct=3, horizon=3, cost_bps=20,
                                slippage_bps=10)
        self.assertTrue(outcome.eligible[0])
        self.assertFalse(outcome.hit[0])
        self.assertEqual(int(outcome.exit_session[0]), 3)
        self.assertEqual(float(outcome.exit_price[0]), 98.)
        self.assertAlmostEqual(float(outcome.gross_return[0]), -.02)
        self.assertAlmostEqual(float(outcome.net_return[0]),
                               98. * .998 / (100. * 1.002) - 1)

    def test_missing_session_never_backfills_later_high(self):
        dataset, source, _ = self._fixture(overrides={32: (100., 105., 99., 100., 1000, 1)},
                                           missing=(31,))
        bank = self._load(dataset, source)
        self.assertTrue(bank.outcomes(horizon=1).eligible[0])
        result = bank.outcomes(horizon=3)
        self.assertFalse(result.eligible[0])
        self.assertFalse(result.hit[0])
        self.assertEqual(int(result.exit_date[0]), -1)
        self.assertTrue(np.isnan(result.net_return[0]))

    def test_zero_volume_or_new_segment_invalidates_future_prefix(self):
        for changed in ((100., 104., 99., 100., 0, 1),
                        (100., 104., 99., 100., 1000, 2)):
            with self.subTest(changed=changed):
                if self.database.exists():
                    self.database.unlink()
                dataset, source, _ = self._fixture(overrides={31: changed})
                bank = self._load(dataset, source)
                self.assertTrue(bank.outcomes(horizon=1).eligible[0])
                self.assertFalse(bank.outcomes(horizon=2).eligible[0])

    def test_split_boundary_excludes_future_horizon_crossing(self):
        boundaries = dict(self.bounds)
        boundaries["train"] = self.days[31].isoformat()
        dataset, source, _ = self._fixture(bounds=boundaries)
        bank = self._load(dataset, source, bounds=boundaries)
        self.assertTrue(bank.outcomes(horizon=2).eligible[0])
        self.assertFalse(bank.outcomes(horizon=3).eligible[0])

    def test_nontrain_history_must_begin_after_prior_split(self):
        dataset, source, _ = self._fixture(sample_days=(38,),
                                           splits={"tune": np.array([0], dtype=np.int64)})
        bank = self._load(dataset, source)
        self.assertFalse(bank.outcomes(horizon=1, lookback=2).eligible[0])
        self.assertTrue(bank.outcomes(horizon=1, lookback=1).eligible[0])
        self.assertFalse(bank.outcomes(horizon=1, lookback=1,
                                       split_name="train").eligible[0])

    def test_entry_must_match_frozen_cache(self):
        dataset, source, _ = self._fixture()
        dataset.target_ohlc[0, 0] = 101.
        with self.assertRaisesRegex(ValueError, "approved entry differs"):
            self._load(dataset, source)


if __name__ == "__main__":
    unittest.main()
