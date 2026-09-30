"""Offline paper-mode guard tests; every bar here is a synthetic test fixture."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta, timezone
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.research.minute_transfer import Bar, sha256_file
from examples.paper_minute_signal import run as paper_run
from examples.train_daily_to_minute import run as train_run
from tests.test_minute_transfer import daily_bars, minute_bars, row


class PaperMinuteSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        daily = cls.root / "train_daily.jsonl"
        minute = cls.root / "train_minute.jsonl"
        daily.write_text("".join(json.dumps(row(b)) + "\n" for b in daily_bars()),
                         encoding="utf-8")
        minute.write_text("".join(json.dumps(row(b)) + "\n" for b in minute_bars()),
                          encoding="utf-8")
        minute.with_suffix(minute.suffix + ".receipt.json").write_text(json.dumps({
            "source": "kiwoom_rest_demo_minute_chart", "sha256": sha256_file(minute),
            "market": "domestic", "interval_minutes": 5,
            "regular_session_filter": True,
            "accepted_bars": len(minute_bars())}), encoding="utf-8")
        cls.bundle = cls.root / "bundle"
        trained = train_run(SimpleNamespace(
            market="domestic", daily=daily, minute=minute, output=cls.bundle,
            as_of="2026-01-01T00:00:00+00:00", lookback=5, horizon=1,
            fee_bps=2., slippage_bps=8., purge_sessions=1,
            pretrain_epochs=1, adapt_epochs=1, min_samples_per_split=1,
            min_validation_trades=1, max_train_samples=100, seed=17))
        assert trained["sources"]["minute_collector_receipt_verified"] is True
        cls.session_day = date(2026, 10, 1)
        session = session_on(Market.DOMESTIC, cls.session_day)
        cls.bars = tuple(Bar("domestic", "005930",
                             session.opened + timedelta(minutes=5 * (index + 2)),
                             100. + .02 * index, 100.3 + .02 * index,
                             99.8 + .02 * index, 100.1 + .02 * index,
                             1000. + index, 5, "KRX") for index in range(28))
        cls.as_of = cls.bars[-1].timestamp + timedelta(minutes=6)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def setUp(self):
        self.case = TemporaryDirectory(dir=self.root)
        self.directory = Path(self.case.name)
        self.data = self.directory / "fresh.jsonl"
        self._write_snapshot(self.bars, collected_at=self.as_of - timedelta(minutes=1))
        self.args = SimpleNamespace(bundle=self.bundle, minute=self.data,
                                    market="domestic", exchange="KRX", symbol="005930",
                                    session=self.session_day.isoformat(),
                                    output=self.directory / "paper.json")

    def tearDown(self):
        self.case.cleanup()

    def _write_snapshot(self, bars, *, collected_at):
        self.data.write_text("".join(json.dumps(row(b)) + "\n" for b in bars),
                             encoding="utf-8")
        self.data.with_suffix(self.data.suffix + ".receipt.json").write_text(json.dumps({
            "source": "kiwoom_rest_demo_minute_chart",
            "market": "domestic", "exchange": "KRX",
            "symbols": sorted({bar.symbol for bar in bars}),
            "interval_minutes": 5, "regular_session_filter": True,
            "accepted_bars": len(bars), "sha256": sha256_file(self.data),
            "collected_at_utc": collected_at.astimezone(timezone.utc).isoformat(),
        }), encoding="utf-8")

    def test_three_paper_models_and_new_report_only(self):
        report = paper_run(self.args, now=self.as_of)
        self.assertTrue(self.args.output.is_file())
        self.assertEqual(json.loads(self.args.output.read_text(encoding="utf-8")), report)
        self.assertEqual(set(report["models"]), {"linear", "conv", "gru"})
        self.assertEqual(report["completed_contiguous_history_count"], 5)
        self.assertTrue(report["source"]["collector_receipt_verified"])
        self.assertFalse(report["safety"]["deployment_allowed"])
        self.assertFalse(report["safety"]["order_routing_connected"])
        with self.assertRaises(FileExistsError):
            paper_run(self.args, now=self.as_of)

    def test_null_threshold_never_emits_candidate(self):
        copy = self.directory / "null_threshold_bundle"
        shutil.copytree(self.bundle, copy)
        manifest_path = copy / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for model in manifest["models"].values():
            model["validation_threshold"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.args.bundle = copy
        report = paper_run(self.args, now=self.as_of)
        self.assertTrue(all(model["paper_candidate"] is False
                            for model in report["models"].values()))

    def test_receipt_hash_and_market_mismatch_fail_closed(self):
        self.data.write_text(self.data.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "receipt does not match"):
            paper_run(self.args, now=self.as_of)
        self.assertFalse(self.args.output.exists())
        self._write_snapshot(self.bars, collected_at=self.as_of - timedelta(minutes=1))
        self.args.exchange = "NYS"
        with self.assertRaisesRegex(ValueError, "receipt does not match"):
            paper_run(self.args, now=self.as_of)
        self.assertFalse(self.args.output.exists())

    def test_untrained_stock_cannot_emit_paper_signal(self):
        other_stock_bars = tuple(replace(bar, symbol="000660") for bar in self.bars)
        self._write_snapshot(other_stock_bars, collected_at=self.as_of - timedelta(minutes=1))
        self.args.symbol = "000660"
        with self.assertRaisesRegex(ValueError, "not in minute adaptation training"):
            paper_run(self.args, now=self.as_of)
        self.assertFalse(self.args.output.exists())

    def test_stale_snapshot_stale_bar_and_closed_session_fail_closed(self):
        old_receipt = self.as_of - timedelta(minutes=16)
        self._write_snapshot(self.bars, collected_at=old_receipt)
        with self.assertRaisesRegex(ValueError, "receipt is stale"):
            paper_run(self.args, now=self.as_of)
        later = self.bars[-1].timestamp + timedelta(minutes=16)
        self._write_snapshot(self.bars, collected_at=later - timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, "too stale"):
            paper_run(self.args, now=later)
        end = session_on(Market.DOMESTIC, self.session_day).closed
        with self.assertRaisesRegex(ValueError, "not in progress"):
            paper_run(self.args, now=end)
        self.assertFalse(self.args.output.exists())

    def test_unfinished_bar_and_last_window_gap_fail_closed(self):
        extra = Bar("domestic", "005930", self.bars[-1].timestamp + timedelta(minutes=5),
                    101., 101.3, 100.8, 101.1, 1000., 5, "KRX")
        self._write_snapshot(self.bars + (extra,),
                             collected_at=self.as_of - timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, "still forming"):
            paper_run(self.args, now=self.as_of)
        self._write_snapshot(self.bars[:-2] + self.bars[-1:],
                             collected_at=self.as_of - timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            paper_run(self.args, now=self.as_of)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
