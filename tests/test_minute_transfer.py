"""Safety/identity tests; synthetic OHLCV is used only inside test fixtures."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from dockdack.market_schedule import session_on
from dockdack.models import Market
from dockdack.research.minute_transfer import (
    ARCHITECTURES, Bar, build_model, load_bars, load_paper_artifact, make_samples,
    paper_probability, split_sessions, train_transfer,
)
from examples.train_daily_to_minute import run


def sessions(count: int, *, first=date(2025, 1, 2)):
    result = []
    day = first
    while len(result) < count:
        session = session_on(Market.DOMESTIC, day)
        if session is not None:
            result.append(session)
        day += timedelta(days=1)
    return result


def daily_bars(count: int = 60, *, first=date(2025, 1, 2)):
    result = []
    for i, session in enumerate(sessions(count, first=first)):
        price = 100. + .03 * i + (i % 7) * .06
        result.append(Bar("domestic", "005930", session.closed,
                          price, price + 1., price - 1., price + .1,
                          1000. + 10 * i, 1440, "KRX"))
    return tuple(result)


def minute_bars(count_sessions: int = 30, bars_per_session: int = 36):
    result = []
    for day_i, session in enumerate(sessions(count_sessions, first=date(2025, 4, 1))):
        for j in range(bars_per_session):
            price = 100. + day_i * .01 + j * .018 + ((j + day_i) % 5) * .09
            result.append(Bar("domestic", "005930",
                              session.opened + timedelta(minutes=5 * (j + 1)),
                              price, price + .25, price - .25, price + .02,
                              800. + j * 3, 5, "KRX"))
    return tuple(result)


def row(bar: Bar):
    value = asdict(bar)
    value["timestamp"] = bar.timestamp.isoformat()
    return value


class MinuteTransferDataTest(unittest.TestCase):
    def test_loader_exchange_optional_and_regular_session_filter(self):
        regular = minute_bars(1, 1)[0]
        early = row(regular)
        early["timestamp"] = (regular.timestamp - timedelta(minutes=10)).isoformat()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.jsonl"
            path.write_text("\n".join(json.dumps(v) for v in (early, row(regular))) + "\n")
            stats = {}
            loaded = load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(minutes=6),
                               stats=stats)
            self.assertEqual(loaded, (regular,))
            self.assertEqual(stats, {"jsonl_rows": 2, "regular_session_excluded": 1,
                                     "accepted_bars": 1})
            no_exchange = row(regular)
            no_exchange.pop("exchange")
            path.write_text(json.dumps(no_exchange) + "\n")
            self.assertEqual(load_bars(path, kind="minute",
                                       as_of=regular.timestamp + timedelta(minutes=6))[0].exchange, "")

    def test_loader_rejects_duplicate_missing_offset_and_unfinished_bar(self):
        regular = minute_bars(1, 1)[0]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.jsonl"
            text = json.dumps(row(regular)) + "\n"
            path.write_text(text + text)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(minutes=6))
            malformed = row(regular)
            malformed["timestamp"] = regular.timestamp.replace(tzinfo=None).isoformat()
            path.write_text(json.dumps(malformed) + "\n")
            with self.assertRaisesRegex(ValueError, "explicit UTC offset"):
                load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(minutes=6))
            malformed["timestamp"] = regular.timestamp.replace(tzinfo=timezone(timedelta(hours=8))).isoformat()
            path.write_text(json.dumps(malformed) + "\n")
            with self.assertRaisesRegex(ValueError, "exchange local zone"):
                load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(hours=2))
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "still forming"):
                load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(minutes=5))
            malformed = row(regular)
            malformed.pop("volume")
            path.write_text(json.dumps(malformed) + "\n")
            with self.assertRaisesRegex(ValueError, "requires"):
                load_bars(path, kind="minute", as_of=regular.timestamp + timedelta(minutes=6))

    def test_daily_close_and_missing_session_boundary(self):
        bars = daily_bars(30)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily.jsonl"
            path.write_text(json.dumps(row(bars[0])) + "\n")
            self.assertEqual(load_bars(path, kind="daily",
                                       as_of=bars[0].timestamp + timedelta(seconds=1)), (bars[0],))
            wrong = row(bars[0])
            wrong["timestamp"] = (bars[0].timestamp - timedelta(minutes=5)).isoformat()
            path.write_text(json.dumps(wrong) + "\n")
            with self.assertRaisesRegex(ValueError, "exchange session close"):
                load_bars(path, kind="daily", as_of=bars[0].timestamp + timedelta(seconds=1))
        absent = bars[15].session_date
        samples = make_samples(bars[:15] + bars[16:], kind="daily", lookback=5, horizon=1)
        self.assertTrue(samples)
        self.assertTrue(all(not (s.first_session < absent < s.exit_session) for s in samples))

    def test_missing_minute_breaks_window_and_features_do_not_read_label(self):
        bars = minute_bars(1, 12)
        original = make_samples(bars, kind="minute", lookback=5, horizon=1,
                                fee_bps_per_side=0, slippage_bps_per_side=0)
        self.assertTrue(original)
        missing = make_samples(bars[:5] + bars[6:], kind="minute", lookback=5, horizon=1)
        self.assertEqual(missing, ())
        changed = list(bars)
        target = changed[8]
        changed[8] = Bar(target.market, target.symbol, target.timestamp,
                         target.open * 1.1, target.high * 1.2, target.low,
                         target.close, target.volume, target.bar_minutes, target.exchange)
        altered = make_samples(changed, kind="minute", lookback=5, horizon=1,
                               fee_bps_per_side=0, slippage_bps_per_side=0)
        np.testing.assert_array_equal(original[0].features, altered[0].features)
        self.assertNotEqual(original[0].net_return, altered[0].net_return)
        self.assertEqual(original[0].entry_at, bars[7].timestamp)
        self.assertEqual(original[0].exit_at, bars[8].timestamp)

    def test_cost_and_purged_session_splits(self):
        rows = minute_bars()
        no_cost = make_samples(rows, kind="minute", lookback=5, horizon=1,
                               fee_bps_per_side=0, slippage_bps_per_side=0)
        cost = make_samples(rows, kind="minute", lookback=5, horizon=1,
                            fee_bps_per_side=2, slippage_bps_per_side=8)
        self.assertLess(cost[0].net_return, no_cost[0].net_return)
        splits = split_sessions(cost, purge_sessions=1)
        by_split = {name: {sample.entry_session for sample in selected}
                    for name, selected in splits.items()}
        self.assertTrue(all(splits.values()))
        self.assertFalse(by_split["train"] & by_split["val"])
        self.assertFalse(by_split["val"] & by_split["test"])
        self.assertLess(max(by_split["train"]), min(by_split["val"]))
        self.assertLess(max(by_split["val"]), min(by_split["test"]))


class MinuteTransferTrainingTest(unittest.TestCase):
    def test_three_distinct_architectures_and_paper_only_training(self):
        self.assertEqual(len(ARCHITECTURES), 3)
        self.assertEqual(len({sum(p.numel() for p in build_model(name, 5).parameters())
                              for name in ARCHITECTURES}), 3)
        artifacts, report = train_transfer(
            daily_bars(), minute_bars(), lookback=5, horizon=1, purge_sessions=1,
            pretrain_epochs=1, adapt_epochs=1, min_samples_per_split=1,
            min_validation_trades=1, max_train_samples=100, seed=17)
        self.assertEqual(set(artifacts), set(ARCHITECTURES))
        self.assertTrue(report["domain_mismatch"])
        self.assertFalse(report["deployment_allowed"])
        chronology = report["cross_domain_chronology"]
        self.assertTrue(chronology["daily_pretraining_strictly_before_minute_adaptation"])
        self.assertLess(chronology["latest_daily_pretraining_label_exit"],
                        chronology["earliest_minute_adaptation_entry"])
        self.assertEqual(report["minute_adaptation_identities"],
                         [{"exchange": "KRX", "symbol": "005930"}])
        for name, artifact in artifacts.items():
            self.assertGreater(artifact["test"]["samples"], 0)
            self.assertEqual(artifact["test"]["unit"],
                             "fraction per latency-adjusted bar-open to later bar-open historical proxy; not actual fills")
            history = minute_bars(1, 10)[-5:]
            output = paper_probability(artifact, history, architecture=name,
                                       as_of=history[-1].timestamp + timedelta(minutes=6))
            self.assertTrue(0 <= output["probability_proxy"] <= 1)
            self.assertFalse(output["deployment_allowed"])
            with self.assertRaisesRegex(ValueError, "completed"):
                paper_probability(artifact, history, architecture=name,
                                  as_of=history[-1].timestamp + timedelta(minutes=5))
            wrong_stock = tuple(replace(bar, symbol="000660") for bar in history)
            with self.assertRaisesRegex(ValueError, "not in minute adaptation training"):
                paper_probability(artifact, wrong_stock, architecture=name,
                                  as_of=history[-1].timestamp + timedelta(minutes=6))

    def test_future_daily_pretraining_cannot_leak_into_minute_adaptation(self):
        with self.assertRaisesRegex(ValueError, "future daily pretraining label"):
            train_transfer(
                daily_bars(first=date(2025, 5, 1)), minute_bars(),
                lookback=5, horizon=1, purge_sessions=1,
                pretrain_epochs=1, adapt_epochs=1, min_samples_per_split=1,
                min_validation_trades=1, max_train_samples=100, seed=17)

    def test_new_bundle_is_paper_only_hashed_and_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            daily, minute = folder / "daily.jsonl", folder / "minute.jsonl"
            daily.write_text("".join(json.dumps(row(b)) + "\n" for b in daily_bars()),
                             encoding="utf-8")
            minute.write_text("".join(json.dumps(row(b)) + "\n" for b in minute_bars()),
                              encoding="utf-8")
            output = folder / "new_bundle"
            args = SimpleNamespace(
                market="domestic", daily=daily, minute=minute, output=output,
                as_of="2026-01-01T00:00:00+00:00", lookback=5, horizon=1,
                fee_bps=2., slippage_bps=8., purge_sessions=1,
                pretrain_epochs=1, adapt_epochs=1, min_samples_per_split=1,
                min_validation_trades=1, max_train_samples=100, seed=17)
            manifest = run(args)
            self.assertFalse(manifest["safety"]["deployment_allowed"])
            self.assertEqual(manifest["sources"]["minute"]["regular_session_excluded"], 0)
            self.assertFalse(manifest["sources"]["minute_collector_receipt_verified"])
            for name in ARCHITECTURES:
                artifact, loaded_manifest = load_paper_artifact(output, name)
                self.assertEqual(loaded_manifest, manifest)
                self.assertEqual(artifact["market"], "domestic")
                self.assertEqual(artifact["minute_adaptation_identities"],
                                 frozenset({("KRX", "005930")}))
            missing_identity = folder / "missing_identity_bundle"
            missing_identity.mkdir()
            altered = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            altered["study"].pop("minute_adaptation_identities")
            (missing_identity / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks valid minute adaptation"):
                load_paper_artifact(missing_identity, "linear")
            with self.assertRaises(FileExistsError):
                run(args)
            receipt = minute.with_suffix(minute.suffix + ".receipt.json")
            receipt.write_text(json.dumps({"source": "kiwoom_rest_demo_minute_chart",
                                           "sha256": "wrong", "market": "domestic",
                                           "interval_minutes": 5}))
            args.output = folder / "second_bundle"
            with self.assertRaisesRegex(ValueError, "receipt does not match"):
                run(args)
            state = output / "linear.pt"
            with state.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_paper_artifact(output, "linear")


if __name__ == "__main__":
    unittest.main()
