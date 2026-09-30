"""Daily-only fitting and actual-minute inference boundaries."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import shutil
import tempfile
import unittest
from zoneinfo import ZoneInfo

from dockdack.market_schedule import session_on
from dockdack.models import Market, MinuteBar
from dockdack.research.daily_proxy_minute import (
    DAILY_PROXY_CONFIGS, _actual_minute_bars, _relative_bar_features,
    _validation_threshold,
    infer_daily_proxy, load_configured_daily_proxy,
    make_daily_proxy_samples, train_daily_proxy,
)
from dockdack.research.minute_transfer import Bar


KST = ZoneInfo("Asia/Seoul")


def daily_stock(symbol: str, *, sessions: int = 110) -> tuple[Bar, ...]:
    rows = []
    cursor = date(2025, 1, 1)
    while len(rows) < sessions:
        session = session_on(Market.DOMESTIC, cursor)
        if session is not None:
            index = len(rows)
            op = 100. + index * .06 + (int(symbol) % 11) * .5
            cl = op * (1 + .002 * ((index % 5) - 2))
            rows.append(Bar("domestic", symbol, session.closed,
                            op, max(op, cl) * 1.002,
                            min(op, cl) * .998, cl,
                            100_000. + (index % 7) * 500, 1440, "KRX"))
        cursor += timedelta(days=1)
    return tuple(rows)


def minute_window(symbol: str = "005930", count: int = 5):
    base = datetime(2025, 12, 1, 9, 10, tzinfo=KST)
    return tuple(MinuteBar(
        Market.DOMESTIC, symbol, "KRX", base + timedelta(minutes=5 * i),
        Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"),
        Decimal("1000"), "KRW") for i in range(count))


class DailyProxyMinuteTests(unittest.TestCase):
    def test_feature_shape_is_price_unit_invariant(self):
        import numpy as np
        original = daily_stock("005930", sessions=8)
        scaled = tuple(replace(bar, open=bar.open * 100, high=bar.high * 100,
                               low=bar.low * 100, close=bar.close * 100)
                       for bar in original)
        np.testing.assert_allclose(_relative_bar_features(original),
                                   _relative_bar_features(scaled), atol=1e-5)

    def test_daily_proxy_uses_third_later_open_without_minute_retraining(self):
        rows = daily_stock("005930", sessions=30)
        samples = make_daily_proxy_samples(
            rows, lookback=5, horizon=2,
            fee_bps_per_side=2, slippage_bps_per_side=8)
        self.assertTrue(samples)
        first = samples[0]
        self.assertEqual(first.decision_at, rows[4].timestamp)
        self.assertEqual(first.entry_session, rows[7].session_date)
        self.assertEqual(first.exit_session, rows[9].session_date)
        self.assertLess(first.decision_at, first.entry_at)
        self.assertLess(first.entry_at, first.exit_at)

    def test_daily_proxy_rejects_minute_training_rows(self):
        minute = _actual_minute_bars(minute_window())
        with self.assertRaisesRegex(ValueError, "daily"):
            make_daily_proxy_samples(minute, lookback=5, horizon=2,
                                     fee_bps_per_side=2, slippage_bps_per_side=8)

    def test_validation_threshold_requires_nonconstant_scores_and_days(self):
        rows = make_daily_proxy_samples(
            daily_stock("005930", sessions=35), lookback=5, horizon=2,
            fee_bps_per_side=2, slippage_bps_per_side=8)
        import numpy as np
        self.assertIsNone(_validation_threshold(rows, np.ones(len(rows)) * .5,
                                                 min_candidates=2, min_candidate_days=2))
        threshold = _validation_threshold(rows, np.linspace(.1, .9, len(rows)),
                                          min_candidates=2, min_candidate_days=2)
        self.assertIsNotNone(threshold)

    def test_train_only_daily_then_infer_actual_minute(self):
        bars = tuple(bar for k in range(10) for bar in daily_stock(f"{k+1:06d}"))
        artifacts, summary = train_daily_proxy(
            bars, lookback=5, horizon=2, epochs=1, max_train_samples=600,
            min_samples_per_split=20, min_training_symbols=10,
            min_validation_candidates=2, min_candidate_days=2,
            purge_sessions=1)
        self.assertEqual(set(artifacts), {"linear", "conv", "gru"})
        self.assertFalse(summary["minute_performance_tested"])
        self.assertEqual(summary["train_domain"],
                         "completed_daily_bars_as_generic_ordered_OHLCV_tokens_only")
        art = artifacts["linear"]
        window = minute_window("000001", count=7)
        outcome = infer_daily_proxy(
            art, window, architecture="linear",
            as_of=datetime(2025, 12, 1, 9, 46, tzinfo=KST))
        self.assertEqual(outcome["max_hold_bars"], 2)
        self.assertEqual(outcome["earliest_order_at"], "2025-12-01T09:45:00+09:00")
        self.assertEqual(outcome["signal_bar_label"], "2025-12-01T09:30:00+09:00")
        self.assertTrue(outcome["decision_ready_now"])
        self.assertTrue(outcome["symbol_in_daily_training_universe"])
        self.assertFalse(outcome["minute_performance_tested"])
        changed_last_two = window[:-2] + tuple(replace(
            bar, open=Decimal("120"), high=Decimal("121"),
            low=Decimal("119"), close=Decimal("120")) for bar in window[-2:])
        unchanged = infer_daily_proxy(
            art, changed_last_two, architecture="linear",
            as_of=datetime(2025, 12, 1, 9, 46, tzinfo=KST))
        self.assertEqual(outcome["probability_proxy"], unchanged["probability_proxy"])
        other = infer_daily_proxy(
            art, minute_window("999999", count=7), architecture="linear",
            as_of=datetime(2025, 12, 1, 9, 46, tzinfo=KST))
        self.assertFalse(other["candidate"])
        self.assertFalse(other["symbol_in_daily_training_universe"])
        with self.assertRaisesRegex(ValueError, "exactly"):
            infer_daily_proxy(art, window[:-1], architecture="linear",
                              as_of=datetime(2025, 12, 1, 9, 46, tzinfo=KST))
        with self.assertRaisesRegex(ValueError, "completed"):
            infer_daily_proxy(art, window, architecture="linear",
                              as_of=datetime(2025, 12, 1, 9, 41, tzinfo=KST))
        exact_slot = infer_daily_proxy(
            art, window, architecture="linear",
            as_of=datetime(2025, 12, 1, 9, 45, tzinfo=KST))
        self.assertTrue(exact_slot["decision_ready_now"])

    def test_shipped_model_identity_and_hash(self):
        root = Path(__file__).resolve().parents[1] / "models" / "mark1_minute"
        self.assertEqual(len(DAILY_PROXY_CONFIGS), 9)
        for model_id, config in DAILY_PROXY_CONFIGS.items():
            artifact, manifest = load_configured_daily_proxy(root, model_id)
            self.assertEqual(artifact["horizon"], config.horizon)
            self.assertEqual(manifest["models"][config.architecture]["model_id"], model_id)
            self.assertEqual(len(artifact["training_symbols"]), 40)
            self.assertFalse(manifest["study"]["minute_performance_tested"])

    def test_modified_weight_file_fails_closed(self):
        root = Path(__file__).resolve().parents[1] / "models" / "mark1_minute"
        config = DAILY_PROXY_CONFIGS["mark1-29-prototype"]
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / config.bundle_name
            shutil.copytree(root / config.bundle_name, copied)
            with (copied / "linear.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_configured_daily_proxy(Path(temporary), "mark1-29-prototype")


if __name__ == "__main__":
    unittest.main()
