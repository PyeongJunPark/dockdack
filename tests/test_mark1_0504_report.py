"""Report rendering uses synthetic summaries only; no training or evaluation."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples import report_mark1_0504 as report


def synthetic_report_data():
    """Schema-only fixture, explicitly not reported financial experiment output."""
    data = {"training": {}, "backtest": {}, "labels": {}, "splits": {},
            "protocol": {"protocol": {}}, "bundle": {"version": "synthetic-test"},
            "validation": {"markets": {key: {"prediction_calls": 12} for key in report.MARKETS}},
            "backtest_config": {"initial_krw": 10_000_000, "initial_usd": 10_000,
                                "max_positions": 20, "position_fraction": .05, "volume_fraction": .001},
            "paths": {"training": Path("synthetic/train"), "backtest": Path("synthetic/test"),
                      "bundle": Path("synthetic/bundle")}}
    for market in report.MARKETS:
        gate = {"qualified": False, "reasons": ["insufficient_signal_count"]}
        data["training"][market] = {"research_qualified": False,
            "experiment_data": {"raw_samples": 1000, "eligible_samples": 900, "eligible_symbol_count": 20},
            "ensembles": {fold: {"members": [{"best_iteration": 10} for _ in range(3)],
                "qualification": gate, "validation_qualification": gate} for fold in ("walk_2022", "walk_2024")}}
        data["splits"][market] = {fold: {part: {"first": "2024-01-01", "last": "2024-12-31", "count": 100}
            for part in ("train", "tune", "probability_calibration", "policy_calibration", "audit")}
            for fold in ("walk_2022", "walk_2024")}
        data["labels"][market] = {"walk_2024": {"audit": {"samples": 100, "old_success_rate_same_rows": .3,
            "new_success_rate": .15, "old_both_touch_rate": .4, "new_both_touch_rate": .65}}}
        data["backtest"][market] = {"samples": 100, "range": {"first": "2025-02-17", "last": "2026-09-15", "sessions": 390},
            "models": {}, "cost_sensitivity": []}
        for model in report.MODELS:
            result = {"raw_signals": 0, "classification": {"signal_days": 0, "symbol_count": 0, "precision": None,
                "net_mean_return": None, "block_bootstrap": {"precision_lower": None, "precision_upper": None}},
                "qualification": gate, "portfolios": {}}
            for mode in report.MODES:
                result["portfolios"][mode] = {"trade_count": 0, "total_return": 0., "max_drawdown": 0., "uncertain_trades": 0}
                for cost in report.COSTS:
                    data["backtest"][market]["cost_sensitivity"].append({"model": model, "exit_mode": mode,
                                                                       "cost_bps": cost, "total_return": 0.})
            data["backtest"][market]["models"][model] = result
    return data


class SmallBarrierReportTests(unittest.TestCase):
    def test_zero_signals_remain_unavailable_precision_not_success(self):
        rendered = report.report_text(synthetic_report_data())
        self.assertIn("신호가 0개면 성공률은 0%나 100%가 아니라 산출 불가", rendered)
        self.assertIn("산출 불가", rendered)
        self.assertIn("12개 학습 모델", rendered)
        self.assertIn("24개 예측 비교", rendered)
        self.assertIn("가격 변형 증강은 하지 않았다", rendered)
        self.assertIn("재사용 역사 데이터", rendered)
        self.assertIn("SONY", rendered)
        self.assertIn("30.00% | 15.00% | 40.00% | 65.00%", rendered)
        self.assertIn("GUI에 자동 적용하지 않았고", rendered)
        self.assertIn("../../docs/MARK1_0504.md", rendered)

    def test_percent_none_zero_negative_and_nonfinite(self):
        self.assertEqual(report.percent(None), "산출 불가")
        self.assertEqual(report.percent(0), "0.00%")
        self.assertEqual(report.percent(-.1234), "-12.34%")
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                report.percent(value)

    def test_report_input_values_not_modified(self):
        data = synthetic_report_data()
        before = copy.deepcopy(data)
        report.report_text(data)
        self.assertEqual(data, before)

    def test_incomplete_market_output_cannot_be_reported(self):
        with patch.object(report, "read_json", return_value={}):
            with self.assertRaisesRegex(ValueError, "Both markets"):
                report.load_inputs(Path("not-run"), Path("not-run"), Path("not-run"))

    def test_figure_synthetic_zero_and_negative_results(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("Optional plotting dependency is absent")
        data = synthetic_report_data()
        data["backtest"]["domestic"]["models"]["old_0109"]["raw_signals"] = 441
        data["backtest"]["domestic"]["models"]["new_0504"]["portfolios"]["eod"]["total_return"] = -.01
        with tempfile.TemporaryDirectory(prefix="mark1-report-test-") as temporary:
            destination = Path(temporary) / "synthetic.png"
            report.comparison_figure(data, destination)
            self.assertGreater(destination.stat().st_size, 10_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
