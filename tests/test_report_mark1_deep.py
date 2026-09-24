"""Synthetic CPU-only report rendering; no real outcome files or brokers read."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "matplotlib"))
if AVAILABLE:
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    from examples import report_mark1_deep as report


@unittest.skipUnless(AVAILABLE, "requires optional ML and plotting dependencies")
class DeepReportTests(unittest.TestCase):
    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def metrics(self, *, no_signals=False):
        return {"count": 500, "brier": .18, "signal_count": 0 if no_signals else 240,
                "precision": None if no_signals else .55,
                "net_mean_return": None if no_signals else -.00055,
                "block_bootstrap": {"precision_lower": None if no_signals else .48,
                                    "precision_upper": None if no_signals else .62,
                                    "net_mean_lower": None if no_signals else -.002,
                                    "net_mean_upper": None if no_signals else .0008,
                                    "signal_days": 0 if no_signals else 60}}

    def fixture(self, root):
        training, backtest, output = root / "training", root / "backtest", root / "report"
        all_training, all_backtests = {}, {}
        for market in ("domestic", "us"):
            # A positive-looking precision estimate deliberately does not pass
            # the fixed research gate: its precision lower bound is only .48.
            summary = {"selected": "resnet18", "research_qualified": False, "results": {},
                       "seed_results": [], "ensemble_selection": self.metrics()}
            for fold in ("walk_2022", "walk_2024"):
                summary["results"][fold] = {}
                for index, name in enumerate(report.NAMES):
                    metric = self.metrics(no_signals=index == 0)
                    summary["results"][fold][name] = {
                        "selection": metric, "brier_skill": -.02 if index == 0 else .015 + index * .002,
                        "epochs": 3, "best_epoch": 2,
                    }
                    history = [{"epoch": 1, "tune_binary_loss": .60},
                               {"epoch": 2, "tune_binary_loss": .57},
                               {"epoch": 3, "tune_binary_loss": .58}]
                    self.write_json(training / market / fold / f"{name}-42" / "history.json", history)
            for seed in (42, 43, 44):
                summary["seed_results"].append({"seed": seed, "selection": self.metrics(no_signals=seed == 44)})
            self.write_json(training / market / "summary.json", summary)
            all_training[market] = summary
            initial = 10_000_000 if market == "domestic" else 10_000
            summary_bt = {"models": {}, "cost_sensitivity": [], "completed": True,
                          "research_evaluation": report.EVALUATION_STATUS,
                          "initial_cash": initial, "samples": 500,
                          "source": {"market": market, "database_sha256": "0" * 64, "version": 2},
                          "range": {"first": "2025-03-03", "last": "2025-03-05", "sessions": 3}}
            for name in ("baseline", "deep"):
                summary_bt["models"][name] = {"classification": self.metrics(),
                                               "raw_signals": 240, "portfolios": {}}
                for mode in ("carry", "eod"):
                    portfolio_summary = {"initial_cash": initial, "final_equity": initial * .995, "trade_count": 2,
                                         "total_return": -.005, "max_drawdown": -.01,
                                         "fully_observed": name != "deep", "win_rate": .5,
                                         "mean_exposure": .08,
                                         "cost_bps": 20, "max_positions": 20, "position_fraction": .05,
                                         "volume_fraction": .001, "uncertain_trades": int(name == "deep")}
                    dates = np.arange(np.datetime64("2025-03-02"), np.datetime64("2025-03-06")).astype(np.int64)
                    # Exact engine schema: integer epoch-day dates plus an
                    # initial equity anchor, not preformatted date strings.
                    equity = [{"date": int(day), "equity": value}
                              for day, value in zip(dates, (initial, initial * 1.005, initial * .99495, initial * .995))]
                    payload = {"summary": portfolio_summary, "equity": equity, "trades": []}
                    self.write_json(backtest / market / f"{name}-{mode}.json", payload)
                    summary_bt["models"][name]["portfolios"][mode] = portfolio_summary
                    for cost in (0, 10, 20, 40):
                        summary_bt["cost_sensitivity"].append({"model": name, "exit_mode": mode,
                                                               "cost_bps": cost, "total_return": -.003 - cost / 10000})
            self.write_json(backtest / market / "summary.json", summary_bt)
            all_backtests[market] = summary_bt
        self.write_json(training / "summary.json", all_training)
        self.write_json(backtest / "summary.json", all_backtests)
        self.write_json(backtest / "config.json", {"max_positions": 20, "position_fraction": .05,
                                                    "volume_fraction": .001, "costs_bps": [0, 10, 20, 40],
                                                    "initial_krw": 10_000_000, "initial_usd": 10_000,
                                                    "research_evaluation": report.EVALUATION_STATUS})
        return training, backtest, output

    def audit_fixture(self, training, backtest, market="domestic"):
        source = json.loads((backtest / market / "summary.json").read_text(encoding="utf-8"))["source"]
        old_metrics, new_metrics = self.metrics(), self.metrics()
        old_metrics.update(signal_count=100, precision=.34, net_mean_return=-.004)
        audit = {"market": market, "input_artifacts_unchanged": True, "sample_count": 500,
                 "source": source, "scope": "synthetic development-only fixture",
                 "frozen_architecture": "resnet18", "models": {
                     "original_mlp_no_price_aug": {
                         "metrics": old_metrics,
                         "signal_diagnostic": {"mean_prediction": .61, "fraction_log_gap_below_minus_2pct": .97}},
                     "frozen_deep_ensemble": {
                         "metrics": new_metrics,
                         "signal_diagnostic": {"mean_prediction": .52, "fraction_log_gap_below_minus_2pct": .25}},
                 }}
        path = training / market / "development-comparison-audit.json"
        self.write_json(path, audit)
        return path, audit

    def change_json(self, path, operation):
        value = json.loads(path.read_text(encoding="utf-8"))
        operation(value)
        self.write_json(path, value)

    def change_backtest_market(self, folder, operation):
        path = folder / "domestic/summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        operation(value)
        self.write_json(path, value)
        self.change_json(folder / "summary.json", lambda all_markets: all_markets.__setitem__("domestic", value))

    def run_main(self, training, backtest, output):
        argv = ["report_mark1_deep", "--training", str(training), "--backtest", str(backtest), "--output", str(output)]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            report.main()

    def test_percent_distinguishes_missing_zero_and_finite_values(self):
        self.assertEqual(report.percent(None), "—")
        self.assertEqual(report.percent(0), "0.00%")
        self.assertEqual(report.percent(.5), "50.00%")
        self.assertEqual(report.percent(-.009, 3), "-0.900%")

    def test_refuses_partial_or_mismatched_global_summaries_before_output_creation(self):
        cases = (
            ("training", "summary.json", lambda value: value.pop("us")),
            ("training", "domestic/summary.json", lambda value: value.__setitem__("selected", "inception")),
            ("backtest", "summary.json", lambda value: value.pop("us")),
            ("backtest", "domestic/summary.json", lambda value: value.__setitem__("samples", 999)),
        )
        for folder_name, filename, operation in cases:
            with self.subTest(folder=folder_name, file=filename), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                folder = training if folder_name == "training" else backtest
                self.change_json(folder / filename, operation)
                with self.assertRaises(ValueError):
                    self.run_main(training, backtest, output)
                self.assertFalse(output.exists())

    def test_refuses_incomplete_wrong_status_and_incomplete_scenario_results(self):
        operations = (
            lambda value: value.__setitem__("completed", False),
            lambda value: value.__setitem__("completed", 1),
            lambda value: value.__setitem__("research_evaluation", "untouched_test"),
            lambda value: value["cost_sensitivity"].pop(),
            lambda value: value["models"]["deep"]["portfolios"].pop("eod"),
            lambda value: value["models"]["deep"]["portfolios"]["carry"].__setitem__("position_fraction", .1),
        )
        for index, operation in enumerate(operations):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change_backtest_market(backtest, operation)
                with self.assertRaises(ValueError):
                    self.run_main(training, backtest, output)
                self.assertFalse(output.exists())

    def test_refuses_nondefault_report_configuration(self):
        cases = {"max_positions": 10, "position_fraction": .1, "volume_fraction": .005,
                 "initial_krw": 1_000_000, "initial_usd": 20_000,
                 "costs_bps": [0, 10, 20], "research_evaluation": "untouched_test"}
        for key, value in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change_json(backtest / "config.json", lambda config: config.__setitem__(key, value))
                with self.assertRaises(ValueError):
                    self.run_main(training, backtest, output)
                self.assertFalse(output.exists())

    def test_guard_failure_preserves_existing_report_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            output.mkdir()
            (output / "REPORT.md").write_text("existing final report", encoding="utf-8")
            self.write_json(output / "results.json", {"existing": "preserve"})
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            self.change_json(backtest / "config.json", lambda config: config.__setitem__("max_positions", 10))
            with self.assertRaisesRegex(ValueError, "default max_positions=20"):
                self.run_main(training, backtest, output)
            after = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(before, after)

    def test_missing_completion_file_raises_clear_value_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Complete training run summary JSON is required"):
                self.run_main(root / "training", root / "backtest", root / "output")
            self.assertFalse((root / "output").exists())

    def test_optional_development_audit_renders_without_requiring_both_markets(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            path, audit = self.audit_fixture(training, backtest)
            self.run_main(training, backtest, output)
            rendered = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("## 기존 과신이 줄었는가: 동일 2024 표본 진단", rendered)
            self.assertIn("| 국내 | 기존 MLP | 100 | 61.00% | 34.00% | 97.00% | -0.400% |", rendered)
            self.assertIn("| 국내 | 새 앙상블 | 240 | 52.00% | 55.00% | 25.00% | -0.055% |", rendered)
            self.assertIn("별도의 독립 시험이나 모델 재선정 근거로 쓰지 않았다", rendered)
            self.assertIn("이 구간을 사후에 제외하거나 거래 규칙을 변경하지 않았다", rendered)
            stored = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["development_comparisons"], {"domestic": audit})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), audit)

    def test_optional_audit_rejects_wrong_market_count_source_or_protection_flag(self):
        changes = (
            ("market", "us"),
            ("sample_count", 501),
            ("source", {"market": "domestic", "database_sha256": "different"}),
            ("input_artifacts_unchanged", False),
            ("input_artifacts_unchanged", 1),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                path, _ = self.audit_fixture(training, backtest)
                self.change_json(path, lambda audit: audit.__setitem__(key, value))
                with self.assertRaisesRegex(ValueError, "Development comparison audit does not match"):
                    self.run_main(training, backtest, output)
                self.assertFalse(output.exists(), "Invalid optional audit must fail before creating outputs")

    def test_optional_audit_failure_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            path, _ = self.audit_fixture(training, backtest)
            self.change_json(path, lambda audit: audit.__setitem__("input_artifacts_unchanged", False))
            output.mkdir()
            (output / "REPORT.md").write_text("previous complete report", encoding="utf-8")
            self.write_json(output / "results.json", {"preserved": True})
            before = {file.name: file.read_bytes() for file in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "Development comparison audit does not match"):
                self.run_main(training, backtest, output)
            self.assertEqual(before, {file.name: file.read_bytes() for file in output.iterdir()})

    def test_synthetic_cpu_render_and_metric_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, backtest, output = self.fixture(root)
            saved_figures = {}
            original_savefig = Figure.savefig

            def capture(figure, destination, *args, **kwargs):
                saved_figures[Path(destination).name] = {
                    "ylabels": [axis.get_ylabel() for axis in figure.axes],
                    "titles": [axis.get_title() for axis in figure.axes],
                    "lines": [[np.asarray(line.get_ydata(), dtype=float).copy() for line in axis.lines]
                              for axis in figure.axes],
                    "texts": [[text.get_text() for text in axis.texts] for axis in figure.axes],
                }
                return original_savefig(figure, destination, *args, **kwargs)

            argv = ["report_mark1_deep", "--training", str(training), "--backtest", str(backtest), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(Figure, "savefig", capture), redirect_stdout(io.StringIO()):
                report.main()
            for filename in ("model-comparison.png", "learning-curves.png", "equity-comparison.png"):
                image = imread(output / filename)
                self.assertGreater(image.shape[0], 500)
                self.assertGreater(image.shape[1], 800)
                self.assertGreater(float(np.ptp(image[..., :3])), .5)
            rendered = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("연구 기준 미통과", rendered)
            self.assertIn("재사용 역사 평가", rendered)
            conclusion = rendered.split("## 무엇을 바꿨는가", 1)[0]
            self.assertIn("국내 동일 원본 표본 500개", conclusion)
            self.assertIn("원신호 성공률 55.00% → 55.00%", conclusion)
            self.assertIn("비용 포함 보유형 수익률 -0.50% → -0.50%; 당일 청산형 -0.50% → -0.50%", conclusion)
            self.assertIn("신호 없음은 성공률 0%가 아니라 정의 불가", rendered)
            self.assertIn("| 국내 | 44 | 0 | — |", rendered)
            seed_section = rendered.split("## 선택 구조의 세 초기값과 앙상블", 1)[1]
            seed_table = seed_section.split("국내 앙상블 10거래일", 1)[0]
            seed_lines = [line for line in seed_table.splitlines() if line.strip()]
            self.assertEqual(len(seed_lines), 10)  # Header, separator and eight consecutive rows.
            self.assertTrue(all(line.startswith("|") and line.endswith("|") for line in seed_lines))
            self.assertIn("원신호 성공률", rendered)
            self.assertIn("거래 승률", rendered)
            # Classification signals (240) must remain distinct from the two
            # portfolio transactions; preserve both facts in the output JSON.
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(result["development_comparisons"], {})
            self.assertEqual(result["backtests"]["domestic"]["models"]["deep"]["classification"]["signal_count"], 240)
            self.assertEqual(result["backtests"]["domestic"]["models"]["deep"]["portfolios"]["carry"]["trade_count"], 2)
            # Figure transforms fractional account returns and drawdown to
            # percent and includes the initial-capital anchor correctly.
            lines = saved_figures["equity-comparison.png"]["lines"]
            np.testing.assert_allclose(lines[0][0], [0., .5, -.505, -.5], atol=1e-10)
            np.testing.assert_allclose(lines[2][0], [0., 0., -1., (9950 / 10050 - 1) * 100], atol=1e-10)
            self.assertEqual(len(saved_figures["learning-curves.png"]["titles"]), 4)
            self.assertEqual(["tune 2020" in value for value in saved_figures["learning-curves.png"]["titles"]],
                             [True, True, False, False])
            self.assertEqual(["tune 2022" in value for value in saved_figures["learning-curves.png"]["titles"]],
                             [False, False, True, True])
            self.assertFalse(report.plt.get_fignums(), "All figures must be closed after rendering")


if __name__ == "__main__":
    unittest.main()
