"""Synthetic-only report tests: no actual training or reused-history reads."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "matplotlib"))
if AVAILABLE:
    import numpy as np
    from matplotlib.image import imread
    from examples import report_mark1_selective as report


@unittest.skipUnless(AVAILABLE, "requires optional numpy and matplotlib")
class SelectiveReportTests(unittest.TestCase):
    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @staticmethod
    def metric(signals=60):
        return {"count": 600, "signal_count": signals, "signal_days": min(signals, 30),
                "symbol_count": min(signals, 12), "coverage": signals / 600,
                "precision": .7 if signals else None, "net_mean_return": .0013 if signals else None,
                "overall": {"count": 600, "brier": .17},
                "block_bootstrap": {"precision_lower": .54 if signals >= 50 else None,
                                    "precision_upper": .81 if signals >= 50 else None,
                                    "net_mean_lower": -.0002 if signals >= 50 else None,
                                    "net_mean_upper": .0034 if signals >= 50 else None}}

    def selection(self):
        grid = []
        for index, threshold in enumerate((.5, .55, .6, .65, .7, .75, .8, .85, .9, .95)):
            grid.append({"policy": {"threshold": threshold, "stop_probability_cap": 1.0},
                         "metrics": self.metric(max(0, 120 - index * 20)), "eligible": index < 4})
        return {"chosen_policy": grid[3]["policy"], "chosen_metrics": grid[3]["metrics"],
                "grid": grid, "calibration_qualified": False}

    def result(self, name, seed):
        return {"architecture": name, "seed": seed, "audit": self.metric(),
                "policy_selection": self.selection(), "qualification": {"qualified": False},
                "model": {"trained_iterations": 200, "best_iteration": 50, "fit_seconds": 30.0}}

    def fixture(self, root):
        training_dir, backtest_dir, output = root / "training", root / "backtest", root / "report"
        training, backtests = {}, {}
        for market in report.MARKETS:
            selected = "cat_binary6"
            source = {"database_sha256": ("1" if market == "domestic" else "2") * 64}
            train = {"market": market, "completed": True, "selected": selected,
                     "research_only": True, "deployment_allowed": False, "research_qualified": False,
                     "source": source, "results": {}, "ensembles": {}}
            for fold in report.FOLDS:
                train["results"][fold] = {name: self.result(name, 42) for name in report.NAMES}
                train["ensembles"][fold] = {"seed_results": [self.result(selected, seed) for seed in (42, 43, 44)],
                                            "audit": self.metric(), "policy_selection": self.selection(),
                                            "qualification": {"qualified": False}}
            training[market] = train
            self.write(training_dir / market / "summary.json", train)
            initial = 10_000_000 if market == "domestic" else 10_000
            backtest = {"market": market, "completed": True, "source": source,
                        "research_evaluation": report.EVALUATION_STATUS,
                        "selected_architecture": selected, "frozen_policy": self.selection()["chosen_policy"],
                        "initial_cash": initial, "samples": 600,
                        "range": {"first": "2025-02-01", "last": "2025-03-02", "sessions": 30},
                        "models": {}, "cost_sensitivity": []}
            for name in report.MODELS:
                # No-signal US selective case is deliberate: its flat equity
                # must never become 100% precision or be called profitable.
                signals = 0 if market == "us" and name == "selective" else 60
                metric = self.metric(signals)
                stability = {key: metric[key] for key in ("count", "signal_count", "signal_days", "symbol_count", "precision", "net_mean_return")}
                stability.update(largest_symbol_share=.2 if signals else None,
                                 largest_symbol_id=1 if signals else None, cost_bps=20)
                stability["yearly"] = [{**stability, "period": "2025"}]
                stability["quarterly"] = [{key: value for key, value in stability.items() if key != "yearly"} | {"period": "2025Q1"}]
                model = {"classification": metric, "raw_signals": signals,
                         "portfolios": {}, "stability": stability}
                for mode in report.EXIT_MODES:
                    total_return = 0 if signals == 0 else -.01 if mode == "carry" else -.02
                    portfolio = {"initial_cash": initial, "final_equity": initial * (1 + total_return),
                                 "total_return": total_return, "trade_count": 0 if signals == 0 else 4,
                                 "win_rate": .5 if signals else None, "max_drawdown": total_return,
                                 "cost_bps": 20, "max_positions": 20, "position_fraction": .05,
                                 "volume_fraction": .001, "fully_observed": name != "baseline",
                                 "uncertain_trades": int(name == "baseline")}
                    model["portfolios"][mode] = portfolio
                    first = int(np.datetime64("2025-01-31", "D").astype(np.int64))
                    equity = [{"date": first + index, "equity": initial * (1 + total_return * index / 30)}
                              for index in range(31)]
                    ledger = {"summary": portfolio, "equity": equity, "trades": []}
                    self.write(backtest_dir / market / f"{name}-{mode}.json", ledger)
                    for cost in report.REPORT_COSTS:
                        backtest["cost_sensitivity"].append({"model": name, "exit_mode": mode, "cost_bps": cost,
                                                             "total_return": total_return if signals == 0 else total_return + (20-cost)/10000})
                backtest["models"][name] = model
            backtests[market] = backtest
            self.write(backtest_dir / market / "summary.json", backtest)
        self.write(training_dir / "summary.json", training)
        self.write(backtest_dir / "summary.json", backtests)
        self.write(backtest_dir / "config.json", {**report.REPORT_DEFAULTS,
            "costs_bps": list(report.REPORT_COSTS), "research_evaluation": report.EVALUATION_STATUS})
        return training_dir, backtest_dir, output

    def change(self, path, operation):
        value = json.loads(path.read_text(encoding="utf-8"))
        operation(value)
        self.write(path, value)

    def change_market(self, root, market, operation):
        value = json.loads((root / market / "summary.json").read_text(encoding="utf-8"))
        operation(value)
        self.write(root / market / "summary.json", value)
        self.change(root / "summary.json", lambda all_markets: all_markets.__setitem__(market, value))

    def test_percent_distinguishes_missing_and_zero(self):
        self.assertEqual(report.percent(None), "—")
        self.assertEqual(report.percent(0), "0.00%")
        self.assertEqual(report.percent(.65), "65.00%")
        self.assertEqual(report.policy_text({"threshold": .65, "stop_probability_cap": .25}),
                         "p>65%, 손절확률≤25%")

    def test_complete_fixture_trial_deduplication_and_all_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, _ = self.fixture(Path(temporary))
            train, tested, ledgers = report.load_completed_inputs(training, backtest)
            self.assertEqual(report.trial_totals(train), {"trials": 24, "trained_iterations": 4800,
                                                        "selected_iterations": 1200, "fit_seconds": 720.0})
            self.assertEqual(set(tested["us"]["models"]), set(report.MODELS))
            self.assertEqual(len(ledgers["domestic"]), 4)

    def test_training_completion_precedes_every_backtest_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            self.change_market(training, "us", lambda value: value.__setitem__("completed", False))
            real_read = report.read
            seen = []
            def guarded(path):
                seen.append(Path(path))
                self.assertFalse(Path(path).is_relative_to(backtest))
                return real_read(path)
            with patch.object(report, "read", side_effect=guarded), self.assertRaises(ValueError):
                report.render_report(training, backtest, output)
            self.assertGreater(len(seen), 0)
            self.assertFalse(output.exists())

    def test_partial_mismatched_missing_fold_or_seed_rejected_before_writes(self):
        operations = [lambda value: value.pop("us"),
                      lambda value: value["us"].__setitem__("completed", 1),
                      lambda value: value["us"]["results"].pop("walk_2022"),
                      lambda value: value["us"]["ensembles"]["walk_2024"]["seed_results"].pop()]
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change(training / "summary.json", operation)
                with self.assertRaises(ValueError):
                    report.render_report(training, backtest, output)
                self.assertFalse(output.exists())

    def test_backtest_incomplete_or_wrong_protocol_rejected(self):
        operations = [lambda value: value.__setitem__("completed", False),
                      lambda value: value.__setitem__("research_evaluation", "untouched_test"),
                      lambda value: value.__setitem__("selected_architecture", "lgbm_binary"),
                      lambda value: value["frozen_policy"].__setitem__("threshold", .8),
                      lambda value: value["models"].pop("deep"),
                      lambda value: value["models"]["selective"]["portfolios"].pop("eod"),
                      lambda value: value["cost_sensitivity"].pop(),
                      lambda value: value["source"].__setitem__("database_sha256", "bad"),
                      lambda value: value["models"]["selective"]["stability"]["quarterly"].clear()]
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change_market(backtest, "domestic", operation)
                with self.assertRaises(ValueError):
                    report.render_report(training, backtest, output)
                self.assertFalse(output.exists())

    def test_unsupported_portfolio_assumptions_and_costs_rejected(self):
        cases = {"initial_krw": 2000, "max_positions": 10, "position_fraction": .1,
                 "volume_fraction": .01, "costs_bps": [0, 10, 20],
                 "research_evaluation": "future_unseen"}
        for key, value in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change(backtest / "config.json", lambda payload: payload.__setitem__(key, value))
                with self.assertRaises(ValueError):
                    report.render_report(training, backtest, output)
                self.assertFalse(output.exists())

    def test_ledger_mismatch_and_invalid_equity_refused(self):
        operations = [lambda value: value["summary"].__setitem__("total_return", .1),
                      lambda value: value["equity"].clear(),
                      lambda value: value["equity"][0].__setitem__("equity", 1),
                      lambda value: value["equity"][1].__setitem__("date", value["equity"][0]["date"])]
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.change(backtest / "domestic/selective-carry.json", operation)
                with self.assertRaises(ValueError):
                    report.render_report(training, backtest, output)
                self.assertFalse(output.exists())

    def test_no_signal_precision_not_fabricated_and_ci_floor_preserved(self):
        metric = self.metric(0)
        report._metrics(metric)
        metric["precision"] = 1
        with self.assertRaises(ValueError):
            report._metrics(metric)
        metric = self.metric(10)
        metric["block_bootstrap"].update(precision_lower=.9, precision_upper=1)
        with self.assertRaises(ValueError):
            report._metrics(metric)

    def test_document_reports_both_modes_costs_no_signal_and_nonindependence(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, _ = self.fixture(Path(temporary))
            train, tested, _ = report.load_completed_inputs(training, backtest)
            markdown = report.markdown_report(train, tested, report.trial_totals(train))
            for expected in ("24회 학습", "재사용 역사 평가", "미통과", "당일 청산 -2.00%", "보유형 -1.00%",
                             "무신호", "성공률은 정의되지", "65%는 관측 성공률의 사전 연구 목표", "신호 수는 체결 가능한",
                             "2021 상반기", "2023 하반기", "가장 큰 한 종목", "2025Q1", "40bp"):
                self.assertIn(expected, markdown)
            self.assertIn("완전히 독립된 최종 시험이 아니다", markdown)
            self.assertIn("실제 시가 표본", markdown)

    def test_unrestricted_policy_lead_does_not_claim_abstention_improvement(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, _ = self.fixture(Path(temporary))
            train, tested, _ = report.load_completed_inputs(training, backtest)
            tested["domestic"]["frozen_policy"] = {"threshold": .5, "stop_probability_cap": 1.0}
            tested["domestic"]["models"]["deep"]["classification"] = self.metric(30)
            markdown = report.markdown_report(train, tested, report.trial_totals(train))
            self.assertIn("신호 30 → 60개(**증가**)", markdown)
            self.assertIn("무선별/선별 신호가 동일", markdown)
            self.assertIn("희소화의 개선 효과를 입증한 결과가 아니다", markdown)
            self.assertIn("두 방식 모두 여전히 손실", markdown)
            self.assertIn("미국 시장은 무신호", markdown)
            self.assertNotIn("미국는", markdown)
            tested["domestic"]["models"]["selective"]["classification"]["precision"] = .8
            with self.assertRaises(ValueError):
                report.markdown_report(train, tested, report.trial_totals(train))

    def diagnostic_fixture(self, training):
        source = training / "domestic/summary.json"
        summary = json.loads(source.read_text(encoding="utf-8"))
        diagnostic = {"market": "domestic", "selected_architecture": summary["selected"],
                      "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "frozen_policy_changed": False}
        for fold in report.FOLDS:
            rows = []
            for item in summary["ensembles"][fold]["policy_selection"]["grid"][:4]:
                metric = item["metrics"]
                rows.append({"threshold": item["policy"]["threshold"], "signals": metric["signal_count"],
                             "signal_days": metric["signal_days"], "symbols": metric["symbol_count"],
                             "precision": metric["precision"], "net_mean_return": metric["net_mean_return"],
                             "precision_lower": metric["block_bootstrap"]["precision_lower"],
                             "eligible": item["eligible"]})
            diagnostic[fold] = {"confidence_only_comparison": rows, "stop_cap_0_25_maximum_signals": 3}
        self.write(training / "domestic/development-diagnostic.json", diagnostic)
        return diagnostic

    def test_optional_diagnostic_is_verified_and_rendered_with_audit_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, _ = self.fixture(Path(temporary))
            self.diagnostic_fixture(training)
            train, tested, _ = report.load_completed_inputs(training, backtest)
            audit = {"input_files_unchanged": True, "ledger_count": 64, "trade_rows_checked": 1234,
                     "markets": {market: {"samples": 600} for market in report.MARKETS},
                     "accounting_and_selection_checks_passed": True, "passed": False,
                     "reproducibility_note": {"us_deep_max_probability_difference": .0008,
                         "previous_batch_size": 2048, "current_batch_size": 8192, "deep_precision": "BF16",
                         "all_prior_comparator_strict_0_5_signal_masks_equal": True}}
            self.write(backtest / "independent-ledger-audit.json", audit)
            diagnostics = report.load_optional_diagnostics(training, backtest, train, tested)
            markdown = report.markdown_report(train, tested, report.trial_totals(train), diagnostics)
            self.assertIn("왜 더 높은 문턱", markdown)
            self.assertIn("2021 하반기 | p>65%", markdown)
            self.assertIn("최대 3신호", markdown)
            self.assertIn("거래 행 1,234개", markdown)
            self.assertIn("전체 감사 통과로 표기하지 않는다", markdown)
            self.assertIn("0.00080000", markdown)
            self.assertEqual(markdown.count("| 정책 검증 기간 | 문턱 |"), 2)
            diagnostics["independent"]["passed"] = True
            diagnostics["independent"]["reproducibility_note"].update(
                original_batch_full_reproduction_exact=True, new_batch_aligned_probe_reproduction_exact=True)
            diagnostics["independent"]["resolved_findings"] = [{"full_original_reproduced_rows": 600,
                                                               "new_batch_probe_reproduced_rows": 100}]
            resolved = report.markdown_report(train, tested, report.trial_totals(train), diagnostics)
            self.assertNotIn("전체 감사 통과로 표기하지 않는다", resolved)
            self.assertIn("전체 600표본은 이전 확률을 정확히 재현", resolved)
            self.assertIn("정렬된 100표본은 새 확률을 정확히 재현", resolved)
            self.assertIn("비트 단위 확률이 같다는 보장이 아니다", resolved)

    def test_stale_or_mismatched_optional_diagnostic_refused(self):
        for operation in (lambda value: value.__setitem__("summary_sha256", "wrong"),
                          lambda value: value["walk_2024"]["confidence_only_comparison"][0].__setitem__("signals", 999)):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                training, backtest, output = self.fixture(Path(temporary))
                self.diagnostic_fixture(training)
                self.change(training / "domestic/development-diagnostic.json", operation)
                with self.assertRaises(ValueError):
                    report.render_report(training, backtest, output)
                self.assertFalse(output.exists())

    def test_zero_signal_plot_has_nonnegative_coverage_and_inset_audit_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            train, _, _ = report.load_completed_inputs(training, backtest)
            for fold in report.FOLDS:
                for row in train["us"]["ensembles"][fold]["policy_selection"]["grid"]:
                    row["metrics"] = self.metric(0)
                train["us"]["ensembles"][fold]["policy_selection"]["chosen_metrics"] = self.metric(0)
                for candidate in train["us"]["results"][fold].values():
                    candidate["audit"] = self.metric(0)
            figures = []
            with patch.object(report, "_save", side_effect=lambda fig, *args: figures.append(fig)):
                report._policy_plot(train, output)
                report._audit_plot(train, output)
            try:
                for axis in figures[0].axes:
                    self.assertGreaterEqual(axis.get_xlim()[0], 0)
                for axis in (figures[0].axes[1], figures[0].axes[3]):
                    self.assertIn("실패율도 정의되지 않음", "\n".join(text.get_text() for text in axis.texts))
                for axis in figures[1].axes:
                    self.assertLess(axis.get_xlim()[0], 0)
                    self.assertGreater(axis.get_xlim()[1], 3)
            finally:
                for fig in figures:
                    report.plt.close(fig)

    def test_synthetic_render_produces_four_nonblank_figures_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, backtest, output = self.fixture(Path(temporary))
            inputs = {path: path.read_bytes() for root in (training, backtest) for path in root.rglob("*.json")}
            result = report.render_report(training, backtest, output)
            self.assertEqual(len(result["figures"]), 4)
            for filename in result["figures"]:
                array = imread(filename)
                self.assertGreater(array.shape[0], 500)
                self.assertGreater(float(array[..., :3].std()), .02)
            self.assertTrue(Path(result["report"]).is_file())
            saved = json.loads(Path(result["results"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["trial_totals"]["trials"], 24)
            self.assertIsNone(saved["backtests"]["us"]["models"]["selective"]["classification"]["precision"])
            for path, value in inputs.items():
                self.assertEqual(path.read_bytes(), value)


if __name__ == "__main__":
    unittest.main()
