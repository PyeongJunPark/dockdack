"""Truthful Korean selective-model report from completed, measured artifacts.

No model loading, training, policy search, database access, or broker access.
Complete training is checked before reading reused-history backtest artifacts.
Figures are static scientific summaries; unknown/no-trade precision stays null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MARKETS = {"domestic": "국내", "us": "미국"}
FOLDS = ("walk_2022", "walk_2024")
NAMES = {"cat_binary6": "CatBoost-6", "cat_binary8": "CatBoost-8",
         "cat_joint6": "CatBoost joint-6", "lgbm_binary": "LightGBM"}
MODELS = {"baseline": "기존 MLP", "deep": "이전 심층 앙상블",
          "unfiltered": "새 모델 p>50%", "selective": "새 모델 선별 정책"}
COLORS = {"baseline": "#858585", "deep": "#9A79A8", "unfiltered": "#CE8144", "selective": "#267FA8"}
EXIT_MODES = {"carry": "익절·손절까지 보유", "eod": "당일 청산"}
EVALUATION_STATUS = "reused_historical_evaluation_not_untouched_test"
REPORT_DEFAULTS = {"max_positions": 20, "position_fraction": .05, "volume_fraction": .001,
                   "initial_krw": 10_000_000, "initial_usd": 10_000}
REPORT_COSTS = (0, 10, 20, 40)


def read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Complete JSON artifact required: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def percent(value, digits=2):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _interval(metrics, name="precision"):
    block = metrics["block_bootstrap"]
    lower, upper = block.get(name + "_lower"), block.get(name + "_upper")
    return "증거 부족" if lower is None or upper is None else f"{percent(lower)}–{percent(upper)}"


def policy_text(policy):
    text = f'p>{percent(policy["threshold"], 0)}'
    if policy["stop_probability_cap"] < 1:
        text += f', 손절확률≤{percent(policy["stop_probability_cap"], 0)}'
    return text


def _metrics(metrics, count=None):
    if not isinstance(metrics, dict):
        raise ValueError("Signal metrics are required")
    if (not all(_count(metrics.get(key)) for key in ("count", "signal_count", "signal_days", "symbol_count"))
            or not 0 <= max(metrics["signal_days"], metrics["symbol_count"]) <= metrics["signal_count"] <= metrics["count"]
            or (count is not None and metrics["count"] != count)):
        raise ValueError("Signal metric population counts do not match")
    signals, precision = metrics["signal_count"], metrics.get("precision")
    if ((signals == 0 and precision is not None)
            or (signals and (not _finite(precision) or not 0 <= precision <= 1))):
        raise ValueError("Undefined no-signal precision must not be fabricated")
    expected_coverage = signals / metrics["count"] if metrics["count"] else None
    if metrics.get("coverage") != expected_coverage:
        raise ValueError("Signal coverage does not match count")
    overall = metrics.get("overall")
    if not isinstance(overall, dict) or overall.get("count") != metrics["count"]:
        raise ValueError("Original unmodified overall probability metrics required")
    brier = overall.get("brier")
    if metrics["count"] and (not _finite(brier) or not 0 <= brier <= 1):
        raise ValueError("Finite original-probability Brier required")
    block = metrics.get("block_bootstrap")
    if not isinstance(block, dict):
        raise ValueError("Explicit block interval metadata required")
    for name in ("precision", "net_mean"):
        lower, upper = block.get(name + "_lower"), block.get(name + "_upper")
        if ((lower is None) != (upper is None)
                or (lower is not None and (not _finite(lower) or not _finite(upper) or lower > upper))):
            raise ValueError("Malformed confidence interval")
    if signals < 50 or metrics["signal_days"] < 20 or metrics["symbol_count"] < 10:
        if any(block.get(key) is not None for key in
               ("precision_lower", "precision_upper", "net_mean_lower", "net_mean_upper")):
            raise ValueError("Small-sample confidence interval violates declared support floor")


def _training(summary, market):
    if (summary.get("market") != market or summary.get("completed") is not True
            or summary.get("selected") not in NAMES or summary.get("research_only") is not True
            or summary.get("deployment_allowed") is not False
            or type(summary.get("research_qualified")) is not bool):
        raise ValueError("Both completed research-only training markets required")
    if set(summary.get("results", {})) != set(FOLDS) or set(summary.get("ensembles", {})) != set(FOLDS):
        raise ValueError("Both audit folds are required")
    for fold in FOLDS:
        if set(summary["results"][fold]) != set(NAMES):
            raise ValueError("All four candidate architectures are required")
        for name, result in summary["results"][fold].items():
            if result.get("architecture") != name or result.get("seed") != 42:
                raise ValueError("Candidate architecture/seed mismatch")
            _metrics(result["audit"])
        ensemble = summary["ensembles"][fold]
        seeds = ensemble.get("seed_results", [])
        if (len(seeds) != 3 or {item.get("seed") for item in seeds} != {42, 43, 44}
                or any(item.get("architecture") != summary["selected"] for item in seeds)):
            raise ValueError("Exactly three fixed ensemble seeds required for each fold")
        _metrics(ensemble["audit"])
        selection = ensemble["policy_selection"]
        if not selection.get("grid") or not isinstance(selection.get("chosen_policy"), dict):
            raise ValueError("Frozen ensemble policy and full selection grid required")
        if (type(selection.get("calibration_qualified")) is not bool
                or type(ensemble.get("qualification", {}).get("qualified")) is not bool):
            raise ValueError("Explicit boolean research qualification required")
        for row in selection["grid"]:
            _metrics(row["metrics"])


def load_completed_inputs(training_folder, backtest_folder):
    """Validate complete compatible inputs before creating any report output."""
    training_folder, backtest_folder = Path(training_folder), Path(backtest_folder)
    training = read(training_folder / "summary.json")
    if set(training) != set(MARKETS):
        raise ValueError("Both completed training markets required before backtest access")
    for market in MARKETS:
        _training(training[market], market)
        if training[market] != read(training_folder / market / "summary.json"):
            raise ValueError("Training market copy differs from completed run summary")
    # Do not move any reused-history reads above the complete-training checks.
    backtests = read(backtest_folder / "summary.json")
    config = read(backtest_folder / "config.json")
    if set(backtests) != set(MARKETS):
        raise ValueError("Both completed backtest markets required")
    if config.get("research_evaluation") != EVALUATION_STATUS:
        raise ValueError("Reused historical evaluation must be explicitly identified")
    for key, expected in REPORT_DEFAULTS.items():
        if not _finite(config.get(key)) or config[key] != expected:
            raise ValueError(f"Unsupported nondefault report portfolio setting: {key}")
    costs = config.get("costs_bps")
    if (not isinstance(costs, list) or len(costs) != 4
            or any(not _finite(value) for value in costs) or set(costs) != set(REPORT_COSTS)):
        raise ValueError("All four declared costs are required")
    ledgers = {}
    for market, backtest in backtests.items():
        if (backtest != read(backtest_folder / market / "summary.json")
                or backtest.get("completed") is not True
                or backtest.get("research_evaluation") != EVALUATION_STATUS):
            raise ValueError("Complete matching reused-history market summaries required")
        initial = REPORT_DEFAULTS["initial_krw" if market == "domestic" else "initial_usd"]
        if backtest.get("initial_cash") != initial or not _count(backtest.get("samples")):
            raise ValueError("Unsupported initial capital or missing sample count")
        if (backtest.get("selected_architecture") != training[market]["selected"]
                or backtest.get("frozen_policy") != training[market]["ensembles"]["walk_2024"]["policy_selection"]["chosen_policy"]
                or backtest.get("source", {}).get("database_sha256") != training[market].get("source", {}).get("database_sha256")):
            raise ValueError("Training and backtest source/architecture/policy mismatch")
        models = backtest.get("models")
        if not isinstance(models, dict) or set(models) != set(MODELS):
            raise ValueError("All four baseline/deep/unfiltered/selective comparisons required")
        expected = {(model, mode, cost) for model in MODELS for mode in EXIT_MODES for cost in REPORT_COSTS}
        scenarios = backtest.get("cost_sensitivity")
        if (not isinstance(scenarios, list) or len(scenarios) != len(expected)
                or any(not isinstance(row, dict) or not _finite(row.get("cost_bps")) for row in scenarios)
                or {(row.get("model"), row.get("exit_mode"), row["cost_bps"]) for row in scenarios} != expected
                or any(not _finite(row.get("total_return")) for row in scenarios)):
            raise ValueError("All 32 model/mode/cost scenarios per market required")
        ledgers[market] = {}
        for name, model in models.items():
            _metrics(model["classification"], backtest["samples"])
            if model.get("raw_signals") != model["classification"]["signal_count"]:
                raise ValueError("Raw signal count mismatch")
            stability = model.get("stability")
            if (not isinstance(stability, dict)
                    or any(stability.get(key) != model["classification"][key]
                           for key in ("count", "signal_count", "signal_days", "symbol_count", "precision"))
                    or any(not isinstance(stability.get(key), list) for key in ("yearly", "quarterly"))):
                raise ValueError("Matching frozen-policy period/concentration diagnostics required")
            for key in ("yearly", "quarterly"):
                if (sum(item.get("count", -1) for item in stability[key]) != backtest["samples"]
                        or sum(item.get("signal_count", -1) for item in stability[key]) != stability["signal_count"]):
                    raise ValueError("Period diagnostic counts do not reconcile")
            if set(model.get("portfolios", {})) != set(EXIT_MODES):
                raise ValueError("Both carry and end-of-day outcomes are required")
            ledgers[market][name] = {}
            for mode, portfolio in model["portfolios"].items():
                required = {"initial_cash": initial, "cost_bps": 20,
                            **{key: REPORT_DEFAULTS[key] for key in ("max_positions", "position_fraction", "volume_fraction")}}
                if any(not _finite(portfolio.get(key)) or portfolio[key] != value for key, value in required.items()):
                    raise ValueError("Portfolio summary disagrees with declared report assumptions")
                ledger = read(backtest_folder / market / f"{name}-{mode}.json")
                if ledger.get("summary") != portfolio:
                    raise ValueError("Ledger summary differs from completed market summary")
                equity = ledger.get("equity")
                if (not isinstance(equity, list) or len(equity) < 2
                        or any(type(row.get("date")) is not int or not _finite(row.get("equity"))
                               or row["equity"] <= 0 for row in equity)
                        or any(a["date"] >= b["date"] for a, b in zip(equity, equity[1:]))
                        or equity[0]["equity"] != initial or equity[-1]["equity"] != portfolio.get("final_equity")):
                    raise ValueError("Valid dated initial/final equity anchors are required")
                if (not _count(portfolio.get("trade_count")) or type(portfolio.get("fully_observed")) is not bool
                        or not _finite(portfolio.get("total_return"))
                        or not math.isclose(portfolio["total_return"], portfolio["final_equity"] / initial - 1,
                                            rel_tol=1e-9, abs_tol=1e-12)):
                    raise ValueError("Portfolio returns or observation status are invalid")
                if portfolio["trade_count"] == 0 and (portfolio.get("win_rate") is not None
                        or portfolio["total_return"] != 0 or any(row["equity"] != initial for row in equity)):
                    raise ValueError("No-trade portfolio must not fabricate wins or investment profits")
                ledgers[market][name][mode] = ledger
    return training, backtests, ledgers


def trial_totals(training):
    trials = {}
    for market, summary in training.items():
        for fold in FOLDS:
            for name, result in summary["results"][fold].items():
                trials[(market, fold, name, 42)] = result
            for result in summary["ensembles"][fold]["seed_results"]:
                trials[(market, fold, result["architecture"], result["seed"])] = result
    return {"trials": len(trials),
            "trained_iterations": sum(item["model"]["trained_iterations"] for item in trials.values()),
            "selected_iterations": sum(item["model"]["best_iteration"] for item in trials.values()),
            "fit_seconds": sum(item["model"]["fit_seconds"] for item in trials.values())}


def load_optional_diagnostics(training_folder, backtest_folder, training, backtests):
    """Use diagnostics only after checking their identity against completed data."""
    diagnostics = {"development": {}, "independent": None}
    for market in MARKETS:
        path = Path(training_folder) / market / "development-diagnostic.json"
        if not path.exists():
            continue
        value = read(path)
        summary_path = Path(training_folder) / market / "summary.json"
        if (value.get("market") != market or value.get("selected_architecture") != training[market]["selected"]
                or value.get("summary_sha256") != hashlib.sha256(summary_path.read_bytes()).hexdigest()
                or value.get("frozen_policy_changed") is not False):
            raise ValueError("Development diagnostic does not match unchanged training artifacts")
        for fold in FOLDS:
            for item in value[fold]["confidence_only_comparison"]:
                policy = {"threshold": item["threshold"], "stop_probability_cap": 1.0}
                matches = [row for row in training[market]["ensembles"][fold]["policy_selection"]["grid"]
                           if row["policy"] == policy]
                if len(matches) != 1:
                    raise ValueError("Diagnostic policy missing from original calibration grid")
                original, block = matches[0]["metrics"], matches[0]["metrics"]["block_bootstrap"]
                for key, source in (("signals", "signal_count"), ("signal_days", "signal_days"),
                                    ("symbols", "symbol_count"), ("precision", "precision"),
                                    ("net_mean_return", "net_mean_return")):
                    if item[key] != original[source]:
                        raise ValueError("Diagnostic grid metrics differ from measured selection results")
                if item["precision_lower"] != block["precision_lower"] or item["eligible"] != matches[0]["eligible"]:
                    raise ValueError("Diagnostic support/lower bound differs from original selection results")
        diagnostics["development"][market] = value
    path = Path(backtest_folder) / "independent-ledger-audit.json"
    if path.exists():
        value = read(path)
        if (value.get("input_files_unchanged") is not True
                or not _count(value.get("ledger_count")) or not _count(value.get("trade_rows_checked"))
                or any(value.get("markets", {}).get(market, {}).get("samples") != backtests[market]["samples"]
                       for market in MARKETS)):
            raise ValueError("Independent audit identity/population mismatch")
        diagnostics["independent"] = value
    return diagnostics


def _style():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, output, filename):
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def _audit_plot(training, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for row, fold in enumerate(FOLDS):
            axis = axes[row, column]
            for index, name in enumerate(NAMES):
                metrics = training[market]["results"][fold][name]["audit"]
                precision, block = metrics["precision"], metrics["block_bootstrap"]
                if precision is None:
                    axis.text(index, 14, "무신호\nn=0", ha="center", va="center", fontsize=9)
                    continue
                axis.scatter(index, precision * 100, s=45, color="#267FA8", zorder=3)
                if block["precision_lower"] is not None:
                    middle = (block["precision_lower"] + block["precision_upper"]) * 50
                    width = (block["precision_upper"] - block["precision_lower"]) * 50
                    axis.errorbar(index, middle, yerr=width, fmt="none", capsize=4, color="#267FA8")
                axis.annotate(f'n={metrics["signal_count"]:,}', (index, precision * 100),
                              xytext=(0, -15 if precision > .9 else 10), textcoords="offset points", ha="center", fontsize=9)
            axis.axhline(65, linestyle="--", color="#707070", linewidth=1, label="관측 성공률 연구 목표 65%")
            axis.axhline(57.9, linestyle=":", color="#A0A0A0", linewidth=1, label="구간 하한 기준 57.9%")
            axis.set(title=f"{title} · {fold[-4:]} 다음 연도 감사", ylabel="선택 신호 실제 성공률 (%)", ylim=(0, 105))
            axis.set_xticks(range(len(NAMES)), list(NAMES.values()), fontsize=9)
            axis.set_xlim(-.55, len(NAMES) - .45)
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("정밀도 우선 후보 비교 · 각 모델의 이전 연도에 고정한 정책\n50신호·20신호일·10종목 이상일 때만 95% 블록 구간 표시", fontsize=14)
    _save(fig, output, "audit-comparison.png")


def _policy_plot(training, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for row, fold in enumerate(FOLDS):
            axis = axes[row, column]
            selection = training[market]["ensembles"][fold]["policy_selection"]
            caps = sorted({item["policy"]["stop_probability_cap"] for item in selection["grid"]}, reverse=True)
            for cap, color, marker in zip(caps, ("#267FA8", "#CE8144"), ("o", "s")):
                points = [item for item in selection["grid"] if item["policy"]["stop_probability_cap"] == cap
                          and item["metrics"]["precision"] is not None]
                points.sort(key=lambda item: item["metrics"]["coverage"])
                label = "손절 상한 없음" if cap == 1 else f"손절확률≤{percent(cap, 0)}"
                axis.plot([item["metrics"]["coverage"] * 100 for item in points],
                          [(1 - item["metrics"]["precision"]) * 100 for item in points],
                          color=color, linewidth=1, alpha=.7, label=label)
                for item in points:
                    axis.scatter(item["metrics"]["coverage"] * 100, (1 - item["metrics"]["precision"]) * 100,
                                 edgecolor=color, facecolor=color if item["eligible"] else "none", marker=marker, s=40)
            chosen = selection["chosen_metrics"]
            coverage_max = max(item["metrics"]["coverage"] or 0 for item in selection["grid"]) * 100
            if chosen["precision"] is not None:
                axis.scatter(chosen["coverage"] * 100, (1 - chosen["precision"]) * 100,
                             marker="*", color="#292929", s=140, label="선택 정책 (보장 아님)", zorder=5)
            else:
                axis.text(.5, .42, "선택 정책: 신호 없음\n실패율도 정의되지 않음", ha="center", transform=axis.transAxes)
            axis.set(title=f"{title} · {int(fold[-4:]) - 1} 하반기 정책 선택\n{policy_text(selection['chosen_policy'])}",
                     xlabel="전체 기회 중 신호 비율 (%)", ylabel="선택 신호 실패 비율 (%)", ylim=(-3, 103))
            axis.set_xlim(0, coverage_max * 1.08 if coverage_max else 1)
            if not coverage_max:
                axis.set_xticks([0])
            axis.grid(alpha=.2)
            axis.legend(fontsize=8, loc="best")
    fig.suptitle("신호를 줄일 때 실제 위험이 줄었는가? · 정책 선택 구간만 표시\n빈 표식: 최소 증거 부족 · 미래 성과 보장/2025년 이후 최적화가 아님", fontsize=14)
    _save(fig, output, "policy-risk-coverage.png")


def _equity_plot(backtests, ledgers, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for row, (mode, exit_title) in enumerate(EXIT_MODES.items()):
            axis = axes[row, column]
            for name, label in MODELS.items():
                ledger = ledgers[market][name][mode]
                summary = ledger["summary"]
                values = np.asarray([item["equity"] for item in ledger["equity"]], dtype=float)
                dates = np.asarray([item["date"] for item in ledger["equity"]], dtype="datetime64[D]")
                returns = 100 * (values / summary["initial_cash"] - 1)
                suffix = " · 잠정" if not summary["fully_observed"] else ""
                if summary["trade_count"] == 0:
                    suffix += " · 무거래"
                axis.plot(dates, returns, color=COLORS[name], linestyle="--" if name == "unfiltered" else "-",
                          label=f'{label} {summary["trade_count"]:,}건: {percent(summary["total_return"])}{suffix}')
            axis.axhline(0, color="#AAAAAA", linewidth=.7)
            axis.set(title=f"{title} · {exit_title}", xlabel="거래일", ylabel="누적 계좌 수익률 (%)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8, loc="best")
            axis.tick_params(axis="x", rotation=20)
    fig.suptitle("2025년 이후 재사용 역사 비교 · 왕복 비용 0.2%\n무거래 0%는 수익 신호 성공이 아님 · 누락 경로는 잠정 평가", fontsize=14)
    _save(fig, output, "equity-comparison.png")


def _cost_plot(backtests, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for row, (mode, exit_title) in enumerate(EXIT_MODES.items()):
            axis = axes[row, column]
            for name, label in MODELS.items():
                rows = [item for item in backtests[market]["cost_sensitivity"]
                        if item["model"] == name and item["exit_mode"] == mode]
                rows.sort(key=lambda item: item["cost_bps"])
                axis.plot([item["cost_bps"] for item in rows], [item["total_return"] * 100 for item in rows],
                          color=COLORS[name], marker="o", linestyle="--" if name == "unfiltered" else "-", label=label)
            axis.axhline(0, color="#AAAAAA", linewidth=.7)
            axis.set(title=f"{title} · {exit_title}", xlabel="가정한 왕복 비용 (bp)", ylabel="계좌 수익률 (%)")
            axis.set_xticks(REPORT_COSTS)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("거래 비용 민감도 · 각 비용에서 수량/현금 제약까지 다시 계산\n실제 세금·스프레드·시장충격을 완전히 재현한 비용은 아님", fontsize=14)
    _save(fig, output, "cost-sensitivity.png")


def markdown_report(training, backtests, totals, diagnostics=None):
    diagnostics = diagnostics or {"development": {}, "independent": None}
    lines = ["# mark_1 정밀도 우선 선별 모델 연구", "",
             "신호가 드물어도 실제 매수 신호 성공률을 높이려는 별도 연구다. 원본/정제 DB·기존 모델을 바꾸지 않고 자동주문도 하지 않았다.", "",
             "## 결론", ""]
    if not any(summary["research_qualified"] for summary in training.values()):
        lines += ["**양 시장 모두 사전 연구 기준 미통과다. 신호를 드물게 만들면서 높은 정확도를 확보했다는 목표는 달성하지 못했다.**", ""]
    for market, title in MARKETS.items():
        trained, tested = training[market], backtests[market]
        selected = tested["models"]["selective"]
        raw = tested["models"]["unfiltered"]["classification"]
        metric = selected["classification"]
        status = "통과" if trained["research_qualified"] else "미통과"
        previous = tested["models"]["deep"]
        previous_metric = previous["classification"]
        direction = "증가" if metric["signal_count"] > previous_metric["signal_count"] else "감소" if metric["signal_count"] < previous_metric["signal_count"] else "동일"
        lines.append(f'- {title}: `{trained["selected"]}`, {policy_text(tested["frozen_policy"])}. '
                     f'두 감사 구간과 정책 선택 구간의 사전 연구 기준 **{status}**. '
                     f'재사용 역사에서 이전 심층 모델 성공률 {percent(previous_metric["precision"])} → 새 모델 {percent(metric["precision"])}; '
                     f'신호 {previous_metric["signal_count"]:,} → {metric["signal_count"]:,}개(**{direction}**). '
                     f'선별 정책 보유형 {percent(selected["portfolios"]["carry"]["total_return"])}, '
                     f'당일 청산 {percent(selected["portfolios"]["eod"]["total_return"])}.')
        if metric["signal_count"]:
            lines.append(f'  - 이전 심층 모델 대비 비용20bp 보유형 {percent(previous["portfolios"]["carry"]["total_return"])} → '
                         f'{percent(selected["portfolios"]["carry"]["total_return"])}, 당일 청산 '
                         f'{percent(previous["portfolios"]["eod"]["total_return"])} → {percent(selected["portfolios"]["eod"]["total_return"])}. '
                         + ('두 방식 모두 여전히 손실이다.' if all(selected["portfolios"][mode]["total_return"] < 0 for mode in EXIT_MODES) else '각 청산 방식과 비용을 함께 판단해야 한다.'))
            lines.append(f'  - 전체 기간 성공률의95% 블록 구간은 {_interval(metric)}다. '
                         + "; ".join(f'{row["period"]}년 {row["signal_count"]:,}신호 / {percent(row["precision"])}'
                                     for row in selected["stability"]["yearly"])
                         + '. 기간별 변화는 사후 기술 통계이며 이를 근거로 문턱을 재선택하지 않았다.')
        if tested["frozen_policy"] == {"threshold": .5, "stop_probability_cap": 1.0}:
            if raw != metric:
                raise ValueError("Unrestricted >.5 policy must match the new model's unfiltered metrics")
            lines.append('  - 최종 정책은 추가 제한 없는 p>50%여서 새 모델의 무선별/선별 신호가 동일하다. '
                         '**신호 보류·희소화의 개선 효과를 입증한 결과가 아니다.**')
        if metric["signal_count"] == 0:
            lines.append(f'  - {title} 시장은 무신호다. 성공률은 정의되지 않으며 무거래 수익률 0%를 성공으로 보지 않는다.')
        elif metric["block_bootstrap"]["precision_lower"] is None:
            lines.append(f'  - {title} 선별 결과는 최소 증거량이 부족해 신뢰구간을 보류한다. 소수 거래의 높은 성공률을 검증 성공으로 보지 않는다.')
    lines += ["", "65%는 관측 성공률의 사전 연구 목표이며 달성 보장이나 실거래 승률 약속이 아니다. 연구 통과 여부와 관계없이 자동매매에 배포하지 않았다.", "",
              "2025년 이후 자료는 이전 연구에서 이미 본 **재사용 역사 평가**다. 모델/임계값을 이 결과에 맞춰 다시 고르지 않았다. 2022/2024 감사 또한 구조 선택에 쓰였으므로 완전히 독립된 최종 시험이 아니다.", "",
              "## 학습과 분리", "",
              f'실제 기록: **{totals["trials"]:,}회 학습**, 진행 {totals["trained_iterations"]:,} boosting iteration, '
              f'각 실험의 최적 체크포인트에 남긴 iteration 합 {totals["selected_iterations"]:,}, 순수 fit 시간 합 {totals["fit_seconds"] / 60:.1f}분. '
              '이는 딥러닝 epoch·학습 파라미터 수·전체 경과 시간과 다른 값이다.', "",
              "| 구간 | 모델 학습 | 종료 검증 | 확률 보정 | 정책 선택 | 다음 연도 감사 |",
              "|---|---|---|---|---|---|",
              "| walk_2022 | 2012–2019 | 2020 | 2021 상반기 | 2021 하반기 | 2022 |",
              "| walk_2024 | 2014–2021 | 2022 | 2023 상반기 | 2023 하반기 | 2024 |", "",
              "후속 연도 및 반기 경계에는 30거래일 입력 겹침 제거를 적용했다. 완료된 과거30봉과 실제 당일 시가만 입력한다. 당일 최종 고가/저가/종가/전체 거래량은 입력에서 제외한다. 고가/저가는 장벽 정답, 종가는 미접촉 수익 계산에 쓰이고 거래량은 정제 시 관측/거래 존재 조건에 관여한다. 이번 학습은 실제 시가 표본이며 임의가격 증강을 사용하지 않았다.", "",
              "CatBoost depth6/8 이진분류, joint depth6의 네 장벽 사건 분류, LightGBM을 비교했다. 184개 과거 가격/변동성/봉 형태/거래량 특징을 사용한다. 새로운 깊은 신경망이나 논문 완전 재현이라고 부르지 않는다. 구조를 고른 뒤 두 구간 모두 고정 seed42/43/44 앙상블을 만들었다.", "",
              "정책은 확률>.50/.55/.60/.65/.70/.75/.80/.85/.90/.95 중 선택하며 joint만 손절확률≤25% 조건도 비교한다. p>50%는 기본 최소 조건이다. 성공확률과 손절확률을 독립이라고 가정해 곱하지 않는다.", "",
              "최소50신호·20신호일·10종목, 관측 성공률≥65%, 10거래일/2,000회 블록95% 성공률 하한>57.9%, 왕복20bp 후 평균 proxy 수익 하한>0을 요구한다. 이는 금융 시계열/다중 모델 선택 불확실성을 모두 보정하는 확률 보장이 아니다.", "",
              "## 후보별 다음 연도 감사", "",
              "확률 오차(Brier)는 신호에서 탈락한 확률을 0으로 바꾸지 않은 전체 표본 기준이다. 신호 성공률과 별개다.", "",
              "![후보 비교](audit-comparison.png)", ""]
    for market, title in MARKETS.items():
        for fold in FOLDS:
            lines += [f'### {title} · {fold[-4:]} 감사', "",
                      "| 후보 | 고정 정책 | 신호/날짜/종목 | 실제 성공률 | 95% 구간 | 평균 순수익 proxy | Brier | 선택 iteration |",
                      "|---|---|---:|---:|---|---:|---:|---:|"]
            for name, label in NAMES.items():
                row = training[market]["results"][fold][name]
                metric = row["audit"]
                lines.append(f'| {label} | {policy_text(row["policy_selection"]["chosen_policy"])} | '
                             f'{metric["signal_count"]:,}/{metric["signal_days"]:,}/{metric["symbol_count"]:,} | '
                             f'{percent(metric["precision"])} | {_interval(metric)} | {percent(metric["net_mean_return"], 3)} | '
                             f'{metric["overall"]["brier"]:.5f} | {row["model"]["best_iteration"]:,} |')
            ensemble = training[market]["ensembles"][fold]
            metric = ensemble["audit"]
            lines += ["", f'고정 3-seed 앙상블: {policy_text(ensemble["policy_selection"]["chosen_policy"])}; '
                      f'신호 {metric["signal_count"]:,}, 실제 성공률 {percent(metric["precision"])}, '
                      f'95% 구간 {_interval(metric)}, 평균 순수익 proxy {percent(metric["net_mean_return"], 3)}. '
                      f'정책 선택 구간 {"통과" if ensemble["policy_selection"]["calibration_qualified"] else "미통과"}, '
                      f'다음 연도 감사 {"통과" if ensemble["qualification"]["qualified"] else "미통과"}.', ""]
    lines += ["## 신호 빈도와 위험", "",
              "아래 곡선은 선택된 구조의 앙상블에서 이전 연도 하반기 정책 선택 자료만 사용한다. 높은 확률 문턱이 언제나 실제 오류를 줄인다고 가정하지 않는다. 별표는 최종 선택 정책이며 성공을 보장하는 지점이 아니다.", "",
              "![정책 위험과 신호 빈도](policy-risk-coverage.png)", ""]
    for market, diagnostic in diagnostics["development"].items():
        lines += [f'### {MARKETS[market]}: 왜 더 높은 문턱을 선택하지 못했는가?', "",
                  "아래는 결과에 맞춰 바꾼 문턱이 아니라 원래 정책 선택 구간에 저장된 후보 비교다. 높은 성공률이라도 몇 건에 불과하면 증거 기준을 통과하지 못한다.", ""]
        for fold in FOLDS:
            part = diagnostic[fold]
            lines += ["| 정책 검증 기간 | 문턱 | 성공/신호 | 신호 날짜/종목 | 성공률 |95% 하한 | 평균 순수익 proxy | 충분한 증거 |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"]
            for row in part["confidence_only_comparison"]:
                wins = round(row["signals"] * row["precision"]) if row["precision"] is not None else 0
                lines.append(f'| {int(fold[-4:])-1} 하반기 | p>{percent(row["threshold"], 0)} | '
                             f'{wins}/{row["signals"]:,} | {row["signal_days"]}/{row["symbols"]} | '
                             f'{percent(row["precision"])} | {percent(row["precision_lower"])} | '
                             f'{percent(row["net_mean_return"], 3)} | {"있음" if row["eligible"] else "부족"} |')
            higher = [row for row in part["confidence_only_comparison"] if row["threshold"] > .5 and row["eligible"]]
            explanation = ('더 높은 문턱은 점추정 성공률을 높였지만 정밀도 신뢰구간 하한이 p>50%보다 낮았다.'
                           if higher else '더 높은 문턱은 신호/날짜/종목의 최소 증거량을 채우지 못했다.')
            lines += ["", f'{int(fold[-4:])-1} 하반기: {explanation} '
                      f'손절확률25% 제한을 더하면 최대 {part["stop_cap_0_25_maximum_signals"]}신호만 남았다. '
                      '충분한 증거가 있는 후보의 비용 차감 수익 신뢰구간 하한도 양수가 아니었다.', ""]
    lines += [
              "## 2025년 이후 재사용 역사 비교", "",
              "네 모델/정책은 시장별 동일한 원본 표본을 평가하지만 발생 신호와 거래 집합은 다르다. 새 모델의 무선별/선별 비교는 정확히 같은 확률에 마스크만 달리 적용한다. 신호 수는 체결 가능한 실제 매매 수가 아니다.", "",
              "| 시장/모델 | 원신호 | 실제 성공률 | 95% 구간 | 신호 비율 | 전체 Brier |",
              "|---|---:|---:|---|---:|---:|"]
    for market, title in MARKETS.items():
        for name, label in MODELS.items():
            metric = backtests[market]["models"][name]["classification"]
            lines.append(f'| {title} · {label} | {metric["signal_count"]:,} | {percent(metric["precision"])} | '
                         f'{_interval(metric)} | {percent(metric["coverage"], 4)} | {metric["overall"]["brier"]:.5f} |')
    lines += ["", "![계좌 수익 비교](equity-comparison.png)", "",
              "자본 국내1천만원/미국1만달러, 최대20보유, 종목당5%, 과거20일 거래량 중앙값의0.1% 한도, 정수수량, 왕복20bp 비용 가정이다. 시가 진입에 오후 매도금을 미리 사용할 수 없다. 두 장벽 동시 도달은 손절 우선, 이전 보유의 시가 갭은 관측 시가로 처리한다.", ""]
    for market, title in MARKETS.items():
        result, scope = backtests[market], backtests[market]["range"]
        lines += [f'### {title} 계좌 결과', "",
                  f'{scope["first"]}–{scope["last"]}, {scope["sessions"]:,}거래일, 동일 적격 표본 {result["samples"]:,}개.', "",
                  "| 모델/청산 | 거래 수 | 거래 순수익 승률 | 계좌 수익률 | 최대 낙폭 | 관측 상태 |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, label in MODELS.items():
            for mode, exit_title in EXIT_MODES.items():
                row = result["models"][name]["portfolios"][mode]
                status = "완전 관측" if row["fully_observed"] else f'잠정 · 불확실 거래 {row.get("uncertain_trades", "?")}건'
                if row["trade_count"] == 0:
                    status += " · 무거래"
                lines.append(f'| {label} / {exit_title} | {row["trade_count"]:,} | {percent(row["win_rate"])} | '
                             f'{percent(row["total_return"])} | {percent(row["max_drawdown"])} | {status} |')
        lines += [""]
    lines += ["## 비용 민감도", "", "![비용 민감도](cost-sensitivity.png)", "",
              "| 시장/모델/청산 | 0bp | 10bp | 20bp | 40bp |", "|---|---:|---:|---:|---:|"]
    for market, title in MARKETS.items():
        for name, label in MODELS.items():
            for mode, exit_title in EXIT_MODES.items():
                rows = {item["cost_bps"]: item for item in backtests[market]["cost_sensitivity"]
                        if item["model"] == name and item["exit_mode"] == mode}
                lines.append(f'| {title} / {label} / {exit_title} | ' +
                             " | ".join(percent(rows[cost]["total_return"]) for cost in REPORT_COSTS) + " |")
    lines += ["", "## 기간별 안정성과 종목 집중", "",
              "아래는 사전에 고정한 선별 정책의 사후 기술 통계이며 이를 보고 정책을 다시 선택하지 않았다. 작은 분기의 높은 성공률은 별도의 신뢰구간/유의성 보장이 아니다. 한 종목에 신호가 몰리면 전체 성공률을 일반화하기 어렵다.", "",
              "| 시장/모델 | 신호 종목 수 | 가장 큰 한 종목의 신호 비중 |",
              "|---|---:|---:|"]
    for market, title in MARKETS.items():
        for name, label in MODELS.items():
            stability = backtests[market]["models"][name]["stability"]
            lines.append(f'| {title} / {label} | {stability["symbol_count"]:,} | {percent(stability["largest_symbol_share"])} |')
    for market, title in MARKETS.items():
        stability = backtests[market]["models"]["selective"]["stability"]
        lines += ["", f'### {title} 선별 정책 기간별', "",
                  "| 기간 | 신호 수 | 신호 날짜/종목 | 실제 성공률 | 평균 순수익 proxy | 최대 한 종목 비중 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in stability["yearly"] + stability["quarterly"]:
            lines.append(f'| {row["period"]} | {row["signal_count"]:,} | {row["signal_days"]:,}/{row["symbol_count"]:,} | '
                         f'{percent(row["precision"])} | {percent(row["net_mean_return"], 3)} | '
                         f'{percent(row["largest_symbol_share"])} |')
    audit = diagnostics.get("independent")
    if audit is not None:
        lines += ["", "## 검증 상태", "",
                  f'독립 장부 점검: {audit["ledger_count"]:,}개 비용·모델·청산 조합, 거래 행 {audit["trade_rows_checked"]:,}개. '
                  '동일 거래의 비용별 재계산을 포함한 행 수이며 실제 계좌 거래 수가 아니다.', ""]
        if audit.get("accounting_and_selection_checks_passed") is True:
            lines.append('현금/수량/비용/손익 대사와 선택 마스크·지표 일치 점검은 통과했다.')
        if audit.get("passed") is not True:
            lines.append('비교 모델의 확률 재현성 항목이 남아 있어 **전체 감사 통과로 표기하지 않는다**.')
        note = audit.get("reproducibility_note", {})
        if note.get("us_deep_max_probability_difference") is not None:
            lines.append(f'미국 이전 심층 모델의 확률 최대 차이 {note["us_deep_max_probability_difference"]:.8f}가 기록되었다 '
                         f'(이전 batch {note.get("previous_batch_size")}, 이번 {note.get("current_batch_size")}, {note.get("deep_precision")}). '
                         '배치 크기가 달라도 확률이 완전히 같았다고 주장하지 않는다.')
            if (audit.get("passed") is True and note.get("original_batch_full_reproduction_exact") is True
                    and note.get("new_batch_aligned_probe_reproduction_exact") is True):
                findings = audit.get("resolved_findings", [])
                if findings:
                    result = findings[0]
                    lines.append(f'추가 재계산에서 이전 배치의 전체 {result["full_original_reproduced_rows"]:,}표본은 이전 확률을 정확히 재현했고, '
                                 f'새 배치의 정렬된 {result["new_batch_probe_reproduced_rows"]:,}표본은 새 확률을 정확히 재현했다. '
                                 'BF16 배치 크기에 따른 수치 차이로 분리 확인했으며, 원래 차이를 숨기거나 판정 허용오차를 늘리지 않았다.')
                lines.append('회계·선택 검증은 통과했고, 비교 모델의 배치 의존 수치 차이는 추가 재계산으로 확인해 기록했다. '
                             '이는 모든 하드웨어/배치에서 비트 단위 확률이 같다는 보장이 아니다.')
            if note.get("all_prior_comparator_strict_0_5_signal_masks_equal") is True:
                lines.append('이전 비교 모델의 p>50% 매수 마스크는 동일했으며 미국 심층 모델은 양 실행 모두 무신호다.')
    lines += ["", "## 해석의 한계", "",
              "- 일봉 전체의 성공 정답이지 임의 장중 진입 이후 +1%가 −0.9%보다 먼저 도달할 확률이 아니다. 실제 진입 이후 경로에는 분봉/체결 자료가 필요하다.",
              "- 현재 목록의 생존 편향, 수정주가 일관성, 승인/정제 표본 선택, 누락된 보유 가격 경로, 실제 세금·스프레드·시장충격 불확실성이 남는다.",
              "- 양성 신호 정밀도·전체 정확도·개별 거래 순수익 승률·계좌 수익률은 다른 지표다. 최소 증거 없는 높은 성공률이나 거래가 없는0%를 성과로 인정하지 않는다.",
              "- 정책/구조를 고른 자료의 신뢰구간은 선택 이후의 독립 확률 보장이 아니다. 완전히 새로운 미래 구간의 고정 예측과 비용/체결 검증 전에는 배포하지 않는다.", "",
              "[고정 연구 계획](../../docs/mark1-selective-protocol.md) · [관련 논문 검토](../../docs/mark1-selective-literature.md)", ""]
    return "\n".join(lines)


def render_report(training_folder, backtest_folder, output_folder):
    training, backtests, ledgers = load_completed_inputs(training_folder, backtest_folder)
    totals = trial_totals(training)
    diagnostics = load_optional_diagnostics(training_folder, backtest_folder, training, backtests)
    document = markdown_report(training, backtests, totals, diagnostics)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    _audit_plot(training, output)
    _policy_plot(training, output)
    _equity_plot(backtests, ledgers, output)
    _cost_plot(backtests, output)
    (output / "results.json").write_text(json.dumps({"training": training, "backtests": backtests,
        "trial_totals": totals, "diagnostics": diagnostics}, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    (output / "REPORT.md").write_text(document, encoding="utf-8")
    return {"report": str(output / "REPORT.md"), "results": str(output / "results.json"),
            "figures": [str(output / name) for name in ("audit-comparison.png", "policy-risk-coverage.png",
                                                        "equity-comparison.png", "cost-sensitivity.png")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("outputs/mark1/selective-20260916"))
    parser.add_argument("--backtest", type=Path, default=Path("outputs/mark1/selective-backtest-20260916"))
    parser.add_argument("--output", type=Path, default=Path("reports/mark1-selective-20260916"))
    args = parser.parse_args(argv)
    result = render_report(args.training, args.backtest, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
