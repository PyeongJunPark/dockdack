"""Render measured deep research results; no model fitting or broker access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.train_mark1 import save_json


NAMES = {"mlp_deep": "MLP", "resnet18": "ResNet-18", "resnet34": "ResNet-34", "inception": "Inception"}
COLORS = {"mlp_deep": "#507B9B", "resnet18": "#369B75", "resnet34": "#CF8053", "inception": "#9872B0"}
MARKETS = {"domestic": "국내", "us": "미국"}
EVALUATION_STATUS = "reused_historical_evaluation_not_untouched_test"
REPORT_DEFAULTS = {"max_positions": 20, "position_fraction": .05, "volume_fraction": .001,
                   "initial_krw": 10_000_000, "initial_usd": 10_000}
REPORT_COSTS = (0, 10, 20, 40)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value, digits=2):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _required_object(path, description):
    try:
        payload = read(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Complete {description} JSON is required: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def load_completed_inputs(training_folder, backtest_folder):
    """Fail before output writes if run completion/settings cannot support prose.

    The document intentionally describes one fixed portfolio protocol. A
    nondefault run needs a separate report implementation, never silently
    default-labelled results. Complete run summaries must agree exactly with
    both market copies, so partially written market summaries cannot render.
    """
    training_folder, backtest_folder = Path(training_folder), Path(backtest_folder)
    training_all = _required_object(training_folder / "summary.json", "training run summary")
    backtest_all = _required_object(backtest_folder / "summary.json", "backtest run summary")
    for name, payload in (("training", training_all), ("backtest", backtest_all)):
        if set(payload) != set(MARKETS) or any(not isinstance(value, dict) for value in payload.values()):
            raise ValueError(f"Complete {name} summary must contain both domestic and us markets")
    config = _required_object(backtest_folder / "config.json", "backtest configuration")
    for key, expected in REPORT_DEFAULTS.items():
        value = config.get(key)
        if type(value) not in (int, float) or value != expected:
            raise ValueError(f"This report supports only default {key}={expected}; got {value!r}")
    costs = config.get("costs_bps")
    if (not isinstance(costs, list) or len(costs) != len(REPORT_COSTS)
            or any(type(value) not in (int, float) for value in costs)
            or set(costs) != set(REPORT_COSTS)):
        raise ValueError("This report requires all four cost scenarios: 0, 10, 20 and 40 bps")
    if config.get("research_evaluation") != EVALUATION_STATUS:
        raise ValueError("Backtest configuration must identify reused historical evaluation")

    training, backtests = {}, {}
    for market in MARKETS:
        train = _required_object(training_folder / market / "summary.json", f"{market} training summary")
        backtest = _required_object(backtest_folder / market / "summary.json", f"{market} backtest summary")
        if training_all[market] != train:
            raise ValueError(f"{market} training summary differs from completed run summary")
        if backtest_all[market] != backtest:
            raise ValueError(f"{market} backtest summary differs from completed run summary")
        if backtest.get("completed") is not True:
            raise ValueError(f"{market} backtest must be completed before reporting")
        if backtest.get("research_evaluation") != EVALUATION_STATUS:
            raise ValueError(f"{market} backtest must identify reused historical evaluation")
        initial = REPORT_DEFAULTS["initial_krw" if market == "domestic" else "initial_usd"]
        if backtest.get("initial_cash") != initial:
            raise ValueError(f"{market} initial cash differs from the default report protocol")
        models = backtest.get("models")
        if not isinstance(models, dict) or set(models) != {"baseline", "deep"}:
            raise ValueError(f"{market} requires both completed baseline and deep comparisons")
        for name, model in models.items():
            portfolios = model.get("portfolios") if isinstance(model, dict) else None
            if not isinstance(portfolios, dict) or set(portfolios) != {"carry", "eod"}:
                raise ValueError(f"{market}/{name} requires carry and eod portfolio results")
            for mode, portfolio in portfolios.items():
                required = {"initial_cash": initial, "cost_bps": 20,
                            **{key: REPORT_DEFAULTS[key] for key in ("max_positions", "position_fraction", "volume_fraction")}}
                if (not isinstance(portfolio, dict)
                        or any(type(portfolio.get(key)) not in (int, float) or portfolio[key] != value
                               for key, value in required.items())):
                    raise ValueError(f"{market}/{name}/{mode} portfolio settings do not match default report protocol")
        scenarios = backtest.get("cost_sensitivity")
        expected_scenarios = {(model, mode, cost) for model in ("baseline", "deep")
                              for mode in ("carry", "eod") for cost in REPORT_COSTS}
        if (not isinstance(scenarios, list) or len(scenarios) != len(expected_scenarios)
                or any(not isinstance(row, dict) or type(row.get("cost_bps")) not in (int, float)
                       or not isinstance(row.get("model"), str) or not isinstance(row.get("exit_mode"), str)
                       for row in scenarios)
                or {(row["model"], row["exit_mode"], row["cost_bps"]) for row in scenarios} != expected_scenarios):
            raise ValueError(f"{market} requires all four costs for both models and exit modes")
        training[market], backtests[market] = train, backtest
    return training, backtests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("outputs/mark1/deep-20260916"))
    parser.add_argument("--backtest", type=Path, default=Path("outputs/mark1/deep-backtest-20260916"))
    parser.add_argument("--output", type=Path, default=Path("reports/mark1-deep-20260916"))
    args = parser.parse_args()
    training, backtests = load_completed_inputs(args.training, args.backtest)
    development = {}
    for market in MARKETS:
        audit_path = args.training / market / "development-comparison-audit.json"
        if audit_path.exists():
            audit = _required_object(audit_path, "development comparison audit")
            if (audit.get("market") != market or audit.get("input_artifacts_unchanged") is not True
                    or audit.get("sample_count") != training[market]["ensemble_selection"]["count"]
                    or audit.get("source") != backtests[market]["source"]):
                raise ValueError("Development comparison audit does not match the frozen experiment")
            development[market] = audit
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    save_json(args.output / "results.json", {"training": training, "backtests": backtests, "development_comparisons": development})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        summary = training[market]
        names = list(NAMES)
        positions = np.arange(len(names))
        for offset, fold in ((-.19, "walk_2022"), (.19, "walk_2024")):
            rows = [summary["results"][fold][name] for name in names]
            axes[0, column].bar(positions + offset, [row["brier_skill"] * 100 for row in rows], .35,
                                label=fold[-4:])
            points = axes[1, column].scatter(positions + offset,
                                    [np.nan if row["selection"]["precision"] is None else row["selection"]["precision"] * 100 for row in rows],
                                    label=fold[-4:], s=45)
            for index, row in enumerate(rows):
                metric = row["selection"]
                interval = metric["block_bootstrap"]
                if interval["precision_lower"] is not None:
                    midpoint = (interval["precision_lower"] + interval["precision_upper"]) * 50
                    half_width = (interval["precision_upper"] - interval["precision_lower"]) * 50
                    axes[1, column].errorbar(index + offset, midpoint, yerr=half_width,
                                             fmt="none", capsize=4, alpha=.7,
                                             color=points.get_facecolor()[0])
                if metric["precision"] is not None:
                    axes[1, column].annotate(f'n={metric["signal_count"]:,}',
                                             (index + offset, metric["precision"] * 100),
                                             xytext=(0, 8 if offset > 0 else -15), textcoords="offset points",
                                             ha="center", fontsize=8)
        axes[0, column].axhline(0, color="gray", linewidth=.8)
        axes[0, column].set(title=f"{title}: 보정기간 평균확률 대비 예측 오차 개선", ylabel="Brier 개선율 (%)")
        axes[1, column].axhline(50, color="gray", linestyle="--", linewidth=1, label="50% 기준")
        axes[1, column].set(title=f"{title}: 50% 초과 신호의 실제 성공률", ylabel="실제 성공률 (%)", ylim=(0, 105))
        for axis in axes[:, column]:
            axis.set_xticks(positions, [NAMES[name] for name in names])
            axis.legend(fontsize=8)
            axis.grid(axis="y", alpha=.2)
    fig.suptitle("깊은 모델 비교 · 2022/2024 개발 검증\n표본이 충분할 때만 95% 블록 구간 표시 · 작은 표본의 높은 성공률에 주의", fontsize=14)
    fig.savefig(args.output / "model-comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for row, fold in enumerate(("walk_2022", "walk_2024")):
            for name in NAMES:
                history = read(args.training / market / fold / f"{name}-42" / "history.json")
                axes[row, column].plot([item["epoch"] for item in history],
                                       [item["tune_binary_loss"] for item in history], label=NAMES[name], color=COLORS[name])
                best = training[market]["results"][fold][name]["best_epoch"]
                axes[row, column].scatter(best, history[best - 1]["tune_binary_loss"], s=30, color=COLORS[name])
            tune_year = int(fold[-4:]) - 2
            axes[row, column].set(title=f"{title} · {fold[-4:]} 평가용 모델 / tune {tune_year}", xlabel="학습 반복 (epoch)", ylabel="시간 검증 BCE · 낮을수록 좋음")
            axes[row, column].legend(fontsize=8)
            axes[row, column].grid(alpha=.2)
    fig.suptitle("오래 학습한 마지막 값 대신 시간 검증 오차가 최소인 지점 선택", fontsize=14)
    fig.savefig(args.output / "learning-curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (market, title) in enumerate(MARKETS.items()):
        for name, label, color in (("baseline", "기존 MLP", "#858585"), ("deep", "새 3-seed 앙상블", "#267FA8")):
            portfolio = read(args.backtest / market / f"{name}-carry.json")
            values = np.asarray([item["equity"] for item in portfolio["equity"]], dtype=float)
            dates = np.asarray([item["date"] for item in portfolio["equity"]], dtype="datetime64[D]")
            initial = portfolio["summary"]["initial_cash"]
            returns = 100 * (values / initial - 1)
            drawdown = 100 * (values / np.maximum.accumulate(values) - 1)
            provisional = " · 잠정" if not portfolio["summary"]["fully_observed"] else ""
            axes[0, column].plot(dates, returns,
                                 label=f'{label} ({portfolio["summary"]["trade_count"]:,}건): {returns[-1]:+.2f}%{provisional}', color=color)
            axes[1, column].plot(dates, drawdown, label=label, color=color)
        axes[0, column].set(title=f"{title} · 익절/손절까지 보유", ylabel="누적 계좌 수익률 (%)")
        axes[1, column].set(ylabel="고점 대비 하락률 (%)", xlabel="거래일")
        for axis in axes[:, column]:
            axis.axhline(0, color="gray", linewidth=.7)
            axis.grid(alpha=.2)
            axis.legend(fontsize=9)
            axis.tick_params(axis="x", rotation=20)
    fig.suptitle("2025년 이후 재사용 역사 비교 · 왕복 비용 0.2%\n실제 체결 보장 없음 · 가격 관측 누락 시 잠정 평가", fontsize=14)
    fig.savefig(args.output / "equity-comparison.png", dpi=160)
    plt.close(fig)

    lines = ["# mark_1 심층 모델 연구 결과", "",
             "원본·정제 DB와 기존 배포 모델을 보존한 별도 연구다. 자동매매를 켜거나 주문하지 않았다.", "",
             "## 결론", ""]
    for market, title in MARKETS.items():
        summary = training[market]
        lines.append(f'- {title}: 두 개발 구간의 평균 Brier 개선율로 `{summary["selected"]}` 선택. '
                     f'사전 연구 기준 {"통과" if summary["research_qualified"] else "미통과"}. '
                     '통과 여부와 무관하게 기존 자동매매에 배포하지 않았다.')
    lines += ["", "2025년 이후 재사용 역사 평가의 실제 결과를 우선 판단한다. 개발 기간의 확률 오차 개선은 매수 신호 성공률이나 순이익 개선과 같지 않다.", ""]
    for market, title in MARKETS.items():
        result = backtests[market]
        old, new = result["models"]["baseline"], result["models"]["deep"]
        before, after = old["classification"], new["classification"]
        scope = result["range"]
        lines.append(f'- {title} 동일 원본 표본 {result["samples"]:,}개 '
                     f'({scope["first"]}–{scope["last"]}, {scope["sessions"]}거래일): '
                     f'원신호 성공률 {percent(before["precision"])} → {percent(after["precision"])} '
                     f'(신호 {before["signal_count"]:,} → {after["signal_count"]:,}개). '
                     f'비용 포함 보유형 수익률 {percent(old["portfolios"]["carry"]["total_return"])} → '
                     f'{percent(new["portfolios"]["carry"]["total_return"])}; '
                     f'당일 청산형 {percent(old["portfolios"]["eod"]["total_return"])} → '
                     f'{percent(new["portfolios"]["eod"]["total_return"])}.')
        if after["signal_count"] == 0:
            lines.append(f'  {title} 새 모델은 신호가 없어 성공률을 정의할 수 없다. 무거래 수익률 0%는 수익성 입증이 아니다.')
        elif before["precision"] is not None and after["precision"] < before["precision"]:
            lines.append(f'  {title} 매수 신호 성공률은 오히려 하락했다. 보유형과 당일 청산형 중 유리한 결과만 골라 개선으로 판단하지 않는다.')
        if not old["portfolios"]["carry"]["fully_observed"]:
            lines.append(f'  {title} 기존 보유형 결과에는 가격 공백으로 인한 잠정 평가가 포함된다. 자세한 관측 상태는 아래 표에 표시했다.')
    trials = [item for summary in training.values() for models in summary["results"].values() for item in models.values()]
    trials += [item for summary in training.values() for item in summary["seed_results"] if item["seed"] != 42]
    if all("seconds" in item and "epochs" in item for item in trials):
        lines += ["", f'실제 완료: {len(trials)}회 학습, 총 {sum(item["epochs"] for item in trials):,} epoch. '
                  f'기록된 학습 epoch 시간 합계 {sum(item["seconds"] for item in trials) / 60:.1f}분 '
                  '(자료 준비·별도 감사·백테스트 시간 제외).']
    if development:
        lines += ["", "## 기존 과신이 줄었는가: 동일 2024 표본 진단", "",
                  "개발 자료의 추가 진단이며 별도의 독립 시험이나 모델 재선정 근거로 쓰지 않았다. 새 앙상블과 동일한 2024 원본 표본 색인으로 기존 모델을 다시 계산했다. 평가 대상 원본은 같지만 각 모델이 고른 매수 신호 집합은 서로 다르다.", "",
                  "| 시장 | 모델 | 신호 | 평균 예측 | 실제 성공 | 큰 하락 갭 신호 비중 | 비용 후 일봉 proxy |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for market, audit in development.items():
            for name, item in audit["models"].items():
                diagnostic, metric = item["signal_diagnostic"], item["metrics"]
                lines.append(f'| {MARKETS[market]} | {"기존 MLP" if name == "original_mlp_no_price_aug" else "새 앙상블"} | '
                             f'{metric["signal_count"]:,} | {percent(diagnostic["mean_prediction"])} | {percent(metric["precision"])} | '
                             f'{percent(diagnostic["fraction_log_gap_below_minus_2pct"])} | {percent(metric["net_mean_return"], 3)} |')
        lines += ["", "큰 하락 갭은 log(시가/전일 종가)<−0.02인 진단 구간이다. 이 구간을 사후에 제외하거나 거래 규칙을 변경하지 않았다. 과신 감소와 비용 후 수익성 입증은 별개다."]
    lines += ["", "## 무엇을 바꿨는가", "",
              "- 과거 30일 + 현재 후보가격 입력은 유지했다. 당일 최종 고가·저가·종가·거래량은 입력하지 않는다.",
              "- 변동성으로 조정한 가격·봉 모양·거래량 등 18개 특징과 별도 후보가격 헤드를 사용했다.",
              "- MLP 215,684개, ResNet-18-inspired 1,040,708개, ResNet-34-inspired 1,886,148개, InceptionTime-inspired 259,396개 파라미터를 비교했다. 파라미터 수는 데이터 수가 아니다.",
              "- 실제 시가 성공 BCE + 0.25×4상태 CE + 0.10×가상가격 성공 BCE. 가상가격은 ±0.25%/±0.5%에서 골랐다. 합성 표본은 독립된 실제 거래가 아니다.",
              "- 시장·분할당 기본 학습 사례 최대 75만 개, tune 최대 10만 개. calibration/selection은 해당 적격 자료 전체. 최대 40 epoch, 최소 10 epoch, 개선 없는 8 epoch 뒤 종료했다.",
              "- 구조를 먼저 선택한 다음 seed 42·43·44의 raw success logit 평균을 별도 연도로 다시 보정했다. 좋은 seed만 고르지 않았다.",
              "- 매수 p>50%, 익절 +1%, 손절 −0.9%, 양쪽 접촉 시 손절 우선은 그대로다.", "",
              "이전 모델 대비 입력·학습자료·손실도 변경했으므로 성능 차이를 층 수만의 효과로 해석할 수 없다. 새 ResNet-18/34 쌍은 동일한 새 조건에서 깊이를 비교한다.", "",
              "## 개발 검증: 모든 구조를 공개", "",
              "| 시장 | 평가연도 | 모델 | 최적/실행 epoch | Brier | 상수 대비 개선 | 신호 | 성공률 | 비용 후 평균 일봉 proxy |", 
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for market, title in MARKETS.items():
        for fold, models in training[market]["results"].items():
            for name, item in models.items():
                metric = item["selection"]
                lines.append(f'| {title} | {fold[-4:]} | {NAMES[name]} | {item["best_epoch"]}/{item["epochs"]} | '
                             f'{metric["brier"]:.5f} | {percent(item["brier_skill"])} | {metric["signal_count"]:,} | '
                             f'{percent(metric["precision"])} | {percent(metric["net_mean_return"], 3)} |')
    lines += ["", "![모델 비교](model-comparison.png)", "", "![학습 곡선](learning-curves.png)", "",
              "Brier는 낮을수록 좋다. 상수 기준은 평가연도 정답 비율이 아니라 이전 calibration 연도 성공 비율로 고정했다. 신호 없음은 성공률 0%가 아니라 정의 불가다. 막대는 자료가 충분한 경우의 95% 블록 재표집 구간이며, 표시되지 않은 작은 표본에 불확실성이 없다는 뜻이 아니다.", "",
              "## 선택 구조의 세 초기값과 앙상블", "",
              "| 시장 | seed | 2024 신호 | 성공률 | Brier |", "|---|---:|---:|---:|---:|"]
    interval_notes = []
    for market, title in MARKETS.items():
        for item in training[market]["seed_results"]:
            metric = item["selection"]
            lines.append(f'| {title} | {item["seed"]} | {metric["signal_count"]:,} | {percent(metric["precision"])} | {metric["brier"]:.5f} |')
        metric = training[market]["ensemble_selection"]
        lines.append(f'| {title} | 앙상블 | {metric["signal_count"]:,} | {percent(metric["precision"])} | {metric["brier"]:.5f} |')
        block = metric["block_bootstrap"]
        interval_notes.append(f'{title} 앙상블 10거래일 블록 재표집 성공률 95% 구간: '
                     f'{percent(block["precision_lower"])}–{percent(block["precision_upper"])}. '
                     f'비용 후 proxy 평균 구간: {percent(block["net_mean_lower"], 3)}–{percent(block["net_mean_upper"], 3)}. '
                     f'신호 발생일 {block["signal_days"]}일. 표본 부족 시 구간은 제시하지 않는다.')
    lines += [""]
    for note in interval_notes:
        lines += [note, ""]
    lines += ["연구 기준: 200신호·50신호일 이상, 블록 재표집 성공률 하한 >50%, 비용 후 proxy 평균 하한 >0. 두 개발 구간과 최종 앙상블 모두 통과해야 한다. 다중 모델 선택 편향까지 제거한 검정이나 실거래 승인은 아니다.", "",
              "## 이미 본 2025+ 자료: 동일 표본·동일 체결 가정 비교", "",
              "이 기간은 이미 이전 연구에서 확인했다. 이번 구조 선택에는 쓰지 않았지만 **새로운 미사용 시험이 아닌 재사용 역사 평가**다.", "",
              "| 시장 | 모델 | 청산 방식 | 원신호 수 / 성공률 | 거래 | 거래 승률 | 계좌 수익률 | 최대 낙폭 | 평균 노출 | 완전 관측 |", 
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for market, title in MARKETS.items():
        for name, model in backtests[market]["models"].items():
            for mode, item in model["portfolios"].items():
                lines.append(f'| {title} | {"기존" if name == "baseline" else "새 앙상블"} | {"익절/손절 보유" if mode == "carry" else "당일 종가 청산"} | '
                             f'{model["classification"]["signal_count"]:,} / {percent(model["classification"]["precision"])} | {item["trade_count"]:,} | '
                             f'{percent(item["win_rate"])} | {percent(item["total_return"])} | {percent(item["max_drawdown"])} | '
                             f'{percent(item["mean_exposure"])} | {item["fully_observed"]} |')
    lines += ["", "![계좌 수익과 낙폭 비교](equity-comparison.png)", "",
              "시가 즉시 진입·최대 20종목·종목당 계좌 5%·과거 20일 중앙 거래량의 0.1% 한도·정수 주식·왕복 비용 20bps. 갭 손절은 다음 관측 시가로 계산하며 손실은 −0.9%를 넘을 수 있다. 같은 날 두 경계가 닿으면 손절 우선이다.", "",
              "원신호 성공률은 p>50%인 모든 적격 표본의 당일 정답 비율이다. 거래 승률은 자금·보유 제한을 거쳐 모의 체결된 거래의 비용 후 이익 비율이므로 서로 다르다. 거래가 없어서 계좌 수익률 0%인 모델은 수익성 개선을 입증한 모델이 아니다.", "",
              "가격 관측 누락·거래량 0은 임의 체결로 보충하지 않는다. 경로 불확실 거래와 마지막 오래된 가격 평가는 잠정치로 표시한다. 가격제한·호가·부분체결·지연·실제 세금은 완전히 모델링하지 않았다. 당일 종가 청산은 보유 지속과 다른 가정이며 사용자 매도 조건을 변경한 것이 아니다.", "",
              "## 비용 민감도", "",
              "| 시장 | 모델 | 청산 | 왕복 비용 | 계좌 수익률 |", "|---|---|---|---:|---:|"]
    for market, title in MARKETS.items():
        for row in backtests[market]["cost_sensitivity"]:
            lines.append(f'| {title} | {row["model"]} | {row["exit_mode"]} | {row["cost_bps"]}bps | {percent(row["total_return"])} |')
    lines += ["", "## 한계와 다음 증거", "",
              "- 일봉 전체 고저가 정답은 임의 장중 진입 이후 순서를 보여주지 않는다. 실제 체결 후 +1% 선도달/−0.9% 회피를 검증하려면 분봉·체결 자료가 필요하다.",
              "- +1%와 −0.9% 두 결과만 있고 왕복 0.2%를 뺀 단순 계산의 손익분기 성공률은 약 57.9%다. 50% 초과 규칙 자체가 순이익을 보장하지 않는다.",
              "- 현재 종목목록 생존 편향, 정제 표본 선택, 기업행동 조정, 누락 일봉 문제가 남는다. 정제 DB를 변경하지 않았다.",
              "- 이번에 개선돼도 이후 미사용 기간의 순방향 모의 검증이 필요하다. 평가 결과를 보고 이익이 날 때까지 같은 시험기간을 반복 튜닝하지 않는다.", "",
              "## 논문과 재현", "",
              "[시계열 ResNet](https://arxiv.org/abs/1611.06455), [InceptionTime](https://arxiv.org/abs/1909.04939), "
              "[확률 보정](https://proceedings.mlr.press/v70/guo17a.html), [금융 LSTM 저자 논문](https://www.iwf.rw.fau.de/files/2015/12/11-2017.pdf), "
              "[백테스트 과적합](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). 구조에서 영감을 얻은 실험이며 정확한 논문 재현이나 금융 우위 보장은 아니다.", "",
              "설정·코드 해시: `outputs/mark1/deep-20260916/protocol.json`. 전체 결과: `results.json`. "
              "각 모델의 best/model/resume 체크포인트와 epoch 기록은 원래 학습 폴더에 보존했다."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
