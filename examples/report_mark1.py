"""Render measured Mark_1 results; never select winners from final-test scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {"mlp": "MLP", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN",
          "transformer": "Transformer", "mlp_no_price_aug": "MLP / 증강 없음"}
COLORS = ["#2878b5", "#ea8d24", "#39a275", "#c65373", "#8064b5", "#66737f"]


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def style():
    plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True,
                         "savefig.dpi": 150})


def make_report(run, destination):
    style()
    destination.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((run / market / "summary.json").read_text(encoding="utf-8"))
                 for market in ("domestic", "us")]
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    titles = ("국내", "미국")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="col", layout="constrained")
    for row, (summary, title) in enumerate(zip(summaries, titles)):
        rows = summary["results"]
        names = [LABELS[item["variant"]] + (" ★" if item["variant"] == summary["winner"] else "") for item in rows]
        metrics = ("brier", "precision", "coverage")
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            values = [item["test"][metric] for item in rows]
            scale = 1 if metric == "brier" else 100
            bars = ax.bar(np.arange(len(rows)), [0 if value is None else scale * value for value in values],
                          color=COLORS[:len(rows)], width=.65)
            for bar, value, result in zip(bars, values, rows):
                label = "신호 없음" if value is None else (f"{value:.3f}" if col == 0 else f"{100*value:.3f}%" if col == 2 else f"{100*value:.1f}%")
                label_height = bar.get_height()
                if col == 1 and value is not None:
                    label += f"\nn={result['test']['signal_count']:,}"
                    ci = result["test"]["precision_date_bootstrap"]
                    if ci and ci["upper"] is not None and result["test"]["signal_count"] >= 30 and ci["signal_days"] >= 20:
                        label_height = max(label_height, 100 * ci["upper"])
                ax.annotate(label, (bar.get_x() + bar.get_width()/2, label_height),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
            ax.set_xticks(np.arange(len(rows)), names, rotation=27, ha="right")
            ax.set_xlabel("모델 (★: 2024년 성능으로 미리 선정)")
            ax.set_title(f"{title} · " + ("확률 오차 ↓", "p > 50% 신호의 실제 성공률", "매수 신호 발생률")[col])
            ax.set_ylabel(("Brier score", "성공률 (%)", "평가 표본 중 신호 (%)")[col])
            if col == 0:
                ax.axhline(summary["baseline"]["test"][metric], color="#444444", linestyle="--", label="상수 확률 기준선")
                ax.legend(fontsize=9)
            if col == 1:
                ax.axhline(50, color="#444444", linestyle="--", label="50% 경계")
                for i, result in enumerate(rows):
                    ci = result["test"]["precision_date_bootstrap"]
                    if ci and ci["lower"] is not None and result["test"]["signal_count"] >= 30 and ci["signal_days"] >= 20:
                        ax.vlines(i, ci["lower"]*100, ci["upper"]*100, color="#222222", linewidth=1.2)
                ax.legend(fontsize=9)
            ax.margins(y=.20)
    axes[0, 1].set_ylim(0, 115)
    counts = " / ".join(f"{title} {summary['results'][0]['test']['count']:,}개" for title, summary in zip(titles, summaries))
    fig.suptitle(f"mark_1 · 최종 평가 2025년 이후 / {counts} 실제 시가 표본\n"
                 "오차막대: 30신호·20신호일 이상일 때만 날짜 묶음 95% 구간 · 일봉 사건 / 실제 체결 미검증", fontsize=15)
    fig.savefig(destination / "comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for col, (summary, title) in enumerate(zip(summaries, titles)):
        for i, result in enumerate(summary["results"]):
            history = result["history"]
            axes[0, col].plot([item["epoch"] for item in history], [item["tune_bce"] for item in history],
                              marker="o", color=COLORS[i], label=LABELS[result["variant"]])
            bins = [item for item in result["test"]["reliability_bins"] if item["count"] >= 30]
            axes[1, col].plot([item["mean_probability"] for item in bins], [item["observed_rate"] for item in bins],
                              marker="o", color=COLORS[i], label=LABELS[result["variant"]])
        axes[0, col].set(title=f"{title} · 2022년 검증 손실", xlabel="학습 epoch", ylabel="Binary cross-entropy ↓")
        axes[0, col].legend(fontsize=9)
        axes[1, col].plot([0, 1], [0, 1], "--", color="#444444")
        axes[1, col].set(title=f"{title} · 최종 평가 확률 보정", xlabel="모델이 예측한 성공확률", ylabel="실제 성공 비율",
                         xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axes[1, col].legend(fontsize=8)
    fig.suptitle("학습 추이와 확률 신뢰도 · 보정용 2023년과 최종 평가 기간은 분리\n확률 그림은 동일 폭 10구간 중 표본 30개 이상만 표시", fontsize=14)
    fig.savefig(destination / "learning-calibration.png")
    plt.close(fig)

    lines = ["# mark_1 실험 결과", "", f"{config['device']} · PyTorch {config['torch']} · seed {config['seed']} · 단일 시드 1차 비교", "",
             "**실거래 검증이 아니다.** 결과의 성공은 `당일 고가 ≥ 진입가×1.01 AND 당일 저가 > 진입가×0.991`이다. "
             "동일 일봉에서 양쪽 경계를 건드리면 실패로 처리한다. 장중 진입 이후의 선후관계는 일봉으로 판별할 수 없다.", "",
             "## 이번 실행의 결론", "",
             "아래에서 선정한 모델은 2024년 전체 확률 오차가 가장 작았던 모델이다. 이것은 매수 신호의 실제 성공률 50% 초과나 수익성 검증을 뜻하지 않는다. "
             "50% 초과 구간의 신호 수·적중률·비용 후 대용 손익을 함께 확인해야 한다. 자동매매는 켜지 않았다.", "",
             "![모델별 최종 평가](comparison.png)", "", "![학습과 확률 보정](learning-calibration.png)", "",
             "## 공통 실험 설계", "",
             "- 입력은 완료된 30봉 + 후보 진입가 질문 토큰 1개(총 31개)다. 당일 최종 고가·저가·종가·거래량은 입력하지 않았다.",
             f"- 시장별 원본 학습 사건 상한 {config['max_train_samples']:,}개. 실제 시가의 99%, 99.5%, 100%, 100.5%, 101% 가격으로 5배 질의했다. 독립 관측이 5배 늘어난 것은 아니다. 시장별 실제 선택 건수는 아래에 기재했다.",
             "- 증강 없는 MLP도 같은 원본 시가를 5번 반복해 epoch당 제시 수/최대 업데이트 수를 맞췄다. 조기 종료 시 실제 epoch 수는 다르다.",
             "- 학습 2010–2021 / 조기 종료 검증 2022 / 확률 보정 2023 / 모델 선정 2024 / 최종 평가 2025년 이후.",
             f"- 기간 경계에서 30거래일 입력이 이전 구간에 겹치는 표본을 제거했다. 검증·보정·선정·최종 평가 각각 최대 {config['max_eval_samples']:,}개, 실제 시가만 사용했다.",
             "- 2024년 Brier 오차, 동률이면 log loss로 선정을 기록한 뒤에 최종 평가했다. 2025년 이후 결과가 더 좋아 보이는 모델로 교체하지 않았다.",
             f"- {config['epochs']} epoch 상한, patience {config['patience']}, AdamW lr=0.001, weight decay=0.0001, batch {config['batch_size']}, hidden {config['model_config']['hidden_size']}, dropout {config['model_config']['dropout']}. 클래스 가중치·oversampling은 사용하지 않았다.",
             "- 각 모델의 확률은 독립된 2023년 데이터로 양의 기울기 Platt 보정을 했다. 추후 시장 변화에도 보정이 유지된다는 보장은 없다.", "",
             "## 기법 설명", "",
             "| 기법 | 어떻게 보는가 | 비교하려는 점 |", "|---|---|---|",
             "| MLP | 31개 토큰을 펼쳐 비선형 층에 입력 | 복잡한 순서 모델이 정말 필요한지 확인하는 기본선 |",
             "| LSTM | 기억 셀과 게이트로 순서대로 읽음 | 오래된 일봉 정보를 유지하는 효과 |",
             "| GRU | LSTM보다 단순한 게이트로 순서 정보를 압축 | 적은 파라미터와 학습 시간의 효율 |",
             "| TCN | 인과적 1차원 합성곱, dilation 1/2/4/8 | 짧고 긴 구간의 패턴을 병렬로 학습 |",
             "| Transformer | 위치 정보와 attention으로 토큰 관계를 계산 | 멀리 떨어진 일봉 간 관계가 도움이 되는지 |", ""]
    for summary, title in zip(summaries, titles):
        lines += [f"## {title}", "", f"선정 모델: **{LABELS[summary['winner']]}** (2024년 선정 지표 기준).", "",
                  f"정제 승인 입력에서 학습기간 이력이 있는 {summary['dataset']['selected_symbols']:,}종목을 사용했다. "
                  f"학습 원본 {summary['dataset']['selected_counts']['train']:,}개, epoch당 제시 {summary['results'][0]['training_presentations_per_epoch']:,}개, 평가 {summary['dataset']['selected_counts']['test']:,}개다. 여러 종목을 합친 수이며 종목별 수가 아니다.", "",
                  "| 모델 | 파라미터 | 최적 epoch / 수행 | 학습 초 | 선정 Brier ↓ | 최종 Brier ↓ | ROC AUC ↑ | AP ↑ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for item in summary["results"]:
            t = item["test"]
            lines.append(f"| {LABELS[item['variant']]} | {item['parameters']:,} | {item['best_epoch']} / {item['epochs_completed']} | {item['training_seconds']:.1f} | {item['selection']['brier']:.4f} | {t['brier']:.4f} | {number(t['roc_auc'])} | {number(t['pr_auc'])} |")
        baseline = summary["baseline"]
        lines += [f"| 상수 확률 {percent(baseline['probability'])} | 0 | — | — | {baseline['selection']['brier']:.4f} | {baseline['test']['brier']:.4f} | {number(baseline['test']['roc_auc'])} | {number(baseline['test']['pr_auc'])} |", "",
                  f"| 모델 | p > 50% 신호 / {summary['dataset']['selected_counts']['test']:,} | 발생률 | 실제 성공률 | 날짜 묶음 95% 구간 | 비용 {config['cost_bps']:g}bp 후 가상 평균 손익 |", "|---|---:|---:|---:|---:|---:|"]
        for item in summary["results"]:
            t = item["test"]; ci = t["precision_date_bootstrap"]
            interval = "—" if not ci else ("표본 부족" if t["signal_count"] < 30 or ci["signal_days"] < 20
                                           else f"{percent(ci['lower'])}–{percent(ci['upper'])}")
            lines.append(f"| {LABELS[item['variant']]} | {t['signal_count']:,} | {percent(t['coverage'])} | {percent(t['precision'])} | {interval} | {percent(t['net_mean_return'])} |")
        aug, control = next(item for item in summary["results"] if item["variant"] == "mlp"), next(item for item in summary["results"] if item["variant"] == "mlp_no_price_aug")
        delta = aug["test"]["brier"] - control["test"]["brier"]
        lines += ["", f"MLP 가격 증강 효과: 최종 Brier 차이(증강−미증강) **{delta:+.5f}**. 음수면 증강 쪽 오차가 작다. 단일 시드 결과로 보편적인 효과를 단정할 수 없다.", ""]
    lines += ["## 해석과 한계", "",
              "- Brier는 확률 오차이며 작을수록 좋다. AP는 양성 순위 품질이며 기준선은 사건 발생률이다. 대다수의 실패를 모두 실패로 예측해 얻는 높은 정확도만으로 성능을 판단하지 않았다.",
              "- `확률 > 50%`는 모델 추정치에 대한 조건이다. 선택된 신호의 실제 성공률이 반드시 50%를 넘는다는 뜻은 아니다. 정확히 50%는 HOLD다.",
              "- 신호가 없는 경우 적중률과 평균 손익은 계산 불가(—)다. 신호가 나오도록 기준을 사후 조정하지 않았다.",
              "- 신뢰구간은 같은 날짜의 종목들을 묶어 400번 재표집했다. 날짜 간 시계열 의존성·단일 시드 모델 학습 불확실성은 포함하지 않는다.",
              "- 30신호 또는 20신호일 미만은 구간을 표시하지 않았다. 특히 1건 성공/1건 신호의 100%는 신뢰할 만한 100% 성공률을 뜻하지 않는다. 원시 계산값은 JSON에 보존했다.",
              "- 손익은 매 신호를 독립적으로 진입했다고 가정한 OHLC 대용치다. 양쪽 터치는 손절 우선, 미도달은 당일 종가 청산으로 가정했다. 실제 GUI의 손익·포트폴리오 수익률·체결 기록이 아니다.",
              "- 후보 가격이 실제 거래되지 않았을 수 있고 당일 극값이 진입 전에 발생했을 수 있다. 특히 장중 현재가 적용은 실제 시가 평가와 다른 분포이며 재보정/분봉 검증이 필요하다.",
              "- 데이터의 현재 종목목록에 따른 생존 편향, 타깃 봉 관측 가능성, 과거 조정주가 일관성, 저유동성 필터의 대리지표 한계가 남아 있다.",
              "- 기본 현행 GUI는 감시/자동주문 OFF. 이 실험은 원본/정제 DB 수정, 계좌 조회 또는 주문을 수행하지 않았다.", "",
              "## 재현 자료", "", f"원 실행 기록: `{run.resolve()}`. 시장별 `selection_locked.json`, `dataset_manifest.json`, 모델별 `history.json`, `result.json`, `test_predictions.npz` 보관.", "",
              "[전략 및 실행 방법](../../docs/mark1-strategy.md) · [GUI 안내](../../docs/MARK1_GUI.md)", "",
              "확률 보정 해석은 [scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/calibration.html), "
              "봉 기반 체결 순서 가정의 한계는 [TradingView 공식 전략 문서](https://www.tradingview.com/pine-script-docs/concepts/strategies/)를 참고했다."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (destination / "results.json").write_text(json.dumps([{k: v for k, v in summary.items() if k != "dataset"} for summary in summaries],
                                                        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    provenance = {"run_config": config, "markets": {}}
    for summary in summaries:
        market, data = summary["market"], summary["dataset"]
        provenance["markets"][market] = {
            "cache_contract": json.loads((run / market / "cache_contract.json").read_text(encoding="utf-8")),
            "selected_counts": data["selected_counts"], "uncapped_split_counts": data["uncapped_split_counts"],
            "selected_symbols": data["selected_symbols"], "split_dates": data["split_dates"],
            "selection_locked": summary["selection"], "limitations": data["limitations"],
        }
    (destination / "experiment.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("reports/mark1-20260916"))
    args = parser.parse_args()
    make_report(args.run, args.destination)
