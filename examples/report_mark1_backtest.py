"""Plot portfolio simulations without promoting test winners or hiding gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from examples.report_mark1 import LABELS, COLORS, style, number, percent
from examples.train_mark1 import save_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def datetimes(equity):
    return np.asarray([row["date"] for row in equity], dtype="datetime64[D]")


def make_report(run, destination, primary):
    style()
    destination.mkdir(parents=True, exist_ok=True)
    config = read(run / "config.json")
    summaries = [read(run / market / "summary.json") for market in ("domestic", "us")]
    if not all(row.get("completed") is True for row in summaries):
        raise ValueError("Both market backtests must finish before reporting")
    titles = ("국내", "미국")
    portfolios = {(s["market"], item["variant"], mode): read(run / s["market"] / f"{item['variant']}-{mode}.json")
                  for s in summaries for item in s["results"] for mode in ("carry", "eod")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row", layout="constrained")
    for col, (summary, title) in enumerate(zip(summaries, titles)):
        market = summary["market"]
        for i, row in enumerate(summary["results"]):
            result = portfolios[market, row["variant"], primary]
            values = np.asarray([item["equity"] for item in result["equity"]])
            suffix = " ★" if row["variant"] == summary["winner_frozen"] else ""
            if not result["summary"]["fully_observed"]:
                suffix += " †"
            axes[0, col].plot(datetimes(result["equity"]), (values / summary["initial_cash"] - 1) * 100,
                              color=COLORS[i], label=LABELS[row["variant"]] + suffix,
                              linewidth=2 if suffix.startswith(" ★") else 1.2)
        for mode, line, color in (("carry", "-", "#2878b5"), ("eod", "--", "#ea8d24")):
            result = portfolios[market, summary["winner_frozen"], mode]
            values = np.asarray([item["equity"] for item in result["equity"]])
            drawdown = values / np.maximum.accumulate(values) - 1
            mode_label = "익절·손절까지 보유" if mode == "carry" else "당일 종가 청산"
            if not result["summary"]["fully_observed"]:
                mode_label += " † 잠정"
            axes[1, col].plot(datetimes(result["equity"]), drawdown * 100, color=color, linestyle=line,
                              label=mode_label)
        axes[0, col].axhline(0, color="#777777", linestyle=":", linewidth=1)
        axes[0, col].set_title(f"{title} · 모델별 비용 후 자산 수익률")
        axes[1, col].set_title(f"{title} · 사전 선정 MLP의 낙폭")
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("평가 거래일")
            ax.set_ylabel("초기 자산 대비 수익률 (%)" if row == 0 else "직전 최고 자산 대비 낙폭 (%)")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.legend(fontsize=9)
    mode_title = "익절·손절까지 보유" if primary == "carry" else "당일 종가 청산"
    fig.suptitle(f"mark_1 · 정제 승인 평가 기간 전체 · {mode_title} · 왕복 비용 {config['cost_bps']/100:.2f}%\n"
                 "★ 2024년 사전 선정 / † 가격 결측으로 잠정치 / 현금 대기는 수익률 0% / 실제 체결 아님", fontsize=14)
    fig.savefig(destination / "equity-drawdown.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, summary, title in zip(axes, summaries, titles):
        for mode, color, marker in (("carry", "#2878b5", "o"), ("eod", "#ea8d24", "s")):
            rows = sorted([item for item in summary["cost_sensitivity"] if item["exit_mode"] == mode], key=lambda item: item["cost_bps"])
            mode_label = "익절·손절까지 보유" if mode == "carry" else "당일 종가 청산"
            if any(not item["fully_observed"] for item in rows):
                mode_label += " † 잠정"
            ax.plot([item["cost_bps"]/100 for item in rows], [item["total_return"]*100 for item in rows],
                     color=color, marker=marker, label=mode_label)
        ax.axhline(0, linestyle=":", color="#777777")
        ax.set(title=f"{title} · 사전 선정 모델", xlabel="가정한 왕복 비용 (%)", ylabel="포트폴리오 총수익률 (%)")
        ax.legend()
    fig.suptitle("비용 민감도 · 각 비용에서 현금·정수 수량을 다시 계산\n실제 증권사 요율 아님 · † 보유 중 가격 결측으로 잠정치")
    fig.savefig(destination / "cost-sensitivity.png")
    plt.close(fig)

    lines = ["# mark_1 백테스트", "",
             f"주 결과: **{mode_title}**. 기존 가중치·2023년 보정·매수 문턱을 고정하고 재학습하지 않았다. "
             "2024년에 선정된 국내·미국 `mlp_no_price_aug`를 그대로 유지했다. 이번 수익률 순위로 모델을 다시 고르지 않는다.", "",
             "## 확정한 규칙", "",
             "- 입력은 **과거 완료 30일봉 + 당일 가상 매수가**. 실제 시가를 당일 가상 매수가로 놓았다.",
             "- 당일 최종 고가·저가·종가·거래량은 매수 판단에 들어가지 않는다. 학습 때 증강한 5가격을 독립된 실제 체결로 세지 않는다.",
             "- 미보유 상태에서 보정 확률 **50% 초과**일 때만 진입. 평균 진입가 대비 **+1% 익절 / −0.9% 손절**.",
             "- **같은 일봉의 익절·손절 경계가 모두 닿으면 손절 우선**. 이 규칙은 사용자 확인 사항이다.",
             "- 다일 보유의 다음 시가가 이미 경계를 넘은 경우에는 알려진 최초 시점인 시가에 먼저 청산한다. 하락 갭 손실은 −0.9%보다 클 수 있다. 이후 장중 순서가 불명확한 두 경계 충돌은 손절로 처리한다.",
             "- 당일 미도달 시 익절·손절까지 보유하는 경우와 당일 종가 청산하는 경우를 모두 제시한다. 후자는 기존 하루 단위 학습 목표에 더 가까운 별도 전략이다.", "",
             "## 포트폴리오 가정", "",
             f"- 초기 자금: 국내 {config['initial_krw']:,.0f}원 / 미국 {config['initial_usd']:,.0f}달러. 양 시장 별도 계좌이며 환율 변환·합산하지 않는다.",
             f"- 종목당 거래일 시작 자산의 최대 {config['position_fraction']:.0%}, 최대 {config['max_positions']}종목. 정수 주식 수, 현금 범위 내 매수, 공매도/레버리지/추가 매수 없음.",
             f"- 한 종목 주문 수량은 직전 완료 20일 거래량 중앙값의 {config['volume_fraction']:.2%} 이내. 종목 간 우선순위는 알려진 시가 기준 확률 내림차순, 동률은 고정 종목 ID.",
             "- 시가 진입을 모두 처리한 뒤 당일 고가/저가 청산을 처리한다. 오후 매도대금으로 같은 날 시가 매수를 소급하지 않는다. 시가 갭 청산 자금의 같은 시가 재사용은 이상적 동시 체결 가정이다.",
             f"- 기본 왕복 비용 {config['cost_bps']/100:.2f}%, 매수·매도에 절반씩 실제 거래금액에 부과. 0%, 0.1%, 0.2%, 0.4% 비용을 별도 계산했다. 수수료·세금·호가·슬리피지의 정확한 개별 모델이 아닌 가정이다.",
             "- 시가를 본 뒤 정확히 그 시가로 체결되는 이상적 가정이 있다. 경매/주문 지연/호가 단위/미체결/상하한가 체결 제한은 재현하지 않았다. 배당·기업행사·조정주가에 따른 현금/수량 변화도 별도로 재구성하지 않았다.",
             "- 기간 마지막 날에는 종가로 평가 종료 청산한다. 거래 없는 날도 자산 곡선에 포함한다. Sharpe는 무위험수익 0, 연 252거래일 가정이다.", "",
             "![자산 추이와 낙폭](equity-drawdown.png)", "", "![거래비용 민감도](cost-sensitivity.png)", ""]
    compact = {"config": config, "primary": primary, "markets": []}
    for summary, title in zip(summaries, titles):
        compact["markets"].append({key: value for key, value in summary.items() if key != "price_panel"})
        dates = summary["range"]
        lines += [f"## {title}", "", f"기간 **{dates['first']}~{dates['last']}**, {dates['sessions']}거래일. "
                  f"학습과 별개인 승인 평가 표본 **{summary['test_samples']:,}개** 전체를 사용했다. 이전 6만 개 평가는 그 부분집합이라 이번 결과는 새로운 독립 홀드아웃은 아니다.", "",
                  "| 모델 | 보유 방식 | 거래 수 | 비용 후 승률 | 총수익률 | 최대 낙폭 | Sharpe | 평균 익스포저 | 불확실 거래 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for item in summary["results"]:
            name = LABELS[item["variant"]] + (" ★" if item["variant"] == summary["winner_frozen"] else "")
            for mode in ("carry", "eod"):
                row = item["portfolios"][mode]
                lines.append(f"| {name} | {'익절·손절' if mode == 'carry' else '당일 종가'} | {row['trade_count']:,} | {percent(row['win_rate'])} | {percent(row['total_return'])} | {percent(row['max_drawdown'])} | {number(row['sharpe'], 2)} | {percent(row['mean_exposure'])} | {row['uncertain_trades']:,} |")
        chosen = next(item for item in summary["results"] if item["variant"] == summary["winner_frozen"])
        row = chosen["portfolios"][primary]
        lines += ["", f"사전 선정 모델의 주 결과: 최종 자산 **{row['final_equity']:,.2f} {summary['currency']}**, "
                  f"순손익 **{row['net_pnl']:+,.2f} {summary['currency']}**, 비용 {row['fees']:,.2f} {summary['currency']}. "
                  f"확률 조건 통과 {chosen['raw_signals']:,}건 중 실제 포트폴리오 거래는 {row['trade_count']:,}건이다.", "",
                  f"전체 매수 신호의 **같은 날 학습 사건 적중률**은 {percent(chosen['classification']['precision'])}다. "
                  "위 표의 **비용 후 거래 승률**은 다른 지표다. 특히 다일 보유 결과는 모델의 하루 단위 확률 보정 대상이 아니다.", "",
                  f"양쪽 경계 충돌 후 손절: {row['both_touch_stop_count']:,}건. "
                  f"불확실한 보유 경로: {row['uncertain_trades']:,}건 / 가격 결측 평가일 {row['gap_calendar_days']:,}일. "
                  + ("**가격 결측이 있어 주 결과는 잠정치이며 완전한 백테스트로 해석하면 안 된다.**" if not row["fully_observed"] else "해당 모델의 실제 보유 경로에서는 가격 결측이 없었다."), ""]
        main_portfolio = portfolios[summary["market"], summary["winner_frozen"], primary]
        lines += ["### 연도별 자산 변화", "", "| 연도 | 해당 구간 수익률 |", "|---|---:|"]
        previous = summary["initial_cash"]
        years = sorted({str(np.datetime64(int(point["date"]), "D"))[:4] for point in main_portfolio["equity"][1:]})
        for year in years:
            points = [point for point in main_portfolio["equity"][1:] if str(np.datetime64(int(point["date"]), "D")).startswith(year)]
            final = points[-1]["equity"]
            lines.append(f"| {year} (평가 기간에 포함된 부분) | {percent(final / previous - 1)} |")
            previous = final
        lines += ["", "### 손실이 큰 거래 5건", "", "| 종목 | 진입일 | 청산일 | 진입 → 청산 가격 | 비용 후 거래 수익률 | 청산 사유 |", "|---|---|---|---:|---:|---|"]
        for trade in sorted(main_portfolio["trades"], key=lambda item: item["net_pnl"])[:5]:
            lines.append(f"| {trade['exchange']}:{trade['symbol']} | {trade['entry_date_iso']} | {trade['exit_date_iso']} | {trade['entry_price']:g} → {trade['exit_price']:g} | {percent(trade['net_return'])} | {trade['exit_reason']} |")
        lines += ["", "`GAP_STOP`은 다음 시가가 손절선보다 아래여서 그 시가로 처리한 경우다. 지정 손절률이 실제 손실 상한은 아니다. "
                  "일봉이 모두 같은 가격인 상하한가/단일가 상황에서도 체결된다고 가정하므로 이 가격조차 실제 체결을 보장하지 않는다.", ""]
        for mode in ("carry", "eod"):
            portfolio = portfolios[summary["market"], summary["winner_frozen"], mode]
            save_json(destination / f"{summary['market']}-selected-{mode}-trades.json", portfolio["trades"])
        lines += [f"[선정 모델 보유형 거래내역]({summary['market']}-selected-carry-trades.json) · "
                  f"[선정 모델 당일 청산 거래내역]({summary['market']}-selected-eod-trades.json)", ""]
    lines += ["## 결측과 해석 한계", "",
              "정제 DB는 승인된 학습 구간들의 합집합이다. 신호 다음날부터 가격 기록이 끊길 수 있다. 없는 날짜를 정상 거래일처럼 보간하거나 미래 결측을 미리 알고 전날 매도하지 않았다.", "",
              "보유 중 가격이 없거나 거래량이 0이면 자금은 묶고 마지막 관측 종가로 잠정 평가한다. 다음 관측 시가에 정리하되 해당 거래의 장중 익절/손절 경로는 불확실로 표시한다. 마지막까지 누락이면 마지막 알려진 가격으로 평가 종료할 뿐 실제 청산이라고 부르지 않는다. 이 경우 수익률·낙폭·Sharpe도 잠정치다.", "",
              "최대 낙폭이 작더라도 거래가 적어 대부분 현금이면 전략이 우수하다는 의미는 아니다. 현재 목록의 생존 편향, 학습기간 승인 종목만 사용하는 모집단, 정답일이 관측된 표본만 존재하는 선택 편향이 남아 있다. 이번 결과로 모델을 다시 고르거나 확률 문턱을 바꾸지 않았다.", "",
              "일봉 내부 경로·갭 체결 가정의 일반적 한계는 [TradingView 공식 전략 설명](https://www.tradingview.com/pine-script-docs/concepts/strategies/)과 같이 실제 더 짧은 주기의 가격 및 체결 자료가 있어야 해소할 수 있다.", "",
              "## 재현", "", "```powershell", "& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.backtest_mark1 --output-dir outputs/mark1/backtest-new --device cuda",
              "& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.report_mark1_backtest --run outputs/mark1/backtest-20260916 --destination reports/mark1-backtest-20260916 --primary " + primary,
              "```", "", f"전체 모델의 거래·현금·자산·거절내역과 예측: `{run.resolve()}`. "
              "`summary.json`에 DB·체크포인트·선정 기록 해시와 이전 예측 재현 오차를 보관했다. 원본/정제 DB와 가중치는 읽기만 했으며 주문·자동매매 재시작을 하지 않았다."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(destination / "results.json", compact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("reports/mark1-backtest-20260916"))
    parser.add_argument("--primary", choices=("carry", "eod"), default="carry")
    args = parser.parse_args()
    make_report(args.run, args.destination, args.primary)
