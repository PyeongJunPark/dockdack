"""Create a small evidence-based report from completed 30-bar training runs."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def percentage(value):
    return "해당 없음" if value is None else f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [(read_json(run / "metrics.json"), read_json(run / "history.json")) for run in args.runs]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 4), squeeze=False)
    lines = ["# 30봉 LSTM 학습 결과", "",
             "정답: 마지막 완료 일봉 종가 대비 다음 유효 관측봉 종가 +1% 이상. "
             "실제 매입가 기준 거래 수익률 백테스트가 아니다.", "",
             "| 시장 | 종목 | 원본 유효 봉 | 학습 윈도 | 검증 윈도 | 테스트 윈도 | 최적 epoch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    metric_lines = ["", "## 최종 테스트 (매수 확률 기준 0.5)", "",
                    "| 시장 | 실제 +1% 비율 | 매수 후보 | 매수 후보 정밀도 | 재현율 | ROC AUC | BCE / 상수 기준 BCE |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summary = []
    for column, ((metrics, history), run) in enumerate(zip(records, args.runs)):
        market, data = metrics["market"], metrics["data"]
        manifest = read_json(run / "manifest.json")
        selected = [item for item in manifest["symbols"] if item["selected"]]
        raw_count = sum(item["valid_bars"] for item in selected)
        samples = {name: sum(item["splits"][name]["samples"] for item in selected)
                   for name in ("train", "validation", "test")}
        # Report the actual training cap if the run used one.
        samples["train"] = sum(item["training_samples_used"] for item in selected)
        label = "국내" if market == "domestic" else "미국"
        lines.append(f"| {label} | {len(selected):,} | {raw_count:,} | {samples['train']:,} | "
                     f"{samples['validation']:,} | {samples['test']:,} | {metrics['best_epoch']} |")
        test = metrics["test"]
        threshold = test["fixed_thresholds"]["0.5"]
        auc = "해당 없음" if test["roc_auc"] is None else f"{test['roc_auc']:.4f}"
        metric_lines.append(f"| {label} | {percentage(test['positive_rate'])} | "
                            f"{threshold['predicted_buy_samples']:,} | {percentage(threshold['precision'])} | "
                            f"{percentage(threshold['recall'])} | {auc} | "
                            f"{test['bce']:.4f} / {test['no_skill_bce']:.4f} |")
        ax = axes[0, column]
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_bce"] for row in history], marker="o", label="Train")
        ax.plot(epochs, [row["validation"]["bce"] for row in history], marker="o", label="Validation")
        ax.axhline(history[0]["validation"]["no_skill_bce"], color="gray", linestyle="--",
                   label="Validation constant-prior baseline")
        ax.axvline(metrics["best_epoch"], color="green", linestyle=":", label="Selected epoch")
        ax.set(title=f"{market.title()} | {len(selected)} symbols", xlabel="Epoch", ylabel="Binary cross-entropy")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        summary.append({"market": market, "checkpoint": str((run / "best_model.pt").resolve()),
                        "symbols": len(selected), "raw_bars": raw_count, "samples": samples,
                        "best_epoch": metrics["best_epoch"], "completed_epochs": metrics["completed_epochs"],
                        "elapsed_seconds": metrics["elapsed_seconds"], "test": test,
                        "peak_cuda_allocated_bytes": metrics.get("peak_cuda_allocated_bytes"),
                        "device": metrics["run"]["device_name"]})
    fig.tight_layout()
    fig.savefig(args.output_dir / "learning-curves.png", dpi=150)
    plt.close(fig)
    lines.extend(metric_lines)
    lines.extend(["", "정밀도는 매수 후보 중 다음 유효 종가가 +1% 이상이었던 비율이다. "
                  "매수 후보가 0개이면 정밀도는 정의되지 않는다. 상수 기준은 학습 구간의 양성 비율을 "
                  "모든 샘플에 똑같이 예측한 결과이며, 테스트로 기준을 다시 맞추지 않았다.", "",
                  "익절 +1% / 손절 −0.8%는 실제 보유 원가와 현재가를 비교하는 규칙이다. "
                  "분류 점수가 좋아도 거래비용·장중 가격 경로·체결 지연을 반영한 수익성을 보장하지 않는다.", "",
                  "![학습 곡선](learning-curves.png)", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
