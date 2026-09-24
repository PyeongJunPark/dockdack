"""Render completed 0.5/0.4 research results; never train or run evaluation.

Inputs are completed training, matched backtest, and portable-export receipts.
The generated chart and Korean report distinguish daily prediction events from
position-limited portfolio trades. Original model/DB/GUI files are read only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"domestic": "국내", "us": "미국"}
MODELS = {"old_0109": "기존 +1% / −0.9%", "new_0504": "신규 +0.5% / −0.4%"}
MODES = {"carry": "목표 미도달 시 보유", "eod": "당일 종가 청산"}
COSTS = (0, 10, 20, 40)
TARGET = "daily_high_ge_entry_0_5pct_and_low_gt_entry_minus_0_4pct_conservative"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def percent(value, decimals=2):
    if value is None:
        return "산출 불가"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Cannot report a non-finite percentage")
    return f"{number * 100:.{decimals}f}%"


def gate(value):
    return "통과" if value.get("qualified") is True else "미통과"


def _bundle_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError("Invalid portable-bundle path")
    return path


def load_inputs(training_dir, backtest_dir, bundle_dir):
    """Read completed artifacts and bind export/backtest to this exact run."""
    training_dir, backtest_dir, bundle_dir = map(Path, (training_dir, backtest_dir, bundle_dir))
    paths = [training_dir / "summary.json", training_dir / "protocol.json",
             training_dir / "completed.json", backtest_dir / "summary.json",
             backtest_dir / "config.json", bundle_dir / "manifest.json",
             bundle_dir / "export-validation.json"]
    train, protocol, completed, backtest, config, bundle, validation = map(read_json, paths)
    if set(train) != set(MARKETS) or set(backtest) != set(MARKETS):
        raise ValueError("Both markets must finish training and backtesting before reporting")
    if (completed.get("no_orders") is not True or bundle.get("completed") is not True
            or validation.get("passed") is not True
            or validation.get("protected_files_unchanged") is not True
            or validation.get("protected_sha256_before") != validation.get("protected_sha256_after")):
        raise ValueError("Missing completed no-order/export validation receipt")
    if (protocol.get("protocol", {}).get("target") != TARGET
            or protocol["protocol"].get("price_augmentation") is not False
            or bundle.get("semantics", {}).get("target") != TARGET
            or bundle.get("feature_count") != 184
            or bundle.get("risk_flags", {}).get("research_only") is not True
            or bundle["risk_flags"].get("deployment_allowed") is not False):
        raise ValueError("Unexpected target, feature, augmentation or deployment contract")
    manifest_hash = digest(bundle_dir / "manifest.json")
    if (validation.get("bundle_manifest_sha256") != manifest_hash
            or (bundle_dir / "manifest.sha256").read_text(encoding="ascii").strip() != manifest_hash
            or bundle.get("provenance", {}).get("summary_sha256") != digest(training_dir / "summary.json")
            or bundle["provenance"].get("protocol_sha256") != digest(training_dir / "protocol.json")):
        raise ValueError("Portable bundle does not match the completed training run")
    splits, labels, trained_members = {}, {}, 0
    for market in MARKETS:
        row, result = train[market], backtest[market]
        train_market, backtest_market = training_dir / market / "summary.json", backtest_dir / market / "summary.json"
        paths.extend((train_market, backtest_market))
        if (row.get("completed") is not True or result.get("completed") is not True
                or row.get("market") != market or result.get("market") != market
                or row.get("target") != TARGET or read_json(train_market) != row
                or read_json(backtest_market) != result
                or result.get("source") != row.get("source")
                or result.get("experiment_data") != row.get("experiment_data")
                or result.get("deployment_allowed") is not False
                or set(result.get("models", {})) != set(MODELS)
                or result.get("artifact_sha256", {}).get(str((training_dir / "summary.json").resolve()))
                   != digest(training_dir / "summary.json")):
            raise ValueError("Completed training/backtest provenance mismatch")
        if set(row.get("ensembles", {})) != {"walk_2022", "walk_2024"}:
            raise ValueError("Expected two chronological training folds")
        splits[market], labels[market] = {}, {}
        for fold, ensemble in row["ensembles"].items():
            if len(ensemble.get("members", [])) != 3:
                raise ValueError("Expected three independently seeded members per fold")
            trained_members += len(ensemble["members"])
            split_path = training_dir / market / fold / "splits.json"
            label_path = training_dir / market / fold / "features" / "label_audit.json"
            paths.extend((split_path, label_path))
            splits[market][fold], labels[market][fold] = read_json(split_path), read_json(label_path)
        keys = [(item["model"], item["exit_mode"], item["cost_bps"]) for item in result["cost_sensitivity"]]
        expected = {(model, mode, cost) for model in MODELS for mode in MODES for cost in COSTS}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Incomplete or duplicate 16-scenario market cost comparison")
        reference = bundle["markets"][market]
        manifest_path = _bundle_file(bundle_dir, reference["path"])
        if digest(manifest_path) != reference["sha256"]:
            raise ValueError("Portable market manifest checksum mismatch")
        paths.append(manifest_path)
        market_bundle = read_json(manifest_path)
        if market_bundle["provenance"]["source_summary_sha256"] != digest(train_market):
            raise ValueError("Portable market summary differs from training")
        for member in market_bundle["members"]:
            for path_key, hash_key in (("path", "sha256"), ("sidecar_path", "sidecar_sha256")):
                file = _bundle_file(bundle_dir, member[path_key])
                if digest(file) != member[hash_key]:
                    raise ValueError("Portable model checksum mismatch")
                paths.append(file)
    if trained_members != 12:
        raise ValueError("Expected twelve trained members across both markets and folds")
    return {"training": train, "protocol": protocol, "completion": completed,
            "backtest": backtest, "backtest_config": config, "bundle": bundle,
            "validation": validation, "splits": splits, "labels": labels,
            "trained_members": trained_members,
            "paths": {"training": training_dir.resolve(), "backtest": backtest_dir.resolve(), "bundle": bundle_dir.resolve()},
            "input_sha256": {str(path.resolve()): digest(path) for path in paths}}


def comparison_figure(data, destination):
    """Compact scientific figure; plotting reads summaries only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    backtest = data["backtest"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    colors, offsets = ("#8392A5", "#007F78"), (-.18, .18)
    x = np.arange(2)
    for i, name in enumerate(MODELS):
        values = [backtest[market]["models"][name]["raw_signals"] for market in MARKETS]
        bars = axes[0].bar(x + offsets[i], values, .34, color=colors[i],
                           label="Old +1.0/-0.9%" if i == 0 else "New +0.5/-0.4%")
        axes[0].bar_label(bars, labels=[f"{value:,}" for value in values], padding=4, fontsize=10)
    axes[0].set_xticks(x, ["Korea", "United States"])
    axes[0].set_ylabel("Daily prediction events (p > 0.50)")
    axes[0].set_title("A. Signal frequency over the same eligible events", loc="left", weight="bold", fontsize=11)
    maximum = max(backtest[market]["models"][name]["raw_signals"] for market in MARKETS for name in MODELS)
    axes[0].set_ylim(0, max(1, maximum) * 1.20)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=.2)
    axes[0].set_axisbelow(True)
    groups = [(market, name) for market in MARKETS for name in MODELS]
    positions = np.arange(len(groups))
    for i, mode in enumerate(MODES):
        values = [backtest[market]["models"][name]["portfolios"][mode]["total_return"] * 100
                  for market, name in groups]
        bars = axes[1].bar(positions + offsets[i], values, .34,
                           color=("#5666B5", "#D68B33")[i], label="Carry until exit" if i == 0 else "End-of-day exit")
        axes[1].bar_label(bars, labels=[f"{value:+.2f}%" for value in values], padding=3, fontsize=8)
    axes[1].set_xticks(positions, ["KR\nOld", "KR\nNew", "US\nOld", "US\nNew"])
    axes[1].set_ylabel("Simulated portfolio net return (%)")
    axes[1].set_title("B. Portfolio result after 20 bps round-trip costs", loc="left", weight="bold", fontsize=11)
    axes[1].axhline(0, color="#444444", linewidth=.8)
    returns = [backtest[market]["models"][name]["portfolios"][mode]["total_return"] * 100
               for market, name in groups for mode in MODES]
    lower, upper = min(0., min(returns)), max(0., max(returns))
    span = max(.5, upper - lower)
    # Bar sticky edges otherwise remove zero-side padding and place zero labels
    # over the panel title when every measured return is non-positive.
    axes[1].set_ylim(lower - span * .20, upper + span * .16)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="best", frameon=False, fontsize=9)
    figure.suptitle("Mark1 target comparison: frozen models, reused 2025+ history", fontsize=15, weight="bold")
    figure.supxlabel("Daily-bar simulation; stop first when both barriers touch. Not executable fills or an untouched test.", fontsize=9)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def report_text(data):
    train, backtest, protocol = data["training"], data["backtest"], data["protocol"]["protocol"]
    passed = [MARKETS[market] for market, row in train.items() if row["research_qualified"]]
    total_calls = sum(value["prediction_calls"] for value in data["validation"]["markets"].values())
    lines = ["# Mark1 +0.5% / −0.4% 재학습 결과", "",
        f"작성: {datetime.now(timezone.utc).isoformat()} (UTC)", "",
        "국내·미국 모델 재학습, 기존 모델과의 같은 평가 사건 비교, 별도 추론 번들 저장을 완료했다. "
        "이 보고서는 완료된 산출물을 읽어 만든 것이며 학습·추론 기준을 변경하지 않았다.", "",
        f"개발 구간 연구 기준 통과 시장: {', '.join(passed) if passed else '없음'}. "
        "모델의 연구 전용·배포 불허 상태를 유지한다. 기존 GUI와 매매 설정은 변경하지 않았고 주문도 실행하지 않았다.", "",
        "![모델별 신호 수와 비용 차감 포트폴리오 수익률](comparison.png)", "",
        "## 실험 정의", "",
        "- 입력: 직전 **완료된 30거래일 OHLCV + 평가일 시가(가상 진입가격)**. 당일 고가·저가·종가·거래량은 입력하지 않는다.",
        "- 184개 특징 중 목표값에 종속적인 과거 장벽 도달률 16개도 +0.5%/−0.4% 기준으로 다시 계산했다.",
        "- 성공: 일봉 고가가 진입가 × 1.005 이상이고, 일봉 저가가 진입가 × 0.996보다 높음. 둘 다 닿으면 손절 우선 실패로 처리한다.",
        "- 매수 연구 신호는 보정된 성공확률이 **50% 초과**일 때만 발생한다. 정확히 50%는 신호가 아니다. 평가 결과를 보고 문턱을 바꾸지 않았다.",
        "- 기존 최종 선택 구조를 고정했다: 국내 CatBoost 4분류(depth 6), 미국 CatBoost 이진분류(depth 8). 새로운 신경망 구조 탐색은 아니다.",
        "- 2개 시장 × 2개 시간순 검증 구간 × 3개 시드(42·43·44) = **12개 학습 모델**. 최종 저장 번들은 마지막 구간의 6개 모델을 사용한다.",
        "- 각 시장에서는 3개 모델의 원시 점수를 평균한 뒤 과거 확률 보정 구간에서 정한 Platt 보정을 적용한다.",
        "- 이번 비교는 이전 선택 모델과 동일하게 실제 시가 표본을 사용했다. **가격 변형 증강은 하지 않았다**. 12개 모델 수는 원시 관측 수나 독립 표본 수를 뜻하지 않는다.", "",
        "## 데이터와 시간 분리", "",
        "FCEL·BNED·BBSI는 가격 단위 불연속이 확인되어 미국 학습·평가에서 종목 전체를 격리했다. 원본 DB를 고친 것이 아니다. "
        "SONY의 가격 단위 이상 경고와 미진단 기업행사·생존편향 위험은 남아 있다.", "",
        "| 시장 | 정제 원시 창 수 | 격리 후 전체 허용 창 수 | 허용 종목 수 | 비교 평가 창 수 | 비교 기간 | 시장 거래일 수 |",
        "|---|---:|---:|---:|---:|---|---:|"]
    for market in MARKETS:
        data_row, evaluation = train[market]["experiment_data"], backtest[market]
        lines.append(f"| {MARKETS[market]} | {data_row['raw_samples']:,} | {data_row['eligible_samples']:,} | "
                     f"{data_row['eligible_symbol_count']:,} | {evaluation['samples']:,} | "
                     f"{evaluation['range']['first']} ~ {evaluation['range']['last']} | {evaluation['range']['sessions']:,} |")
    lines += ["", "한 창은 한 종목·한 기준일의 과거 30봉과 당일 진입가로 구성된다. 위 창들은 여러 종목을 합친 수이며, "
              "서로 겹치는 과거 봉을 사용하므로 독립 표본이 아니다. 모든 후속 연도 및 반기 경계에 30거래일 입력 중복 제거를 적용했다.", "",
              "| 시장·구간 | 학습 기간 / 표본 수 | 조기 종료용 기간 / 표본 수 | 확률 보정 기간 / 표본 수 | 추가 검증 기간 / 표본 수 | 개발 감사 기간 / 표본 수 |",
              "|---|---|---|---|---|---|"]
    for market in MARKETS:
        for fold, split in data["splits"][market].items():
            fields = [f"{MARKETS[market]} · {fold}"]
            for name in ("train", "tune", "probability_calibration", "policy_calibration", "audit"):
                row = split[name]
                fields.append(f"{row['first']}~{row['last']} / {row['count']:,}")
            lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "2025년 이후 기간은 이전 연구에서 이미 확인한 **재사용 역사 데이터**다. 손대지 않은 새로운 최종 테스트라고 해석하면 안 된다. "
              "두 모델을 같은 격리 후 평가 사건에서 비교하지만, 기존 모델은 격리 전 학습했고 새 모델은 격리 후 학습했으므로 차이를 목표폭 변경 하나의 효과라고 단정할 수 없다.", "",
              "## 신호와 백테스트 거래 수 구분", "",
              "다음 신호는 전체 평가 창에서 p>0.5인 사건 수다. 현금·동시 보유·유동성 제한을 적용한 포트폴리오 매매 횟수와 다르다. "
              "성공률도 각 모델 자신의 목표(+1/−0.9 또는 +0.5/−0.4)로 계산하므로 두 성공률은 동일한 라벨의 정확도 비교가 아니다. "
              "거래가 거의 없어 수익률이 0%에 가까워진 결과를 예측력이나 수익성이 개선된 것으로 해석하면 안 된다. 아래 거래는 실제 계좌 체결이 아니다.", "",
              "| 시장 | 모델 | 신호 수 | 신호일 수 | 신호 종목 수 | 신호 성공률 | 성공률 95% 구간 | 신호당 비용 후 평균 | 연구 기준 |",
              "|---|---|---:|---:|---:|---:|---|---:|---|"]
    for market in MARKETS:
        for name in MODELS:
            result = backtest[market]["models"][name]
            metric, bounds = result["classification"], result["classification"]["block_bootstrap"]
            interval = "산출 불가" if bounds["precision_lower"] is None else f"{percent(bounds['precision_lower'])} ~ {percent(bounds['precision_upper'])}"
            lines.append(f"| {MARKETS[market]} | {MODELS[name]} | {result['raw_signals']:,} | {metric['signal_days']:,} | "
                         f"{metric['symbol_count']:,} | {percent(metric['precision'])} | {interval} | "
                         f"{percent(metric['net_mean_return'])} | {gate(result['qualification'])} |")
    lines += ["", "신호가 0개면 성공률은 0%나 100%가 아니라 산출 불가다. 신뢰구간은 10거래일 블록 재표집 2,000회로 계산하며, "
              "최소 50신호·20신호일·10종목에 못 미치면 제시하지 않는다. 비용 후 신호 평균은 동일 비중의 일봉 수익률 대용치이며 아래 포트폴리오 수익률과 다르다.", "",
              "| 시장 | 모델 | 청산 방식 | 포트폴리오 매매 수 | 순수익률 | 최대 낙폭 | 경로 불확실 매매 수 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for market in MARKETS:
        for name in MODELS:
            for mode in MODES:
                value = backtest[market]["models"][name]["portfolios"][mode]
                lines.append(f"| {MARKETS[market]} | {MODELS[name]} | {MODES[mode]} | {value['trade_count']:,} | "
                             f"{percent(value['total_return'])} | {percent(value['max_drawdown'])} | {value['uncertain_trades']:,} |")
    config = data["backtest_config"]
    lines += ["", f"기준 비용은 왕복 **20bps(0.20%)**. 시작자금은 국내 {config['initial_krw']:,.0f}원, 미국 ${config['initial_usd']:,.0f}다. "
              f"최대 {config['max_positions']}종목, 종목당 시작일 평가자산의 {percent(config['position_fraction'])}, "
              f"과거 20일 거래량 중앙값의 {percent(config['volume_fraction'], 3)}까지의 정수 주식 수를 허용했다. "
              "실제 시가 체결을 가정하며, 당일 장중 매도대금을 같은 날 시가 매수에 재사용하지 않는다. 가격 누락은 자금 잠금·불확실 경로로 표시한다.", "",
              "## 비용 민감도", "",
              "| 시장 | 모델 | 청산 방식 | 0bps | 10bps | 20bps | 40bps |",
              "|---|---|---|---:|---:|---:|---:|"]
    for market in MARKETS:
        lookup = {(row["model"], row["exit_mode"], row["cost_bps"]): row for row in backtest[market]["cost_sensitivity"]}
        for name in MODELS:
            for mode in MODES:
                values = [percent(lookup[(name, mode, cost)]["total_return"]) for cost in COSTS]
                lines.append("| " + " | ".join([MARKETS[market], MODELS[name], MODES[mode], *values]) + " |")
    lines += ["", "비용은 매수·매도 실제 금액에 절반씩 적용했다. 호가 스프레드, 시장 충격, 부분 체결, 호가 제한과 체결 지연을 따로 재현하지 않았으므로 실행 가능 수익률을 보장하지 않는다. "
              "+0.5% 이익 / −0.4% 손실의 두 결과만 있다고 단순화하면 왕복 20bps에서 손익분기 성공률은 약 66.67%다. "
              "새 연구 기준은 성공률 신뢰구간 하한이 이 값을 넘고 비용 후 평균수익 하한도 양수여야 한다. 미도달 종가 청산이 있어 실제 손익분기는 결과 구성에 따라 다를 수 있다.", "",
              "## 목표폭 축소가 라벨에 준 영향", "",
              "마지막 개발 감사 구간의 같은 창에 두 목표를 적용한 비교다. 양쪽 모두 도달하는 날이 많아지면 손절 우선 규칙 때문에 목표를 좁혀도 성공 라벨이나 매수 신호가 늘지 않을 수 있다.", "",
              "| 시장 | 같은 감사 창 수 | 기존 성공 라벨 비율 | 신규 성공 라벨 비율 | 기존 양쪽 도달 비율 | 신규 양쪽 도달 비율 |",
              "|---|---:|---:|---:|---:|---:|"]
    for market in MARKETS:
        audit = data["labels"][market]["walk_2024"]["audit"]
        lines.append(f"| {MARKETS[market]} | {audit['samples']:,} | {percent(audit['old_success_rate_same_rows'])} | "
                     f"{percent(audit['new_success_rate'])} | {percent(audit['old_both_touch_rate'])} | {percent(audit['new_both_touch_rate'])} |")
    lines += ["", "## 개발 구간 검증과 모델 저장", "",
              "| 시장·구간 | 추가 검증 연구 기준 | 개발 감사 연구 기준 | 최종 보존 트리 수(시드 42/43/44) |",
              "|---|---|---|---|"]
    for market in MARKETS:
        for fold, ensemble in train[market]["ensembles"].items():
            trees = "/".join(str(row["best_iteration"]) for row in ensemble["members"])
            lines.append(f"| {MARKETS[market]} · {fold} | {gate(ensemble['validation_qualification'])} | {gate(ensemble['qualification'])} | {trees} |")
    lines += ["", "미통과 사유(원본 결과의 검증 코드):", ""]
    for market in MARKETS:
        for fold, ensemble in train[market]["ensembles"].items():
            reasons = sorted(set(ensemble["qualification"]["reasons"] + ensemble["validation_qualification"]["reasons"]))
            lines.append(f"- {MARKETS[market]} {fold}: {', '.join(reasons) if reasons else '개발 구간 기준상 미통과 사유 없음'}")
    lines += ["", f"저장 버전: `{data['bundle']['version']}`. CPU 내보내기에서 4개 과거 입력 × 3개 진입가 × 2시장 = "
              f"**{total_calls}개 예측 비교**를 통과했고 입력·기존 보호 파일을 변경하지 않았다. 이는 수치 일치 검증이며 수익성이나 실전 준비 완료 인증이 아니다.", "",
              "- 학습 산출물: `" + str(data["paths"]["training"]) + "`",
              "- 비교 백테스트: `" + str(data["paths"]["backtest"]) + "`",
              "- 독립 추론 번들: `" + str(data["paths"]["bundle"]) + "`",
              "- 재실행·추론 사용법: [MARK1_0504 문서](../../docs/MARK1_0504.md)", "",
              "원본 +1%/−0.9% 모델은 그대로 보존했다. 새 모델은 GUI에 자동 적용하지 않았고 자동주문·예약주문·실전 주문도 시작하지 않았다. "
              "일봉 전체 구간의 보수적 성공 라벨이지, 임의의 장중 시점 이후 익절이 먼저 올 확률로 검증된 모델이 아니다.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "outputs/mark1/half-20260920")
    parser.add_argument("--backtest", type=Path, default=ROOT / "outputs/mark1/half-backtest-20260920")
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/mark1_0504")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/mark1-0504-20260920")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh report folder; existing outputs are not overwritten")
    data = load_inputs(args.training, args.backtest, args.bundle)
    report = report_text(data)
    args.output.mkdir(parents=True, exist_ok=False)
    comparison_figure(data, args.output / "comparison.png")
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    if any(digest(path) != value for path, value in data["input_sha256"].items()):
        raise RuntimeError("Completed input artifacts changed during report rendering")
    (args.output / "report-inputs.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": data["input_sha256"],
        "generator_sha256": digest(Path(__file__)), "input_files_unchanged": True,
        "research_only": True, "orders_started": False}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str((args.output / 'REPORT.md').resolve()),
                      "figure": str((args.output / 'comparison.png').resolve()),
                      "orders_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
