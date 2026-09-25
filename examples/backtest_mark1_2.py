"""Frozen MK1.2 reused-history evaluation; no fitting, promotion or orders.

Both markets must finish their non-pilot training run and be exported before
any evaluation data is read. Actual-open portfolio results are kept separate
from counterfactual candidate-price sensitivity, which is not extra trading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from dockdack.mark1_2_data import load_dataset, read_sessions
from dockdack.mark1_backtest import simulate_portfolio
from dockdack.mark1_backtest_data import load_price_panel
from dockdack.mark1_data import barrier_outcomes
from dockdack.mark1_selective_policy import evaluate_signals, qualification
from dockdack.research_artifacts import (ArtifactResolver, assert_no_wal, read_json,
                                       sha256_file, write_new_json)
from examples.backtest_mark1 import attach_names, date_string, make_candidates
from examples.backtest_mark1_deep import common_test_indices
from examples.backtest_mark1_selective import signal_stability


ROOT = Path(__file__).resolve().parents[1]
MARKETS = ("domestic", "us")
COSTS = (0, 10, 20, 40)
PRICE_FACTORS = (.99, .995, 1., 1.005, 1.01)
SENSITIVITY_SAMPLES = 20_000
EVALUATION_STATUS = "reused_2025_plus_history_not_an_untouched_final_test"
CODE_FILES = (
    "examples/backtest_mark1_2.py", "dockdack/mark1_2_data.py",
    "dockdack/mark1_2_inference.py", "dockdack/mark1_2_models.py",
    "dockdack/mark1_backtest.py", "dockdack/mark1_backtest_data.py",
    "dockdack/mark1_data.py", "dockdack/mark1_deep_data.py",
    "dockdack/mark1_deep_models.py", "dockdack/mark1_metrics.py",
    "dockdack/mark1_selective_policy.py", "dockdack/mark1_deep_validation.py",
    "examples/backtest_mark1.py", "examples/backtest_mark1_deep.py",
    "examples/backtest_mark1_selective.py", "examples/export_mark1_2.py",
    "dockdack/research_artifacts.py", "dockdack/research_compat.py",
)
LIMITATIONS = [
    "2025+ history was inspected in prior research; this is reused history, not pristine out-of-sample evidence.",
    "Actual OPEN assumes immediate fills; no spread, impact, auction queue, partial fills, limits or latency is modeled.",
    "Daily OHLC cannot order both barriers: +1%/-0.9% both-touch is stop-first failure, not a verified intraday path.",
    "Zero-volume bars cannot fill; missing held-price paths lock capital and create explicitly uncertain accounting.",
    "20bp is a hypothetical roundtrip rate, half charged to each side's actual notional, not actual broker fees/tax.",
    "Same-day intraday sales cannot finance that day's OPEN purchases; no leverage or short selling.",
    "EOD CLOSE liquidation is not an app order five minutes before the close; final-session liquidation is a convention.",
    "Four known US symbols are excluded for all periods; other corporate-action issues and survivorship bias may remain.",
    "Price sensitivity uses the same sampled base events repeatedly: counterfactual queries are not independent real trades.",
    "No threshold, architecture, calibration or trading permission is changed by this evaluation.",
]


def _inspect_training(training_run, workspace):
    # Lazy imports allow parser/tests without creating or reading model artifacts.
    from examples.export_mark1_2 import inspect_training_run
    return inspect_training_run(training_run, workspace=workspace)


def _predictor(bundle, market):
    from dockdack.mark1_2_inference import Predictor
    return Predictor(bundle, market, device="cpu")


def _tree_hashes(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Existing unlinked bundle directory required")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked bundle member")
        if path.is_file():
            result[str(path.resolve())] = sha256_file(path)
    if not result:
        raise ValueError("Empty model bundle")
    return result


def recheck_hashes(hashes):
    for path, digest in hashes.items():
        if Path(path).is_symlink() or sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Protected input changed during evaluation: {path}")


def training_comparison(training_run, summaries):
    """Read all seed-42 screening results; verify their original artifact seals."""
    resolver = ArtifactResolver(Path(training_run), relocations={})
    frozen_path = resolver.location("protocol.json")
    frozen = read_json(frozen_path)
    rows, hashes = {}, {str(frozen_path): sha256_file(frozen_path)}
    for market in MARKETS:
        summary = summaries[market]
        protocol = summary.get("protocol", {})
        architectures = protocol.get("architectures")
        if (protocol != frozen.get("protocol")
                or not isinstance(frozen.get("code_sha256"), dict)
                or not isinstance(architectures, list) or len(architectures) != 7
                or len(set(architectures)) != 7):
            raise ValueError("All seven predeclared architecture comparisons are required")
        rows[market] = []
        for fold in ("walk_2022", "walk_2024"):
            for name in architectures:
                relative = f"{market}/{fold}/{name}-42/result.json"
                path = resolver.location(relative)
                row = read_json(path)
                contract = row.get("contract", {})
                context = contract.get("context", {})
                expected_context = {"market": market, "fold": fold, "source": summary.get("source"),
                                    "data_receipt": summary.get("data_receipt"),
                                    "code_sha256": frozen["code_sha256"]}
                if (row.get("completed") is not True or row.get("architecture") != name
                        or row.get("seed") != 42 or contract.get("architecture") != name
                        or contract.get("seed") != 42 or contract.get("pilot") is not False
                        or contract.get("protocol") != protocol
                        or any(context.get(key) != value for key, value in expected_context.items())):
                    raise ValueError("Screening result identity differs from completed training")
                seals = row.get("artifact_sha256")
                if not isinstance(seals, dict) or set(seals) != {"model.pt", "predictions.npz", "history.json"}:
                    raise ValueError("Missing screening model/result provenance")
                contract_path = path.parent / "contract.json"
                seal_path = path.parent / "model.sha256.json"
                if (read_json(contract_path) != contract
                        or read_json(seal_path).get("sha256") != seals["model.pt"]):
                    raise ValueError("Screening contract/checkpoint seal mismatch")
                hashes[str(contract_path)] = sha256_file(contract_path)
                hashes[str(seal_path)] = sha256_file(seal_path)
                for filename, digest in seals.items():
                    artifact = resolver.file(f"{market}/{fold}/{name}-42/{filename}", digest)
                    hashes[str(artifact.physical_path)] = digest
                hashes[str(path)] = sha256_file(path)
                metrics = row["selection"]
                rows[market].append({"architecture": name, "fold": fold, "seed": 42,
                    "parameters": row["parameters"], "best_epoch": row["best_epoch"],
                    "epochs": row["epochs"], "signals": metrics["signal_count"],
                    "precision": metrics["precision"], "net_mean_proxy": metrics["net_mean_return"],
                    "brier": metrics["overall"]["brier"], "qualified": row["qualification"]["qualified"],
                    "selected_architecture": name == summary["selected"],
                    "selected_ensemble_qualified": summary.get("research_qualified") is True})
    return rows, hashes


def verify_inputs(training_run, bundle, workspace):
    """Verify both markets, locks and source checkpoints before reading prices."""
    resolver = ArtifactResolver(Path(workspace))
    training_run = resolver.location(str(training_run))
    bundle = resolver.location(str(bundle))
    inspected = _inspect_training(training_run, resolver.workspace)
    summaries = inspected.get("summary", {})
    if (set(summaries) != set(MARKETS) or set(inspected.get("markets", {})) != set(MARKETS)
            or any(row.get("completed") is not True or row.get("pilot") is not False
                   or row.get("market") != market or row.get("deployment_allowed") is not False
                   or row.get("intraday_path_verified") is not False
                   for market, row in summaries.items())):
        raise ValueError("Both non-pilot markets must be complete and research-only")
    source_hashes = inspected.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not {"summary.json", "protocol.json"} <= set(source_hashes):
        raise ValueError("Missing complete training artifact hashes")
    hashes = _tree_hashes(bundle)
    manifest = read_json(bundle / "manifest.json")
    provenance = manifest.get("source_run", {})
    if (provenance.get("source_files_sha256") != source_hashes
            or provenance.get("summary_sha256") != source_hashes["summary.json"]
            or provenance.get("protocol_sha256") != source_hashes["protocol.json"]):
        raise ValueError("Bundle belongs to a different frozen training run")
    # Record the original run files too: exporter checks the checkpoint seals,
    # and the final recheck prevents concurrent replacement during inference.
    run_resolver = ArtifactResolver(training_run, relocations={})
    for relative, digest in source_hashes.items():
        artifact = run_resolver.file(relative, digest)
        hashes[str(artifact.physical_path)] = digest
    predictors = {}
    for market in MARKETS:
        trained = inspected["markets"][market]
        exported = manifest.get("markets", {}).get(market, {})
        if (exported.get("architecture") != trained.get("selected")
                or exported.get("model_config") != trained.get("model_config")
                or exported.get("calibration") != trained.get("calibration")):
            raise ValueError("Bundle architecture/config/calibration differs from selected training")
        original_members = trained.get("members", [])
        members = exported.get("members", [])
        if (len(members) != 3 or len(original_members) != 3
                or [row.get("seed") for row in members] != [42, 43, 44]
                or [row.get("seed") for row in original_members] != [42, 43, 44]
                or any(item.get("source_checkpoint_sha256") != original.get("sha256")
                       for item, original in zip(members, original_members))):
            raise ValueError("Bundle must contain the three selected source checkpoints in seed order")
        # The public predictor verifies bundle inventory, native checkpoint
        # checksums, runtime source seals, calibration and fixed target flags.
        predictors[market] = _predictor(bundle, market)
    recheck_hashes(hashes)
    return summaries, predictors, hashes


def predict_dataset(dataset, indices, predictor, *, batch_size=1024, factor=1.):
    """Only 30 past OHLCV bars and a candidate entry go into the predictor."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= len(dataset.starts))
            or type(batch_size) is not int or batch_size < 1
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError("Valid event indices, positive batch size and price factor required")
    result = np.empty(len(indices), dtype=np.float64)
    offsets = np.arange(30)
    for first in range(0, len(indices), batch_size):
        chosen = indices[first:first + batch_size]
        history = dataset.bars[dataset.starts[chosen, None] + offsets[None, :]]
        # Match augmentation-query arithmetic, not the separate float64 label
        # arithmetic. The public predictor and training bank both use FP32.
        entries = np.asarray(dataset.target_ohlc[chosen, 0], dtype=np.float32) * np.float32(factor)
        probabilities = np.asarray(predictor.predict_proba(history, entries), dtype=np.float64)
        if (probabilities.shape != (len(chosen),) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError("Predictor returned invalid calibrated probabilities")
        result[first:first + len(chosen)] = probabilities
    return result


def sensitivity_indices(indices, max_samples=SENSITIVITY_SAMPLES):
    """Label-blind uniform sample without replacement; same events for all prices."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or type(max_samples) is not int or max_samples < 1):
        raise ValueError("Nonempty event indices and positive sensitivity cap required")
    if len(indices) <= max_samples:
        return indices.copy()
    positions = np.sort(np.random.default_rng(42).choice(len(indices), max_samples, replace=False))
    return indices[positions].copy()


def price_sensitivity(dataset, indices, predictor, *, batch_size=1024,
                      max_samples=SENSITIVITY_SAMPLES):
    picked = sensitivity_indices(indices, max_samples)
    ohlc = dataset.target_ohlc[picked]
    rows = []
    for factor in PRICE_FACTORS:
        probabilities = predict_dataset(dataset, picked, predictor, batch_size=batch_size, factor=factor)
        outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0] * factor)
        selected = probabilities > .5
        rows.append({"factor": factor, "base_events": len(picked), "signals": int(selected.sum()),
                     "signal_rate": float(selected.mean()), "mean_probability": float(probabilities.mean()),
                     "label_success_rate": float(outcomes["success"].mean()),
                     "both_touch_rate": float(outcomes["both_touch"].mean()),
                     "precision": float(outcomes["success"][selected].mean()) if selected.any() else None,
                     "net_mean_proxy_after_20bps": float(outcomes["gross_return"][selected].mean() - .002)
                     if selected.any() else None})
    return {"diagnostic_only": True, "counterfactual_not_real_trades": True,
            "base_events": len(picked), "maximum_base_events": max_samples,
            "sampling": "uniform without replacement, label-blind seed 42; identical base rows for every factor",
            "sample_indices_sha256": hashlib.sha256(picked.astype("<i8", copy=False).tobytes()).hexdigest(),
            "probability_and_threshold_refit": False, "independent_observations": len(picked),
            "counterfactual_queries": len(picked) * len(PRICE_FACTORS), "rows": rows}


def prepare_prices(dataset, indices, database, market):
    """Compare every approved cache target with its DB row before removing zero volume."""
    start, end = int(dataset.target_dates[indices].min()), int(dataset.target_dates[indices].max())
    prices, sessions, panel = load_price_panel(database, market, dataset.manifest["symbols"], start, end)
    for index in indices:
        key = int(dataset.symbol_ids[index]), int(dataset.target_dates[index])
        if key not in prices or not np.array_equal(np.asarray(prices[key]), dataset.target_ohlc[index]):
            raise ValueError(f"Approved evaluation target differs from frozen source DB: {key}")
    for symbol, day in panel["zero_volume_keys"]:
        prices.pop((symbol, day), None)
    panel["zero_volume_execution_policy"] = "Unfillable; missing held path, never fabricated execution"
    return prices, sessions, panel


def run_market(market, args, trained, predictor, input_hashes):
    started = time.monotonic()
    dataset, source, receipt = load_dataset(market, args.workspace)
    previous = trained.get("data_receipt", {})
    if (source != trained.get("source") or receipt.get("experiment") != previous.get("experiment")
            or receipt.get("cache_sha256") != previous.get("cache_sha256")):
        raise ValueError("Evaluation source or all-period quarantine differs from training")
    database = Path(receipt["source"]["physical_path"])
    indices = common_test_indices(dataset, read_sessions(database))
    prices, sessions, panel = prepare_prices(dataset, indices, database, market)
    assert_no_wal(database)
    if sha256_file(database) != source["database_sha256"]:
        raise RuntimeError("Frozen database changed before evaluation")
    probabilities = predict_dataset(dataset, indices, predictor, batch_size=args.batch_size)
    ohlc = dataset.target_ohlc[indices]
    outcomes = barrier_outcomes(ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], ohlc[:, 0])
    dates, symbols = dataset.target_dates[indices], dataset.symbol_ids[indices]
    selected = probabilities > .5
    metrics = evaluate_signals(outcomes["success"], probabilities, outcomes["gross_return"],
                               dates, symbols, selected, cost_bps=20)
    candidates = make_candidates(dataset, indices, probabilities)
    signalled = {item["symbol_id"] for item in candidates}
    relevant = {key: value for key, value in prices.items() if key[0] in signalled}
    initial = 10_000_000 if market == "domestic" else 10_000
    folder = args.output_dir / market
    folder.mkdir(exist_ok=False)
    with (folder / "predictions.npz").open("xb") as stream:
        np.savez(stream, sample_indices=indices, dates=dates, symbol_ids=symbols,
                 probabilities=probabilities, labels=outcomes["success"],
                 both_touch=outcomes["both_touch"], gross_returns=outcomes["gross_return"],
                 evaluation_status=EVALUATION_STATUS)
    write_new_json(folder / "signals.json", {"actual_open_only": True, "signals": candidates})
    summary = {"market": market, "currency": "KRW" if market == "domestic" else "USD",
        "research_evaluation": EVALUATION_STATUS, "samples": len(indices),
        "range": {"first": date_string(min(sessions)), "last": date_string(max(sessions)), "sessions": len(sessions)},
        "architecture": trained["selected"], "initial_cash": initial, "source": source,
        "data_receipt": receipt, "price_panel": panel, "classification": metrics,
        "research_qualification": qualification(metrics), "raw_signals": len(candidates),
        "stability": signal_stability(outcomes["success"], outcomes["gross_return"], dates, symbols, selected),
        "sample_indices_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        "universe": "Uncapped approved 2014-2021 symbols; original 30-session 2025 boundary purge verified",
        "candidate_contract": "Actual OPEN, calibrated p strictly > 0.5, last 20 completed volumes only",
        "portfolios": {}, "cost_sensitivity": [], "research_only": True,
        "deployment_allowed": False, "intraday_path_verified": False,
        "input_sha256": input_hashes, "limitations": list(LIMITATIONS)}
    for mode in ("carry", "eod"):
        for cost in COSTS:
            simulation = simulate_portfolio(candidates, relevant, sessions, initial_cash=initial,
                exit_mode=mode, max_positions=20, position_fraction=.05,
                volume_fraction=.001, cost_bps=cost)
            attach_names(simulation, dataset.manifest["symbols"])
            simulation.update(market=market, model="mark1_2", research_evaluation=EVALUATION_STATUS,
                              deployment_allowed=False, diagnostic_only=True)
            write_new_json(folder / f"{mode}-cost{cost}.json", simulation)
            summary["cost_sensitivity"].append(simulation["summary"])
            if cost == 20:
                summary["portfolios"][mode] = simulation["summary"]
    summary["price_sensitivity"] = price_sensitivity(dataset, indices, predictor, batch_size=args.batch_size)
    recheck_hashes(input_hashes)
    assert_no_wal(database)
    if sha256_file(database) != source["database_sha256"]:
        raise RuntimeError("Frozen database changed during evaluation")
    summary.update(completed=True, elapsed_seconds=time.monotonic() - started)
    write_new_json(folder / "summary.json", summary)
    del dataset
    gc.collect()
    return summary


def _percent(value):
    return "—" if value is None else f"{value * 100:.4f}%"


def report_text(summaries, artifact_dir=None):
    lines = ["# MK1.2 재사용 역사 백테스트", "",
             "이 결과는 2025년 이후 이미 연구에서 확인한 과거 자료의 재평가입니다. 독립적인 새 테스트나 실전 체결 결과가 아닙니다.", "",
             "## 실제 시가 진입 · 왕복 20bp 비용", "",
             "| 시장 | 방식 | 신호 수 | 신호 성공률 | 체결 가정 거래 수 | 순수익률 | 최대낙폭 | 불확실 경로 거래 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for market in MARKETS:
        row = summaries[market]
        for mode, portfolio in row["portfolios"].items():
            lines.append(f"| {market} | {mode} | {row['raw_signals']} | {_percent(row['classification']['precision'])} "
                         f"| {portfolio['trade_count']} | {_percent(portfolio['total_return'])} "
                         f"| {_percent(portfolio['max_drawdown'])} | {portfolio['uncertain_trades']} |")
    lines += ["", "신호 성공률은 일봉 장벽 라벨 기준이며, 포트폴리오 승률과 다릅니다. 자금·동시보유·유동성 제한 때문에 신호가 모두 거래가 되지는 않습니다.",
              "", "## 학습 구조 비교 · 시드 42의 두 개발 구간", "",
              "아래는 2022/2024 개발 평가이며 위의 2025년 이후 재사용 백테스트와 다릅니다. "
              "‘선택(진단용)’은 선택되었더라도 두 구간 앙상블 자격을 통과하지 못한 모델입니다. 합격도 실전 배포 승인이 아닙니다.", "",
              "| 시장 | 구조 | 구간 | 매개변수 | 최적 epoch | 신호 | 성공률 | 20bp 차감 신호 평균 | Brier | 해당 구간 자격 | 선택 상태 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for market in MARKETS:
        for row in summaries[market].get("training_comparison", []):
            selected = ("선택(합격)" if row["selected_ensemble_qualified"] else "선택(진단용)") if row["selected_architecture"] else "미선택"
            lines.append(f"| {market} | {row['architecture']} | {row['fold']} | {row['parameters']:,} | "
                         f"{row['best_epoch']} | {row['signals']} | {_percent(row['precision'])} | "
                         f"{_percent(row['net_mean_proxy'])} | {row['brier']:.6f} | "
                         f"{'합격' if row['qualified'] else '미달'} | {selected} |")
    lines += ["", "## 기간·표본·신뢰구간", ""]
    for market in MARKETS:
        row, metrics = summaries[market], summaries[market]["classification"]
        block = metrics["block_bootstrap"]
        lines.append(f"- {market}: {row['range']['first']}~{row['range']['last']}, {row['range']['sessions']}거래일, "
                     f"{row['samples']:,}개 기본 사건. 신호 성공률 95% 구간 "
                     f"{_percent(block['precision_lower'])}~{_percent(block['precision_upper'])}. "
                     f"연구 자격 통과: {row['research_qualification']['qualified']}.")
    lines += ["", "구간은 날짜를 묶은 10거래일 이동 블록·2,000회·시드 42 방식이며, 표본 부족 시 산출하지 않습니다. "
              "여러 실험을 반복해 본 영향이나 데이터 선택 편향을 보정한 구간이 아니며 자동매매 승인을 뜻하지 않습니다.",
              "", "## 비용 민감도", "", "| 시장 | 방식 | 왕복 비용(bp) | 순수익률 | 최대낙폭 |",
              "|---|---|---:|---:|---:|"]
    for market in MARKETS:
        for row in summaries[market]["cost_sensitivity"]:
            lines.append(f"| {market} | {row['exit_mode']} | {row['cost_bps']:g} | "
                         f"{_percent(row['total_return'])} | {_percent(row['max_drawdown'])} |")
    lines += ["", "## 가상 후보가격 민감도 · 실거래 아님", "",
              "시장마다 최대 20,000개 기본 사건을 결과와 무관하게 균등 추출(시드 42)하고 같은 행에 다섯 가격을 적용했습니다. "
              "이는 독립 관측이나 실제 매매 횟수를 다섯 배 늘린 결과가 아닙니다. 가격이 당일 실제로 체결 가능했는지도 보장하지 않습니다.",
              "", "| 시장 | 시가 배율 | 기본 사건 수 | 가상 매수신호 | 신호 비율 | 신호 성공률 |",
              "|---|---:|---:|---:|---:|---:|"]
    for market in MARKETS:
        for row in summaries[market]["price_sensitivity"]["rows"]:
            lines.append(f"| {market} | {row['factor']:g} | {row['base_events']:,} | {row['signals']} | "
                         f"{_percent(row['signal_rate'])} | {_percent(row['precision'])} |")
    lines += ["", "## 고정 가정·한계", "",
              "초기자금 국내 1천만 원/미국 1만 달러, 최대 20종목, 시작일 자산의 5% 목표, "
              "과거 20일 중앙 거래량의 0.1% 정수 주 한도입니다. 익절 +1%·손절 -0.9%, 동시 도달 시 손절 우선입니다.", ""]
    lines.extend(f"- {value}" for value in LIMITATIONS)
    location = f"`{Path(artifact_dir).resolve().as_posix()}`" if artifact_dir is not None else "원본 백테스트 출력 폴더"
    lines += ["", f"전체 예측·개별 모의 거래·고정 입력 해시는 {location}의 시장별 JSON/NPZ 및 completed.json에 보관됩니다.", ""]
    return "\n".join(lines)


def output_path(value, workspace, training_run, bundle):
    """One fresh direct run directory, never a child of a frozen input run."""
    resolver = ArtifactResolver(Path(workspace))
    target = resolver.location(str(value))
    allowed = resolver.location("outputs/mark1")
    if target.parent != allowed or target.name in {"cache", "selective-deps", "prototype-gui-deps"}:
        raise ValueError("Use a new direct child directory of workspace/outputs/mark1")
    for protected in (Path(training_run).resolve(), Path(bundle).resolve()):
        if target == protected or target.is_relative_to(protected) or protected.is_relative_to(target):
            raise ValueError("Output overlaps a protected input")
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation output")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path("models/mark1_2_prototype"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Positive batch size required")
    resolver = ArtifactResolver(args.workspace)
    args.workspace = resolver.workspace
    args.training_run = resolver.location(str(args.training_run))
    args.bundle = resolver.location(str(args.bundle))
    args.output_dir = output_path(args.output_dir, args.workspace, args.training_run, args.bundle)
    # No data access or output creation before all training/export checks pass.
    trained, predictors, hashes = verify_inputs(args.training_run, args.bundle, args.workspace)
    comparison, comparison_hashes = training_comparison(args.training_run, trained)
    hashes.update(comparison_hashes)
    hashes.update({str(resolver.location(name)): sha256_file(resolver.location(name)) for name in CODE_FILES})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output_dir / "config.json", {
        "training_run": str(args.training_run), "bundle": str(args.bundle), "workspace": str(args.workspace),
        "batch_size": args.batch_size, "inference_device": "cpu", "inference_precision": "float32",
        "created_utc": datetime.now(timezone.utc).isoformat(), "research_evaluation": EVALUATION_STATUS,
        "threshold": "strict calibrated p > 0.5", "costs_bps": list(COSTS),
        "price_factors": list(PRICE_FACTORS), "sensitivity_max_base_events": SENSITIVITY_SAMPLES,
        "sensitivity_seed": 42, "input_sha256": hashes, "deployment_allowed": False})
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        summaries = {market: run_market(market, args, trained[market], predictors[market], hashes)
                     for market in MARKETS}
    finally:
        torch.set_num_threads(prior_threads)
    recheck_hashes(hashes)
    for market in MARKETS:
        summaries[market]["training_comparison"] = comparison[market]
    write_new_json(args.output_dir / "summary.json", summaries)
    with (args.output_dir / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report_text(summaries, args.output_dir))
    output_hashes = {path.relative_to(args.output_dir).as_posix(): sha256_file(path)
                     for path in args.output_dir.rglob("*") if path.is_file()}
    write_new_json(args.output_dir / "completed.json", {
        "completed": True, "markets": list(MARKETS), "research_evaluation": EVALUATION_STATUS,
        "output_sha256": output_hashes, "input_sha256": hashes,
        "deployment_allowed": False, "orders_started": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
