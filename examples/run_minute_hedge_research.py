"""Verify collected minute JSONL and write an offline market-hedge research report.

No broker, account, order, GUI, or model deployment path is imported.  A
collector receipt verifies byte integrity, not the financial accuracy of the
broker feed.  Every candidate is paper-only; ``pass`` means only that there
was enough test data to evaluate a candidate, never that it was profitable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zoneinfo import ZoneInfo

from dockdack.market_schedule import session_on
from dockdack.minute_hedge_research import (
    EXECUTION_DELAY_BARS, HedgeLeg, HedgePolicy, MinuteBar, MinuteSession,
    PaperCosts, Variant, walk_forward,
)
from dockdack.models import Market


_REQUIRED_BAR_FIELDS = frozenset({"market", "exchange", "symbol", "timestamp", "bar_minutes",
                                  "open", "high", "low", "close", "volume"})
_MARKET_ZONE = {Market.DOMESTIC: ZoneInfo("Asia/Seoul"), Market.US: ZoneInfo("America/New_York")}
_THRESHOLDS = {
    Variant.RESIDUAL_Z: (1.5, 2.0, 2.5),
    Variant.SECTOR_RELATIVE: (0.005, 0.01, 0.015),
    Variant.TURNOVER_VWAP: (0.005, 0.01, 0.015),
    Variant.ATR_DROP: (1.5, 2.0, 2.5),
    Variant.GAP_RELATIVE: (0.005, 0.01, 0.015),
    Variant.VOLUME_SHOCK_REVERSAL: (0.005, 0.01, 0.015),
    Variant.RANGE_RECOVERY: (0.005, 0.01, 0.015),
}
_SIGNAL_DEFINITIONS = {
    Variant.RESIDUAL_Z: "prior-window stock/spot-index beta residual z-score below negative threshold",
    Variant.SECTOR_RELATIVE: "stock underperforms aligned sector over prior window",
    Variant.TURNOVER_VWAP: "completed stock close below prior actual-turnover VWAP",
    Variant.ATR_DROP: "stock drop from prior anchor measured in prior-window ATR units",
    Variant.GAP_RELATIVE: "verified previous-day stock/index gap and current relative underperformance",
    Variant.VOLUME_SHOCK_REVERSAL: (
        "completed signal-bar volume at least 2x prior-window mean with stock/index relative drop"),
    Variant.RANGE_RECOVERY: (
        "completed signal-bar low breaches prior-window low, closes >=0.5% above its low, "
        "while underperforming index"),
}


def _number(raw: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'} and finite")
    return number


def _utc_stamp(raw: Any, name: str) -> datetime:
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be ISO-8601")
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return stamp.astimezone(timezone.utc)


def _verified_file(path: Path, market: Market) -> dict[str, Any]:
    path = path.resolve(strict=True)
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("source") != "kiwoom_rest_demo_minute_chart":
        raise ValueError(f"{path.name}: unsupported or missing collector receipt")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != receipt.get("sha256"):
        raise ValueError(f"{path.name}: SHA-256 does not match collector receipt")
    if receipt.get("market") != market.value or receipt.get("regular_session_filter") is not True:
        raise ValueError(f"{path.name}: market or regular-session receipt mismatch")
    interval = receipt.get("interval_minutes")
    if type(interval) is not int or interval not in {1, 3, 5, 10, 15, 30, 45, 60}:
        raise ValueError(f"{path.name}: unsupported bar interval")
    collected = _utc_stamp(receipt.get("collected_at_utc"), "collected_at_utc")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, datetime]] = set()
    previous_key: tuple[str, str, datetime] | None = None
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSONL") from exc
        if not isinstance(row, dict) or not _REQUIRED_BAR_FIELDS <= row.keys() or (
                set(row) - _REQUIRED_BAR_FIELDS not in (set(), {"turnover"})):
            raise ValueError(f"{path.name}:{line_number}: unexpected minute-bar fields")
        if row["market"] != market.value or row["bar_minutes"] != interval:
            raise ValueError(f"{path.name}:{line_number}: market or interval mismatch")
        if not isinstance(row["exchange"], str) or row["exchange"] != receipt.get("exchange") or (
                not isinstance(row["symbol"], str) or not row["symbol"]):
            raise ValueError(f"{path.name}:{line_number}: receipt exchange or symbol mismatch")
        stamp = _utc_stamp(row["timestamp"], "timestamp").astimezone(_MARKET_ZONE[market])
        if stamp.isoformat() != row["timestamp"]:
            raise ValueError(f"{path.name}:{line_number}: timestamp is not exchange-local with correct offset")
        if stamp.astimezone(timezone.utc) + timedelta(minutes=interval) > collected:
            raise ValueError(f"{path.name}:{line_number}: bar not finalized at collection time")
        trading_session = session_on(market, stamp.date())
        edge = timedelta(minutes=interval)
        if trading_session is None or not trading_session.opened + edge < stamp < trading_session.closed - edge:
            raise ValueError(f"{path.name}:{line_number}: outside a verified regular session")
        bar = MinuteBar(stamp, _number(row["open"], "open", positive=True),
                        _number(row["high"], "high", positive=True),
                        _number(row["low"], "low", positive=True),
                        _number(row["close"], "close", positive=True),
                        _number(row["volume"], "volume"),
                        _number(row["turnover"], "turnover") if "turnover" in row else None)
        key = (row["exchange"], row["symbol"], stamp)
        if key in seen or (previous_key is not None and key <= previous_key):
            raise ValueError(f"{path.name}:{line_number}: duplicate or unordered minute bar")
        seen.add(key)
        previous_key = key
        rows.append({"exchange": row["exchange"], "symbol": row["symbol"], "bar": bar})
    if not rows or type(receipt.get("accepted_bars")) is not int or receipt["accepted_bars"] != len(rows):
        raise ValueError(f"{path.name}: accepted-bar count mismatch")
    valid_symbols = set(receipt.get("symbols") or [])
    index_code = receipt.get("index_code")
    if any(row["symbol"] not in valid_symbols | ({index_code} if index_code else set()) for row in rows):
        raise ValueError(f"{path.name}: row symbol missing from collector receipt")
    return {"path": str(path), "receipt_path": str(receipt_path), "receipt": receipt,
            "sha256": receipt["sha256"], "interval_minutes": interval, "rows": tuple(rows)}


def _series(file: dict[str, Any], symbol: str, *, index: bool = False) -> dict[datetime, MinuteBar]:
    if not symbol or symbol != symbol.strip():
        raise ValueError("symbol must be nonempty and trimmed")
    if index and file["receipt"].get("index_code") != symbol:
        raise ValueError("index code does not match collector receipt")
    matching = [row for row in file["rows"] if row["symbol"] == symbol
                and ((row["exchange"] == "INDEX") if index else (row["exchange"] != "INDEX"))]
    if not matching:
        raise ValueError(f"{symbol}: no {'index' if index else 'stock/ETF'} bars in verified input")
    return {row["bar"].time: row["bar"] for row in matching}


def _runs(times: list[datetime], interval: int) -> list[list[datetime]]:
    groups: list[list[datetime]] = []
    for stamp in times:
        if not groups or stamp - groups[-1][-1] != timedelta(minutes=interval):
            groups.append([])
        groups[-1].append(stamp)
    return groups


def _aligned_sessions(files: dict[str, dict[str, Any]], symbols: dict[str, str], *,
                      market: Market, excluded_days: dict) -> tuple[tuple[MinuteSession, ...], list[dict[str, Any]]]:
    stock = _series(files["stock"], symbols["stock"])
    index = _series(files["index"], symbols["index"], index=True)
    etf = _series(files["etf"], symbols["etf"]) if "etf" in files else None
    sector = _series(files["sector"], symbols["sector"]) if "sector" in files else None
    sources = {"stock": stock, "index": index}
    if etf is not None:
        sources["etf"] = etf
    if sector is not None:
        sources["sector"] = sector
    interval = files["stock"]["interval_minutes"]
    if any(file["interval_minutes"] != interval for file in files.values()):
        raise ValueError("all inputs must have the same minute interval")
    all_days = sorted({stamp.date() for series in sources.values() for stamp in series})
    sessions: list[MinuteSession] = []
    audit: list[dict[str, Any]] = []
    for day in all_days:
        if day in excluded_days:
            audit.append({"day": day.isoformat(), "status": "abstain", "reason": excluded_days[day]})
            continue
        day_sets = [{stamp for stamp in series if stamp.date() == day} for series in sources.values()]
        aligned = sorted(set.intersection(*day_sets))
        if not aligned:
            audit.append({"day": day.isoformat(), "status": "abstain", "reason": "no_timestamp_intersection"})
            continue
        runs = _runs(aligned, interval)
        chosen = sorted(runs, key=lambda run: (-len(run), run[0]))[0]
        # Do not fill gaps; one longest complete run per day keeps the
        # walk-forward unit unambiguous. Discarded segments are reported.
        minimum = 12 + 6 + 2 * EXECUTION_DELAY_BARS
        if len(chosen) < minimum:
            audit.append({"day": day.isoformat(), "status": "abstain", "reason": "contiguous_run_too_short",
                          "longest_run_bars": len(chosen), "required_bars": minimum,
                          "aligned_bars": len(aligned), "discarded_aligned_bars": len(aligned) - len(chosen)})
            continue
        currency = "KRW" if market is Market.DOMESTIC else "USD"
        sessions.append(MinuteSession(
            stock_id=symbols["stock"], benchmark_index_id=symbols["index"],
            stock_currency=currency, interval_minutes=interval,
            stock=tuple(stock[t] for t in chosen), benchmark_index=tuple(index[t] for t in chosen),
            source="collector_sha256:" + ",".join(files[k]["sha256"] for k in sorted(files)),
            sector_id=symbols.get("sector"),
            sector=tuple(sector[t] for t in chosen) if sector is not None else None,
            inverse_etf_id=symbols.get("etf"),
            inverse_etf_currency=currency if etf is not None else None,
            inverse_etf_index_multiple=-1.0 if etf is not None else None,
            inverse_etf=tuple(etf[t] for t in chosen) if etf is not None else None,
        ))
        audit.append({"day": day.isoformat(), "status": "used", "run_start_label": chosen[0].isoformat(),
                      "run_end_label": chosen[-1].isoformat(), "run_bars": len(chosen),
                      "aligned_bars": len(aligned), "discarded_aligned_bars": len(aligned) - len(chosen),
                      "source_gap_count": max(0, len(runs) - 1)})
    return tuple(sessions), audit


def _candidate_report(variant: Variant, sessions: tuple[MinuteSession, ...], costs: PaperCosts,
                      *, hedge_leg: HedgeLeg, notional: float, min_prior_volume: float,
                      min_prior_turnover: float, min_train_sessions: int,
                      min_train_trades: int) -> dict[str, Any]:
    base = {"variant": variant.value, "signal_definition": _SIGNAL_DEFINITIONS[variant],
            "data_gate": "abstain", "pass_means_profitable": False,
            "deployment_allowed": False,
            "entry_threshold_grid": list(_THRESHOLDS[variant]), "folds": [],
            "out_of_sample_trades": 0, "out_of_sample_net_pnl": None}
    if len(sessions) <= min_train_sessions:
        return {**base, "reason": "insufficient_distinct_sessions_for_walk_forward"}
    if variant is Variant.SECTOR_RELATIVE and sessions[0].sector is None:
        return {**base, "reason": "aligned_sector_minute_data_missing"}
    if variant is Variant.TURNOVER_VWAP and any(
            bar.turnover is None for session in sessions for bar in session.stock):
        return {**base, "reason": "actual_minute_turnover_missing; OHLCV_cannot_reconstruct_VWAP"}
    if variant is Variant.GAP_RELATIVE and any(
            session.previous_close_day is None for session in sessions):
        return {**base, "reason": "verified_previous_daily_stock_and_index_closes_missing"}
    # These fixed grids are exploratory candidates, not trained weights or
    # a claim of favorable market performance.
    interval = sessions[0].interval_minutes
    candidates = tuple(HedgePolicy(variant, lookback=12, entry_threshold=threshold,
                                   take_profit=0.01, stop_loss=0.015,
                                   max_hold_minutes=interval * 6,
                                   min_prior_volume=min_prior_volume,
                                   min_prior_turnover=min_prior_turnover)
                       for threshold in _THRESHOLDS[variant])
    folds = walk_forward(sessions, candidates, costs, notional_per_leg=notional,
                         min_train_sessions=min_train_sessions, test_sessions=1,
                         min_train_trades=min_train_trades, hedge_leg=hedge_leg)
    output_folds = []
    all_test_trades = []
    for fold in folds:
        trades = [trade for result in fold.test_results for trade in result.trades]
        all_test_trades.extend(trades)
        output_folds.append({"train_first_day": fold.training_days[0].isoformat(),
                             "train_last_day": fold.training_days[-1].isoformat(),
                             "train_sessions": len(fold.training_days),
                             "test_days": [day.isoformat() for day in fold.test_days],
                             "selected_entry_threshold": (fold.selected_policy.entry_threshold
                                                          if fold.selected_policy else None),
                             "train_net_pnl_for_selection": fold.training_net_pnl,
                             "test_trade_count": len(trades),
                             "test_net_pnl": sum(trade.net_pnl for trade in trades) if trades else None})
    if not all_test_trades:
        return {**base, "reason": "no_out_of_sample_trades_or_training_selection_abstained",
                "folds": output_folds}
    return {**base, "data_gate": "pass", "reason": "historical_paper_evaluation_only",
            "folds": output_folds, "out_of_sample_trades": len(all_test_trades),
            "out_of_sample_sample_sufficiency": ("insufficient_for_profitability_inference"
                                                 if len(all_test_trades) < 30 else "not_automatically_qualified"),
            "out_of_sample_net_pnl": sum(trade.net_pnl for trade in all_test_trades),
            "out_of_sample_inverse_etf_vs_benchmark_return_gap_mean": (
                sum(trade.inverse_etf_vs_benchmark_return_gap for trade in all_test_trades
                    if trade.inverse_etf_vs_benchmark_return_gap is not None) / len(all_test_trades)
                if hedge_leg is HedgeLeg.LONG_INVERSE_ETF else None)}


def run(*, stock_file: Path, index_file: Path, stock_symbol: str, index_code: str,
        market: Market, output: Path, costs: PaperCosts, notional_per_leg: float,
        min_prior_volume: float, min_train_sessions: int, min_train_trades: int,
        inverse_etf_file: Path | None = None, inverse_etf_symbol: str | None = None,
        confirm_inverse_etf_minus_one: bool = False,
        sector_file: Path | None = None, sector_symbol: str | None = None,
        min_prior_turnover: float = 0.0) -> dict[str, Any]:
    """Read-only inputs, new JSON report only. No orders or account calls."""
    if market is not Market.DOMESTIC:
        raise ValueError("this runner supports only the domestic ka20005 index; US index hedge is unverified")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("existing research report will not be overwritten")
    if (inverse_etf_file is None) != (inverse_etf_symbol is None):
        raise ValueError("inverse ETF file and symbol must be supplied together")
    if (sector_file is None) != (sector_symbol is None):
        raise ValueError("sector file and symbol must be supplied together")
    if inverse_etf_file is not None and not confirm_inverse_etf_minus_one:
        raise ValueError("explicit -1x inverse ETF product confirmation is required")
    if inverse_etf_file is None and confirm_inverse_etf_minus_one:
        raise ValueError("-1x confirmation has no ETF input")
    for name, value in (("notional_per_leg", notional_per_leg), ("min_prior_volume", min_prior_volume),
                        ("min_prior_turnover", min_prior_turnover)):
        _number(value, name, positive=name == "notional_per_leg")
    if min_train_sessions < 1 or min_train_trades < 1:
        raise ValueError("walk-forward minimums must be positive")
    file_paths = {"stock": stock_file, "index": index_file}
    symbols = {"stock": stock_symbol, "index": index_code}
    if inverse_etf_file is not None:
        file_paths["etf"], symbols["etf"] = inverse_etf_file, inverse_etf_symbol
    if sector_file is not None:
        file_paths["sector"], symbols["sector"] = sector_file, sector_symbol
    files = {key: _verified_file(path, market) for key, path in file_paths.items()}
    if len({file["path"] for file in files.values()}) != len(files):
        raise ValueError("stock, index, ETF, and sector must use distinct collected files")
    expected_index_basis = "ka20005_signed_abs_divided_by_100_points; demo_sample_verified_2026-09-29"
    if (files["index"]["receipt"].get("exchange") != "INDEX" or
            files["index"]["receipt"].get("index_price_basis") != expected_index_basis):
        raise ValueError("index receipt must verify ka20005 index-point price basis")
    for key in ("stock", "etf", "sector"):
        if key in files and files[key]["receipt"].get("exchange") != "KRX":
            raise ValueError(f"{key} receipt must identify the domestic KRX exchange")
    excluded_days = {}
    data_warnings = []
    for key, file in files.items():
        all_file_days = sorted({row["bar"].time.date() for row in file["rows"]})
        if file["receipt"].get("chart_truncated") is True:
            data_warnings.append(f"{key}:older_history_truncated_by_collection_page_cap")
            excluded_days[all_file_days[0]] = "oldest_page_boundary_may_be_partial"
        last_day = all_file_days[-1]
        local_collected = _utc_stamp(file["receipt"]["collected_at_utc"], "collected_at_utc").astimezone(
            _MARKET_ZONE[market])
        last_session = session_on(market, last_day)
        if (last_session is not None and local_collected.date() == last_day
                and local_collected <= last_session.closed + timedelta(minutes=file["interval_minutes"])):
            excluded_days[last_day] = "collection_during_unfinished_market_session"
    sessions, session_audit = _aligned_sessions(files, symbols, market=market,
                                                excluded_days=excluded_days)
    hedge_leg = (HedgeLeg.LONG_INVERSE_ETF if inverse_etf_file is not None
                 else HedgeLeg.SYNTHETIC_INDEX_SHORT)
    quality_reasons = []
    for key, file in files.items():
        receipt = file["receipt"]
        if receipt.get("chart_pagination_metadata_known") is not True or (
                type(receipt.get("chart_truncated")) is not bool):
            quality_reasons.append(f"{key}:chart_pagination_metadata_unverified")
    candidates = ([] if quality_reasons else [
        _candidate_report(variant, sessions, costs, hedge_leg=hedge_leg,
                          notional=notional_per_leg, min_prior_volume=min_prior_volume,
                          min_prior_turnover=min_prior_turnover,
                          min_train_sessions=min_train_sessions,
                          min_train_trades=min_train_trades)
        for variant in Variant])
    if quality_reasons:
        candidates = [{"variant": variant.value, "signal_definition": _SIGNAL_DEFINITIONS[variant],
                       "data_gate": "abstain", "pass_means_profitable": False,
                       "deployment_allowed": False,
                       "reason": ";".join(quality_reasons), "entry_threshold_grid": list(_THRESHOLDS[variant]),
                       "folds": [], "out_of_sample_trades": 0, "out_of_sample_net_pnl": None}
                      for variant in Variant]
    report = {
        "schema": "dockdack.minute_hedge_research.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline paper research only; no verified profitability, no broker execution",
        "data_gate_pass_definition": "historical paper test had at least one trade; not model qualification",
        "actual_broker_trades": False,
        "market": market.value, "stock_symbol": stock_symbol,
        "supported_scope": "domestic ka20005 spot-index research only; not US index hedge",
        "benchmark_index_code": index_code,
        "benchmark_price_basis": files["index"]["receipt"].get("index_price_basis"),
        "benchmark_unit_note": ("ka20005 prices are signed raw values divided by 100 into index points; "
                                "ratios only; verify receipt basis"),
        "hedge_leg": hedge_leg.value,
        "hedge_note": ("historical inverse ETF bar returns; -1x product designation is caller attestation; "
                       "ETF may target a futures index while benchmark 201 is the spot index"
                       if hedge_leg is HedgeLeg.LONG_INVERSE_ETF
                       else "synthetic market-index short is a reference cash-flow only, not an order"),
        "inverse_etf_symbol": inverse_etf_symbol,
        "execution_delay_bars": EXECUTION_DELAY_BARS,
        "bar_timestamp_note": "broker label start/end unknown; paper fill uses third later bar open",
        "costs": {"commission_bps_per_side": costs.commission_bps_per_side,
                  "slippage_bps_per_side": costs.slippage_bps_per_side,
                  "synthetic_index_borrow_bps_annual": costs.index_short_borrow_bps_annual,
                  "synthetic_borrow_applies_only_to": "synthetic_index_short",
                  "inverse_etf_expenses": "embedded_in_observed_ETF_prices_not_separately_estimated",
                  "trading_minutes_per_year": costs.trading_minutes_per_year,
                  "notional_per_leg": notional_per_leg},
        "policy_controls": {"lookback_bars": 12, "take_profit": 0.01, "stop_loss": 0.015,
                            "max_hold_bars": 6, "min_prior_volume": min_prior_volume,
                            "min_prior_turnover": min_prior_turnover,
                            "min_train_sessions": min_train_sessions,
                            "min_train_trades": min_train_trades,
                            "exit_signal_basis": "completed stock bar close; fill proxy third later bar open",
                            "threshold_selection": "expanding training dates only; next date held out"},
        "inputs": {key: {"path": file["path"], "receipt_path": file["receipt_path"],
                         "sha256": file["sha256"], "source": file["receipt"]["source"],
                         "symbol": symbols[key], "accepted_bars": file["receipt"]["accepted_bars"],
                         "interval_minutes": file["interval_minutes"],
                         "chart_truncated": file["receipt"].get("chart_truncated")}
                   for key, file in files.items()},
        "session_audit": session_audit,
        "used_contiguous_sessions": len(sessions),
        "data_quality_reasons": quality_reasons,
        "data_quality_warnings": data_warnings,
        "candidates": candidates,
        "limitations": [
            "input SHA validates file bytes against its receipt, not source authenticity or exchange prints",
            "broker candle label start/end semantics are unverified; fill is deliberately delayed three bars",
            "one longest aligned contiguous segment per day; gaps are not imputed",
            "OHLCV alone cannot reconstruct actual VWAP or prior official closing auction price",
            "paper third-later-bar prices do not establish executable liquidity or actual fills",
            "observed close triggers are executed three bars later, so realized hold/loss can exceed stated signal horizons",
            "one preselected stock; this report does not rank or validate a high-volume stock universe",
            "long-stock/long-inverse-ETF or synthetic-index-short mean reversion is directional relative-value risk, not risk-free arbitrage",
            "fixed notional uses fractional-share arithmetic; broker integer shares and order rounding omitted",
            "order-book bid/ask spread, displayed depth, queue position, and fill feasibility are not observed",
            "inverse ETF versus spot-index return gap includes differing underlying index, daily reset and fees; not official ETF tracking error",
            "equal stock and inverse-ETF notionals do not establish stock-beta-neutral market exposure",
            "research includes no live short, ETF order, account balance, or automatic trading",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = None
    try:
        with NamedTemporaryFile(prefix="hedge-research-", suffix=".json.tmp", dir=output.parent,
                                delete=False, mode="w", encoding="utf-8") as handle:
            stage = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        if output.exists():
            raise FileExistsError("report path was created while computing")
        stage.rename(output)
    finally:
        if stage is not None and stage.exists():
            stage.unlink()
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("domestic",), required=True,
                        help="currently only domestic ka20005 index research is supported")
    parser.add_argument("--stock-file", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--stock-symbol", required=True)
    parser.add_argument("--index-code", required=True)
    parser.add_argument("--inverse-etf-file", type=Path)
    parser.add_argument("--inverse-etf-symbol")
    parser.add_argument("--confirm-inverse-etf-minus-one", action="store_true")
    parser.add_argument("--sector-file", type=Path)
    parser.add_argument("--sector-symbol")
    parser.add_argument("--commission-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--synthetic-index-borrow-bps-annual", type=float, required=True)
    parser.add_argument("--notional-per-leg", type=float, required=True)
    parser.add_argument("--min-prior-volume", type=float, required=True)
    parser.add_argument("--min-prior-turnover", type=float, default=0.0)
    parser.add_argument("--min-train-sessions", type=int, default=5)
    parser.add_argument("--min-train-trades", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    market = Market(args.market)
    costs = PaperCosts(args.commission_bps, args.slippage_bps,
                       args.synthetic_index_borrow_bps_annual,
                       252 * (390 if market is Market.US else 390))
    report = run(stock_file=args.stock_file, index_file=args.index_file,
                 stock_symbol=args.stock_symbol, index_code=args.index_code,
                 market=market, output=args.output, costs=costs,
                 notional_per_leg=args.notional_per_leg,
                 min_prior_volume=args.min_prior_volume,
                 min_prior_turnover=args.min_prior_turnover,
                 min_train_sessions=args.min_train_sessions,
                 min_train_trades=args.min_train_trades,
                 inverse_etf_file=args.inverse_etf_file,
                 inverse_etf_symbol=args.inverse_etf_symbol,
                 confirm_inverse_etf_minus_one=args.confirm_inverse_etf_minus_one,
                 sector_file=args.sector_file, sector_symbol=args.sector_symbol)
    print(json.dumps({"report": str(args.output.resolve()),
                      "candidates": [{"variant": x["variant"], "data_gate": x["data_gate"]}
                                     for x in report["candidates"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
