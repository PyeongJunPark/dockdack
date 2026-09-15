"""Emit one BUY/SELL/HOLD signal from the LSTM checkpoint and current prices.

uv run --no-sync python -m examples.emit_lstm_signal --checkpoint PATH --kiwoom-demo
Uses read-only Kiwoom requests. No order or cancellation method is called.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from dockdack import BrokerError, KiwoomBroker, KiwoomConfig, Market, TradingMode
from dockdack.signals import decimal_value, evaluate_signal
from examples.train_lstm_daily import FEATURE_NAMES, LSTMClassifier, latest_prediction, load_bars, make_features, prepare_bars


def infer(checkpoint: dict, dates, bars) -> dict:
    config = checkpoint["metadata"]
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)
    if (config["feature_names"] != FEATURE_NAMES or mean.shape != (5,) or std.shape != (5,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Invalid checkpoint features or normalization")
    model = LSTMClassifier(**config["architecture"]).cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction = latest_prediction(
        model, make_features(bars), mean, std, config["lookback"], dates, torch.device("cpu"),
    )
    if not np.isfinite(prediction["up_probability"]) or not 0 <= prediction["up_probability"] <= 1:
        raise ValueError("Model returned an invalid probability")
    return prediction


def result_context(config, price, quantity, average, source):
    return {
        "symbol": config["symbol"], "exchange": config["exchange"], "source": source,
        "current_price": str(price), "position_quantity": str(quantity),
        "average_entry_price": str(average) if average is not None else None,
        "profit_basis": "gross price return from average entry, excluding fees and taxes",
        "prediction": None, "previous_close": None, "previous_trade_date": None,
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "signal_only": True,
    }


def checkpoint_market(config: dict) -> Market:
    if config["exchange"] == "KRX":
        return Market.DOMESTIC
    if config["exchange"] in {"NA", "ND", "NY"}:
        return Market.US
    raise ValueError("Checkpoint must use KRX, NA, ND or NY")


def position_cost(account, market, symbol, exchange):
    currency = "KRW" if market is Market.DOMESTIC else "USD"
    if account.market is not market or account.currency != currency:
        raise ValueError("Account market does not match the checkpoint")
    quantity, cost = Decimal(0), Decimal(0)
    for position in account.positions:
        # US balances expose names such as NASDAQ, while quotes/charts use ND.
        position_exchange = str(position.exchange).strip().upper()
        if market is Market.US:
            position_exchange = {"NASDAQ": "ND", "NYSE": "NY", "AMEX": "NA"}.get(
                position_exchange, position_exchange,
            )
        if position.symbol != symbol or position_exchange != exchange:
            continue
        if position.market is not market or position.currency != currency:
            raise ValueError("Position market mismatch")
        qty = decimal_value(position.quantity, "position quantity", allow_zero=True)
        if qty > 0:
            average = decimal_value(position.average_price, "average entry price")
            quantity += qty
            cost += qty * average
    return quantity, cost / quantity if quantity else None


def recent_completed_bars(broker, config, market, session_date):
    symbol, exchange = config["symbol"], config["exchange"]
    if market is Market.DOMESTIC:
        pages = broker.iter_daily_bars_domestic(
            symbol, exchange=exchange, base_date=session_date - timedelta(days=1), max_pages=20,
        )
    else:
        pages = broker.iter_daily_bars_us(symbol, exchange=exchange, max_pages=20)
    rows = {}
    required = config["lookback"] + 1
    for page in pages:
        for bar in page:
            if bar.symbol != symbol or bar.exchange != exchange or bar.market is not market:
                raise ValueError("Chart symbol, exchange or market mismatch")
            if bar.trade_date < session_date:
                rows[bar.trade_date] = (
                    bar.trade_date.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume,
                )
        if len(rows) >= required:
            dates, bars, _ = prepare_bars([rows[d] for d in sorted(rows)])
            if len(bars) >= required:
                return dates[-required:], bars[-required:]
    raise ValueError(f"Need {required} valid completed bars before {session_date}")


def live_signal(checkpoint, broker, session_date):
    config = checkpoint["metadata"]
    market = checkpoint_market(config)
    if broker.mode is not TradingMode.DEMO:
        raise ValueError("This example uses the Kiwoom demo environment only")
    symbol, exchange = config["symbol"], config["exchange"]
    account = (broker.account_domestic(exchange=exchange) if market is Market.DOMESTIC
               else broker.account_us(exchange=exchange, symbol=symbol))
    quantity, average = position_cost(account, market, symbol, exchange)
    quote = broker.get_quote(market, symbol, exchange=exchange)
    expected_currency = "KRW" if market is Market.DOMESTIC else "USD"
    if (quote.symbol != symbol or quote.exchange != exchange or quote.market is not market
            or quote.currency != expected_currency):
        raise ValueError("Quote does not match the checkpoint's instrument/currency")
    context = result_context(config, quote.price, quantity, average, "kiwoom_demo")
    context["session_date"] = session_date.isoformat()
    decision = evaluate_signal(current_price=quote.price, position_quantity=quantity,
                               average_entry_price=average)
    if decision.action == "SELL":
        return {**context, **decision.to_dict()}
    # Today's unfinished candle is excluded from the daily-trained LSTM.
    # Fresh rolling history comes from Kiwoom, avoiding the old local DB snapshot.
    try:
        dates, bars = recent_completed_bars(broker, config, market, session_date)
        prediction = infer(checkpoint, dates, bars)
        previous = Decimal(str(bars[-1, 3]))
        context.update(prediction=prediction, previous_close=str(previous),
                       previous_trade_date=str(dates[-1]))
        decision = evaluate_signal(current_price=quote.price, previous_close=previous,
                                   predicted_direction=prediction["predicted_direction"],
                                   position_quantity=quantity, average_entry_price=average)
    except (ValueError, RuntimeError, BrokerError) as exc:
        context["prediction_error"] = str(exc)
    return {**context, **decision.to_dict()}


def manual_signal(checkpoint, database, price, previous, previous_date, quantity, average):
    config = checkpoint["metadata"]
    price = decimal_value(price, "current_price")
    previous = decimal_value(previous, "previous_close")
    context = result_context(config, price, quantity, average, "manual_prices")
    context.update(previous_close=str(previous), previous_trade_date=previous_date.isoformat())
    decision = evaluate_signal(current_price=price, position_quantity=quantity,
                               average_entry_price=average)
    if decision.action == "SELL":
        return {**context, **decision.to_dict()}
    try:
        dates, bars, _ = load_bars(database or Path(config["database"]), config["symbol"],
                                   config["exchange"], config["start"], end=previous_date.isoformat())
        if str(dates[-1]) != previous_date.isoformat() or Decimal(str(bars[-1, 3])) != previous:
            return {**context, **decision.to_dict(), "reason": "STALE_OR_MISMATCHED_DAILY_DATA"}
        prediction = infer(checkpoint, dates, bars)
        context["prediction"] = prediction
        decision = evaluate_signal(current_price=price, previous_close=previous,
                                   predicted_direction=prediction["predicted_direction"],
                                   position_quantity=quantity, average_entry_price=average)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        context["prediction_error"] = str(exc)
    return {**context, **decision.to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kiwoom-demo", action="store_true", help="Read current quote, holdings and recent bars")
    parser.add_argument("--db", type=Path, help="Manual mode only: local daily database")
    parser.add_argument("--current-price")
    parser.add_argument("--previous-close")
    parser.add_argument("--previous-date", type=date.fromisoformat)
    parser.add_argument("--quantity", default="0")
    parser.add_argument("--average-price")
    args = parser.parse_args()
    if args.kiwoom_demo and any(v is not None for v in (
        args.db, args.current_price, args.previous_close, args.previous_date, args.average_price,
    )):
        parser.error("Kiwoom mode reads prices/holdings itself; do not mix manual inputs")
    if args.kiwoom_demo and args.quantity != "0":
        parser.error("Kiwoom mode reads quantity from the account")
    if not args.kiwoom_demo and any(v is None for v in (
        args.current_price, args.previous_close, args.previous_date,
    )):
        parser.error("Manual mode requires --current-price, --previous-close and --previous-date")
    torch.set_num_threads(2)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint["metadata"]
        if args.kiwoom_demo:
            market = checkpoint_market(config)
            timezone = ZoneInfo("Asia/Seoul" if market is Market.DOMESTIC else "America/New_York")
            session_date = datetime.now(timezone).date()
            broker = KiwoomBroker(KiwoomConfig.from_env(TradingMode.DEMO, market=market))
            result = live_signal(checkpoint, broker, session_date)
        else:
            result = manual_signal(checkpoint, args.db, args.current_price, args.previous_close,
                                   args.previous_date, args.quantity, args.average_price)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, BrokerError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
