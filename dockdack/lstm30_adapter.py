"""Thirty-completed-bar model -> main's version-1 external signals (demo only).

This module never enables or submits orders. Costs are gross average entry prices,
not fee-adjusted P&L. Use one producer/writer per output and retain its state file:
an export is one immutable decision, including its original expiry time.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo


SOURCE_ID = "lstm30-mark0"
MAX_QUOTE_AGE = 15
SIGNAL_TTL = 120
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def utc_now():
    return datetime.now(timezone.utc)


@lru_cache(maxsize=8)
def _market_calendar(market, year):
    """Import lazily: missing calendar data blocks entries, never held exits."""
    from dockdack.market_schedule import exchange_calendar

    name = "XKRX" if market == "domestic" else "XNYS"
    return exchange_calendar(name, start=f"{year - 1}-01-01", end=f"{year + 1}-12-31")


def previous_trading_day(market, now):
    """Latest exchange session strictly before the market-local current date."""
    if market not in {"domestic", "us"}:
        raise ValueError("Unsupported market for completed-bar freshness")
    zone = ZoneInfo("Asia/Seoul" if market == "domestic" else "America/New_York")
    local_day = now.astimezone(zone).date()
    try:
        calendar = _market_calendar(market, local_day.year)
        latest = calendar.date_to_session((local_day - timedelta(days=1)).isoformat(), direction="previous")
    except Exception as exc:
        raise ValueError("Trading calendar unavailable; cannot validate completed-bar freshness") from exc
    return latest.date()


def timestamp(value, name="timestamp"):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be timezone-aware ISO 8601") from exc
    if result.tzinfo is None:
        raise ValueError(f"{name} requires a timezone")
    return result.astimezone(timezone.utc)


def number(value, name, *, zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite() or result < 0 or (not zero and not result):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return result


def atomic_json(path, payload):
    """Complete temp file + replace; independent of GUI/main-only dependencies."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".lstm30-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=invalid)


def instrument(stock):
    market, symbol, exchange = (stock.get(key) for key in ("market", "symbol", "exchange"))
    if (market == "domestic" and exchange == "KRX"):
        currency = "KRW"
    elif market == "us" and exchange in {"ND", "NY", "NA"}:
        currency = "USD"
    else:
        raise ValueError("Unsupported market/exchange")
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.]{0,19}", symbol):
        raise ValueError("Invalid symbol; preserve the exported symbol's case")
    if stock.get("currency") != currency:
        raise ValueError("Instrument currency mismatch")
    key = f"{market}:{exchange}:{symbol}"
    if stock.get("watch_id") != key:
        raise ValueError("watch_id does not match the instrument")
    return key


def validate_position(position, stock, now):
    """A missing snapshot is unknown, never an implicit zero-share position."""
    if not isinstance(position, dict):
        raise ValueError("A verified position snapshot is required")
    for key in ("market", "symbol", "exchange", "currency"):
        if position.get(key) != stock[key]:
            raise ValueError(f"Position {key} mismatch")
    age = (now - timestamp(position.get("fetched_at"), "position fetched_at")).total_seconds()
    if not 0 <= age <= MAX_QUOTE_AGE:
        raise ValueError("Position snapshot is stale or future-dated")
    quantity = number(position.get("quantity"), "position quantity", zero=True)
    sellable = number(position.get("sellable_quantity"), "sellable quantity", zero=True)
    if quantity != quantity.to_integral_value() or sellable != sellable.to_integral_value() or sellable > quantity:
        raise ValueError("Position/sellable quantity must be integral and consistent")
    average = number(position.get("average_price"), "average price") if quantity else None
    return quantity, sellable, average


def decide_position(*, current_price, quantity, sellable_quantity, average_price=None,
                    prediction=None):
    """Pure decision rule. Held exits take priority, even without a prediction."""
    price = number(current_price, "current price")
    qty = number(quantity, "quantity", zero=True)
    sellable = number(sellable_quantity, "sellable quantity", zero=True)
    if qty != qty.to_integral_value() or sellable != sellable.to_integral_value() or sellable > qty:
        raise ValueError("Invalid position quantities")
    if qty:
        average = number(average_price, "average price")
        if price >= average * Decimal("1.01"):
            return ({"action": "sell", "reason": "TAKE_PROFIT_1PCT", "cost_profit_pct": "1"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        if price <= average * Decimal("0.992"):
            return ({"action": "sell", "reason": "STOP_LOSS_0_8PCT", "cost_loss_pct": "0.8"}
                    if sellable else {"action": "hold", "reason": "NO_SELLABLE_POSITION"})
        return {"action": "hold", "reason": "POSITION_INSIDE_EXIT_BOUNDS"}
    if prediction is None:
        return {"action": "hold", "reason": "PREDICTION_UNAVAILABLE"}
    probability = number(prediction.get("probability_ge_1pct"), "probability", zero=True)
    threshold = number(prediction.get("buy_threshold"), "buy threshold")
    if probability > 1 or threshold > 1 or type(prediction.get("predicts_gain")) is not bool:
        raise ValueError("Invalid model decision")
    if prediction["predicts_gain"] != (probability >= threshold):
        raise ValueError("Model direction conflicts with its probability/threshold")
    return ({"action": "buy", "reason": "PREDICTED_NEXT_CLOSE_GAIN_GE_1PCT"}
            if prediction["predicts_gain"] else {"action": "hold", "reason": "BELOW_BUY_THRESHOLD"})


def completed_bars(stock, now):
    if stock.get("complete") is not True:
        raise ValueError("Chart export reports incomplete history")
    bars = stock.get("bars")
    if not isinstance(bars, list) or stock.get("available_days") != len(bars):
        raise ValueError("Chart bar count mismatch")
    zone = ZoneInfo("Asia/Seoul" if stock["market"] == "domestic" else "America/New_York")
    today = now.astimezone(zone).date()
    result, completed_dates, previous = [], [], None
    for bar in bars:
        day = date.fromisoformat(bar["date"])
        if previous is not None and day <= previous:
            raise ValueError("Bars must be strictly ordered without duplicate dates")
        previous = day
        # Never trust the exported boolean alone around local-midnight boundaries.
        if day >= today or bar.get("is_current_day") is True:
            continue
        row = [number(bar.get(key), key, zero=key == "volume")
               for key in ("open", "high", "low", "close", "volume")]
        if row[2] > min(row[0], row[3]) or row[1] < max(row[0], row[3]) or row[2] > row[1]:
            raise ValueError("Invalid OHLC relationships")
        floats = [float(value) for value in row]
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("OHLCV cannot be represented as finite model inputs")
        result.append(floats)
        completed_dates.append(day)
    if len(result) < 30:
        raise ValueError(f"Need 30 completed bars; got {len(result)} after excluding today's bar")
    expected = previous_trading_day(stock["market"], now)
    if completed_dates[-1] != expected:
        raise ValueError(f"Stale completed bars: last date {completed_dates[-1]} must equal previous exchange session {expected}")
    return result[-30:]


class LSTM30SignalProducer:
    """Callable adapter: ``payload, diagnostics = producer(charts)``.

    ``predictors`` maps domestic/us to ``dockdack.ml30.Predictor`` instances.
    ``position_provider(stock)`` returns an identity- and timestamp-bearing dict;
    see ``validate_position``. It must explicitly report zero shares when flat.
    ``quantity`` is the requested BUY size and maximum shares per SELL. SELL size
    is limited by current holdings, sellable shares and the user's notional cap.
    Persist ``state_path`` across restarts; never delete it to retry an order.
    """

    source_id = SOURCE_ID
    position_decision = staticmethod(decide_position)
    input_bars = staticmethod(completed_bars)

    def model_prediction(self, predictor, bars, price):
        """Strategy hook; the default remains the original mark0 close target."""
        return predictor.predict(bars), {
            "target_basis": "next trading day close >= last completed close * 1.01; not current entry return",
            "reference_close": str(bars[-1][3]),
        }

    def prediction_cache_key(self, predictor, stock, bars, price):
        """Cache only the default price-independent mark0 prediction hook.

        Mark1/prototype override ``model_prediction`` and use the current
        candidate price. They must re-infer each new chart export; inheriting
        the mark0 OHLCV-only cache would silently reuse the wrong probability.
        """
        if type(self).model_prediction is not LSTM30SignalProducer.model_prediction:
            return None
        return (id(predictor), stock["market"], stock["exchange"], stock["symbol"],
                tuple(tuple(bar) for bar in bars))

    def __init__(self, predictors, *, position_provider, quantity, max_krw, max_usd,
                 state_path=None, clock=utc_now, trading_mode="demo"):
        if trading_mode not in {"demo", "real"}:
            raise ValueError('Explicit demo or real trading mode required')
        self.trading_mode = trading_mode
        self._prediction_cache = OrderedDict()
        if type(quantity) is not int or not 1 <= quantity <= 999_999_999:
            raise ValueError("quantity must be an explicit positive integer")
        self.predictors = dict(predictors)
        if set(self.predictors) - {"domestic", "us"}:
            raise ValueError("Predictor keys must be domestic/us")
        self.quantity = quantity
        self.caps = {"domestic": number(max_krw, "max_krw", zero=True),
                     "us": number(max_usd, "max_usd", zero=True)}
        for cap in self.caps.values():
            if len(str(cap)) > 32 or cap.adjusted() > 14 or cap.as_tuple().exponent < -8:
                raise ValueError("Notional caps exceed the external JSON numeric bounds")
        if not callable(position_provider):
            raise ValueError("position_provider must be callable")
        self.position_provider, self.clock = position_provider, clock
        self.state_path = Path(state_path) if state_path else None
        self.state = read_json(self.state_path) if self.state_path and self.state_path.exists() else {}
        if not isinstance(self.state, dict):
            raise ValueError("Invalid adapter state file")

    def __call__(self, charts, *, now=None):
        now = now or self.clock()
        if now.tzinfo is None:
            raise ValueError("now requires a timezone")
        if (not isinstance(charts, dict) or type(charts.get("schema_version")) is not int
                or charts["schema_version"] != 1):
            raise ValueError("Expected chart schema_version=1")
        if charts.get("trading_mode") != self.trading_mode or charts.get("source") != f"kiwoom_{self.trading_mode}":
            raise ValueError("Only matching explicit Kiwoom trading-mode chart exports are accepted")
        export_id = charts.get("export_id")
        if not isinstance(export_id, str) or not IDENTIFIER.fullmatch(export_id):
            raise ValueError("Invalid export_id")
        created = timestamp(charts.get("created_at"), "chart created_at")
        digest = hashlib.sha256(json.dumps(charts, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if export_id in self.state:
            old = self.state[export_id]
            if old["chart_digest"] != digest:
                raise ValueError("An existing export_id's contents changed; export a fresh chart")
            return copy.deepcopy(old["payload"]), copy.deepcopy(old["diagnostics"])
        if not 0 <= (now - created).total_seconds() < SIGNAL_TTL:
            raise ValueError("Refresh/export charts first; export must be less than 2 minutes old")
        stocks = charts.get("stocks")
        if not isinstance(stocks, list) or len(stocks) > 500:
            raise ValueError("Expected at most 500 chart instruments")
        signals, diagnostics, seen = [], [], set()
        for stock in stocks:
            key = instrument(stock)
            if key in seen:
                raise ValueError("Duplicate chart instrument")
            seen.add(key)
            # Non-ok rows are not registered chart_export_members in main.
            if stock.get("status") != "ok":
                diagnostics.append({"watch_id": key, "reason": "UNREGISTERED_INVALID_CHART", "emitted": False})
                continue
            decision, detail = self._decision(stock, charts, now)
            signal = {"signal_id": uuid5(NAMESPACE_URL, f"{self.source_id}:{export_id}:{key}").hex,
                      "export_id": export_id,
                      **{field: stock[field] for field in ("market", "symbol", "exchange")},
                      "action": decision["action"], "generated_at": created.isoformat(),
                      "expires_at": (created + timedelta(seconds=SIGNAL_TTL)).isoformat()}
            if decision["action"] != "hold":
                signal.update({key: value for key, value in decision.items() if key not in {"action", "reason"}})
            signals.append(signal)
            diagnostics.append({"watch_id": key, "reason": decision["reason"], "emitted": True, **detail})
        payload = {"schema_version": 1, "source_id": self.source_id, "trading_mode": self.trading_mode, "signals": signals}
        # Only recent immutable decisions are useful. Deleted entries cannot be
        # regenerated: new processing rejects exports older than SIGNAL_TTL.
        self.state = {key: value for key, value in self.state.items()
                      if 0 <= (now - timestamp(value["created_at"])).total_seconds() <= 600}
        self.state[export_id] = {"chart_digest": digest, "payload": payload, "diagnostics": diagnostics}
        self.state[export_id]["created_at"] = created.isoformat()
        # Commit immutable decisions before publication; a crash can only resend the same payload.
        if self.state_path:
            atomic_json(self.state_path, self.state)
        return copy.deepcopy(payload), copy.deepcopy(diagnostics)

    def _decision(self, stock, charts, now):
        detail = {}
        try:
            quote_time = timestamp(stock.get("quote_fetched_at"), "quote_fetched_at")
            declared_age = number(stock.get("quote_age_seconds"), "quote age", zero=True)
            if (stock.get("quote_stale") is not False or declared_age > MAX_QUOTE_AGE
                    or not 0 <= (now - quote_time).total_seconds() <= MAX_QUOTE_AGE):
                raise ValueError("Quote is stale or future-dated")
            price = number(stock.get("price"), "current price")
            position = self.position_provider(stock)
            # A slow balance request must not make an old chart quote actionable.
            checked_at = max(now, self.clock())
            if not 0 <= (checked_at - quote_time).total_seconds() <= MAX_QUOTE_AGE:
                raise ValueError("Quote expired while reading the position")
            qty, sellable, average = validate_position(position, stock, checked_at)
        except Exception as exc:
            return {"action": "hold", "reason": "QUOTE_OR_POSITION_UNAVAILABLE"}, {"error": str(exc)}
        decision = self.position_decision(current_price=price, quantity=qty, sellable_quantity=sellable,
                                          average_price=average)
        if qty == 0:
            try:
                if charts.get("adjusted_prices") is not True:
                    raise ValueError("Model requires adjusted daily OHLCV")
                predictor = self.predictors.get(stock["market"])
                if predictor is None or predictor.metadata.get("market") != stock["market"]:
                    raise ValueError("A matching market checkpoint is required")
                bars = self.input_bars(stock, checked_at)
                key = self.prediction_cache_key(predictor, stock, bars, price)
                cached = self._prediction_cache.get(key) if key is not None else None
                if cached is None:
                    prediction, prediction_detail = self.model_prediction(predictor, bars, price)
                    if key is not None:
                        self._prediction_cache[key] = copy.deepcopy((prediction, prediction_detail))
                        while len(self._prediction_cache) > 512:
                            self._prediction_cache.popitem(last=False)
                else:
                    prediction, prediction_detail = copy.deepcopy(cached)
                    self._prediction_cache.move_to_end(key)
                decision = self.position_decision(current_price=price, quantity=qty,
                                                  sellable_quantity=sellable, prediction=prediction)
                detail["prediction"] = prediction
                detail.update(prediction_detail)
            except Exception as exc:
                return {"action": "hold", "reason": "PREDICTION_UNAVAILABLE"}, {"error": str(exc)}
        if decision["action"] == "hold":
            return decision, detail
        finished_at = max(now, self.clock())
        if not 0 <= (finished_at - quote_time).total_seconds() <= MAX_QUOTE_AGE:
            return {"action": "hold", "reason": "QUOTE_EXPIRED_DURING_INFERENCE"}, detail
        try:
            validate_position(position, stock, finished_at)
        except ValueError as exc:
            return {"action": "hold", "reason": "POSITION_EXPIRED_DURING_INFERENCE"}, {**detail, "error": str(exc)}
        cap = self.caps[stock["market"]]
        if decision["action"] == "sell":
            quantity = min(self.quantity, int(qty), int(sellable), int(cap // price))
        else:
            quantity = self.quantity if price * self.quantity <= cap else 0
        if quantity <= 0:
            return {"action": "hold", "reason": "USER_QUANTITY_OR_NOTIONAL_CAP"}, detail
        return {**decision, "quantity": quantity, "max_notional": str(cap)}, detail


def demo_position_provider(brokers, *, clock=utc_now):
    """Read-only, fail-closed Kiwoom holdings; no quote/order/enable calls.

    The broker's display-oriented normalization substitutes zero for malformed
    quantities. Validate its raw balance pages before treating an absent holding
    as an explicitly verified flat position.
    """
    from dockdack.models import Market, TradingMode
    from dockdack.symbols import normalize_us_exchange

    def provide(stock):
        instrument(stock)
        broker = brokers[stock["market"]]
        if broker.mode is not TradingMode.DEMO:
            raise ValueError("Only demo broker snapshots are allowed")
        market = Market(stock["market"])
        account = (broker.account_domestic(exchange=stock["exchange"]) if market is Market.DOMESTIC
                   else broker.account_us(exchange=stock["exchange"], symbol=stock["symbol"]))
        if account.market is not market or account.currency != stock["currency"]:
            raise ValueError("Account market/currency mismatch")
        raw = getattr(account, "raw", None)
        pages = raw.get("balance") if isinstance(raw, dict) else None
        if not isinstance(pages, list) or not pages:
            raise ValueError("Account raw balance pages are required to verify holdings")
        row_key, qty_key, sell_key, average_key = (
            ("acnt_evlt_remn_indv_tot", "rmnd_qty", "trde_able_qty", "pur_pric")
            if market is Market.DOMESTIC else
            ("result_list", "poss_qty", "sell_alowq", "frgn_stk_book_uv")
        )
        raw_quantity, raw_sellable, raw_cost = Decimal(0), Decimal(0), Decimal(0)
        for page in pages:
            rows = page.get(row_key) if isinstance(page, dict) else None
            if not isinstance(rows, list):
                raise ValueError("Account raw balance list is missing or invalid")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("stk_cd"), str) or not row["stk_cd"].strip():
                    raise ValueError("Account raw balance instrument is missing or invalid")
                qty = number(str(row.get(qty_key)).replace(",", ""), "raw position quantity", zero=True)
                sell = number(str(row.get(sell_key)).replace(",", ""), "raw sellable quantity", zero=True)
                if qty != qty.to_integral_value() or sell != sell.to_integral_value() or sell > qty:
                    raise ValueError("Account raw position/sellable quantities are inconsistent")
                average = (number(str(row.get(average_key)).replace(",", ""), "raw average price")
                           if qty else Decimal(0))
                symbol = row["stk_cd"]
                if market is Market.DOMESTIC:
                    symbol = symbol.strip().upper()
                    if symbol.startswith("A") and symbol[1:].isdigit():
                        symbol = symbol[1:]
                    for suffix in ("_NX", "_AL"):
                        if symbol.endswith(suffix):
                            symbol = symbol[:-len(suffix)]
                if symbol == stock["symbol"]:
                    raw_quantity += qty
                    raw_sellable += sell
                    raw_cost += qty * average
        quantity, sellable, cost = Decimal(0), Decimal(0), Decimal(0)
        for position in account.positions:
            if position.symbol != stock["symbol"]:
                continue
            exchange = normalize_us_exchange(position.exchange)
            if exchange != stock["exchange"]:
                raise ValueError("Matching account position exchange cannot be verified")
            if position.market is not market or position.currency != stock["currency"]:
                raise ValueError("Account position market/currency mismatch")
            qty = number(position.quantity, "position quantity", zero=True)
            sell = number(position.sellable_quantity, "sellable quantity", zero=True)
            if qty != qty.to_integral_value() or sell != sell.to_integral_value() or sell > qty:
                raise ValueError("Account position/sellable quantities are inconsistent")
            average = number(position.average_price, "average price") if qty else Decimal(0)
            quantity += qty
            sellable += sell
            cost += qty * average
        if (quantity, sellable, cost) != (raw_quantity, raw_sellable, raw_cost):
            raise ValueError("Normalized holdings disagree with verified raw account balances")
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": str(quantity), "sellable_quantity": str(sellable),
                "average_price": str(cost / quantity) if quantity else None,
                "fetched_at": clock().isoformat()}

    return provide
