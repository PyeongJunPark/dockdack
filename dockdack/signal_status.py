"""Read-only signal-file inspection; never creates a rule or contacts a broker.

This checks the file contract and configured limits only. The live receiver still
checks export membership, saved IDs and current watchlist state before ingestion.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from dockdack.gui_service import Instrument
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.signal_bridge import (
    ExternalPolicy, _signal_environment, decimal_string, identifier, read_signal_file, timestamp, validate_exit_conditions,
)
from dockdack.watchlist import TriggerKind, TriggerRule, instrument_key, utc_now


INSPECTION_SCOPE = "파일 형식·설정 상한만 검사합니다. 신호 접수·규칙 생성·주문 전송은 하지 않습니다."


def inspect_signal_file(path: Path | str, policy: ExternalPolicy, *, now: datetime | None = None,
                        mode: TradingMode = TradingMode.DEMO) -> dict:
    """Bounded file read for a background worker, not the GUI event loop.

    A successful result is deliberately called 'format_ok', not 'connected' or
    'accepted'. No store is accepted by this interface, so even a BUY file cannot
    create an executable candidate as a side effect of inspecting it.
    """
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError("검사 시각에 시간대가 필요합니다.")
    result = {"state": "error", "checked_at": now, "summary": "", "scope": INSPECTION_SCOPE,
              "counts": {"buy": 0, "sell": 0, "hold": 0, "expired": 0}}
    try:
        payload = read_signal_file(path)
        if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise ValueError("신호 schema_version은 1이어야 합니다.")
        if set(payload) - {"schema_version", "source_id", "signals", "trading_mode"} or not {"schema_version", "source_id", "signals"}.issubset(payload):
            raise ValueError("신호 최상위 필드는 schema_version, source_id, signals와 선택적 trading_mode만 허용합니다.")
        source = identifier(payload["source_id"], "source_id")
        if "trading_mode" in payload and (not isinstance(payload["trading_mode"], str) or payload["trading_mode"] not in {"demo", "real"}):
            raise ValueError("trading_mode는 demo 또는 real이어야 합니다.")
        environment = SimpleNamespace(mode=TradingMode(mode))
        _signal_environment(environment, source, payload)
        if source != policy.source_id:
            raise ValueError("허용된 source_id가 아닙니다.")
        entries = payload["signals"]
        if not isinstance(entries, list) or len(entries) > 500:
            raise ValueError("signals는 최대 500개인 배열이어야 합니다.")
        seen_watch, seen_signal = set(), set()
        for entry in entries:
            required = {"signal_id", "export_id", "market", "symbol", "exchange", "action", "generated_at", "expires_at"}
            optional = {"quantity", "max_notional", "order_type", "min_sell_price", "cost_profit_pct", "cost_loss_pct"}
            if not isinstance(entry, dict) or not required.issubset(entry) or set(entry) - required - optional:
                raise ValueError("외부 신호의 필수/허용 필드를 확인하세요.")
            sid = identifier(entry["signal_id"], "signal_id")
            identifier(entry["export_id"], "export_id")
            instrument = Instrument(Market(entry["market"]), entry["symbol"], entry["exchange"])
            watch_id = instrument_key(instrument)
            _signal_environment(environment, source, {**entry, "trading_mode": payload.get("trading_mode")}, watch_id=watch_id)
            if watch_id in seen_watch or sid in seen_signal:
                raise ValueError("한 파일에는 종목별 신호를 하나만, 서로 다른 signal_id로 넣으세요.")
            seen_watch.add(watch_id)
            seen_signal.add(sid)
            action = entry["action"]
            if action not in {"buy", "sell", "hold"}:
                raise ValueError("action은 buy, sell, hold 중 하나여야 합니다.")
            created = timestamp(entry["generated_at"], "generated_at")
            expiry = timestamp(entry["expires_at"], "expires_at")
            if not created < expiry <= created + timedelta(minutes=10):
                raise ValueError("신호 유효기간은 생성 이후 최대 10분입니다.")
            if created > now + timedelta(seconds=5):
                raise ValueError("미래에 생성된 신호는 사용할 수 없습니다.")
            if action != "hold":
                quantity = entry.get("quantity")
                amount = decimal_string(entry.get("max_notional"), "max_notional")
                TriggerRule("inspection-only", watch_id, TriggerKind.EXTERNAL, OrderSide(action), quantity, amount)
                policy.validate(source, instrument.market, quantity, amount, entry.get("order_type", "limit"))
                validate_exit_conditions(entry)
            elif any(key in entry for key in optional):
                raise ValueError("hold 신호에는 수량/주문금액을 넣지 마세요.")
            result["counts"][action] += 1
            if expiry <= now or now - created > timedelta(minutes=5):
                result["counts"]["expired"] += 1
        result.update(state="format_ok", source_id=source,
                      summary=f"형식 검사 통과 · {len(entries)}개 신호 · 아직 접수/주문하지 않았습니다.")
    except FileNotFoundError:
        result.update(state="missing", summary="입력 파일 없음 · 신호기가 아직 발행하지 않았거나 경로가 다릅니다.")
    except (OSError, TypeError, ValueError, AttributeError, OverflowError, RecursionError) as exc:
        result["summary"] = f"검사 실패 · {exc}"
    return result
