"""Versioned JSON chart export and data-only, idempotent external-signal ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from dockdack.gui_service import Instrument
from dockdack.history import market_time
from dockdack.models import Market, OrderSide, TradingMode
from dockdack.watchlist import TriggerKind, TriggerRule, WatchItem, WatchStore, instrument_key, positive, utc_now


MAX_SIGNAL_BYTES = 2_000_000
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _signal_environment(store, source, metadata, *, watch_id=None):
    """Data-only mode provenance, not an authentication or strategy endorsement."""
    mode = TradingMode(getattr(store, "mode", TradingMode.DEMO))
    declared = metadata.get("trading_mode")
    if declared is not None and declared != mode.value:
        raise ValueError("매매 신호의 trading_mode가 현재 선택한 모의/실전 환경과 다릅니다.")
    if mode is TradingMode.REAL:
        if declared != "real":
            raise ValueError("실전 매매 신호에는 최상위 trading_mode='real' 명시가 필요합니다.")
        if str(source).strip().lower().replace("_", "-") == "random-demo":
            raise ValueError("내장 모의 테스트 신호(random-demo)는 실전에서 수신할 수 없습니다.")
        if watch_id is not None:
            test_id = uuid5(NAMESPACE_URL, f"random-demo:{metadata.get('export_id')}:{watch_id}").hex
            if metadata.get("signal_id") == test_id:
                raise ValueError("내장 모의 테스트 신호는 출처 이름을 바꿔도 실전에 사용할 수 없습니다.")


def identifier(value, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label}은 영문/숫자와 . _ : - 조합의 1~128자여야 합니다.")
    return value


def timestamp(value, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label}은 시간대가 포함된 ISO 8601 시각이어야 합니다.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label}에 시간대가 없습니다.")
    return parsed.astimezone(timezone.utc)


def decimal_string(value, label: str) -> Decimal:
    if not isinstance(value, str) or len(value) > 32:
        raise ValueError(f"{label}은 숫자 문자열로 전달하세요.")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} 숫자 형식이 잘못되었습니다.") from exc
    positive(result, label)
    if result.adjusted() > 14 or result.as_tuple().exponent < -8:
        raise ValueError(f"{label}의 크기/소수 자릿수가 범위를 넘습니다.")
    return result


def validate_exit_conditions(entry: dict):
    """Optional data-only SELL gates shared by ingestion and no-order inspection."""
    action = entry["action"]
    target_keys = {"take_profit_price", "stop_loss_price"}
    if target_keys.intersection(entry):
        if action != "buy" or not target_keys.issubset(entry):
            raise ValueError("매수 신호의 상방·하방 목표가격을 함께 전달하세요.")
        upper = decimal_string(entry["take_profit_price"], "take_profit_price")
        lower = decimal_string(entry["stop_loss_price"], "stop_loss_price")
        if lower >= upper:
            raise ValueError("하방 목표가격은 상방 목표가격보다 낮아야 합니다.")
    if "min_sell_price" in entry:
        if action != "sell":
            raise ValueError("min_sell_price는 매도에만 사용할 수 있습니다.")
        decimal_string(entry["min_sell_price"], "min_sell_price")
    if "cost_profit_pct" in entry:
        if action != "sell" or decimal_string(entry["cost_profit_pct"], "cost_profit_pct") > 100:
            raise ValueError("cost_profit_pct는 매도의 0 초과 100 이하 비율만 허용합니다.")
    if "cost_loss_pct" in entry:
        if action != "sell" or decimal_string(entry["cost_loss_pct"], "cost_loss_pct") >= 100:
            raise ValueError("cost_loss_pct는 매도의 0 초과 100 미만 손실 비율만 허용합니다.")
        if "cost_profit_pct" in entry or "min_sell_price" in entry:
            raise ValueError("손절 조건에 익절 조건 또는 최소 매도가를 함께 넣지 마세요.")


@dataclass(frozen=True)
class ExternalPolicy:
    source_id: str
    max_quantity: int
    max_krw: Decimal
    max_usd: Decimal
    allow_market: bool = False

    def __post_init__(self):
        if type(self.allow_market) is not bool:
            raise ValueError("시장가 허용 값은 bool이어야 합니다.")
        identifier(self.source_id, "source_id")
        if type(self.max_quantity) is not int or not 1 <= self.max_quantity <= 999_999_999:
            raise ValueError("외부 신호 최대 수량은 1 이상의 정수여야 합니다.")
        for cap in (self.max_krw, self.max_usd):
            if not isinstance(cap, Decimal) or not cap.is_finite() or cap < 0:
                raise ValueError("외부 신호 금액 상한은 0 이상의 유한한 숫자여야 합니다. 0이면 해당 시장 주문을 차단합니다.")

    def cap(self, market: Market) -> Decimal:
        return self.max_krw if market is Market.DOMESTIC else self.max_usd

    def validate(self, source_id: str, market: Market, quantity: int, notional: Decimal, order_type="limit"):
        if source_id != self.source_id:
            raise ValueError("허용된 외부 신호 source_id와 다릅니다.")
        if quantity > self.max_quantity or notional > self.cap(market) or self.cap(market) <= 0:
            raise ValueError("외부 신호가 사용자가 설정한 수량/시장별 금액 상한을 넘습니다.")
        if order_type not in {"limit", "market"} or (order_type == "market" and (not self.allow_market or market is Market.US)):
            raise ValueError("외부 신호의 시장가 주문은 명시적으로 허용한 국내 주문만 지원합니다.")


def atomic_json(path: Path | str, payload: dict):
    """Publish a complete file in one replacement, without exposing half-written JSON."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".dockdack-json-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export_charts(store: WatchStore, path: Path | str, *, now: datetime | None = None,
                  errors: dict[str, str] | None = None, watch_ids: set[str] | None = None) -> dict:
    if Path(path).suffix.lower() != ".json" or Path(path).resolve() == store.path:
        raise ValueError("차트 출력은 별도의 .json 파일이어야 합니다.")
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError("내보내기 시각에 시간대가 필요합니다.")
    rows, members = [], []
    for stored in store.chart_export_rows(watch_ids):
        item = WatchItem(Instrument(Market(stored["market"]), stored["symbol"], stored["exchange"]),
                         stored["name"], stored["days"])
        inst = item.instrument
        result = {"watch_id": item.id, "market": inst.market.value, "symbol": inst.symbol,
                  "exchange": inst.exchange, "currency": inst.currency, "name": item.name,
                  "requested_days": item.days, "status": "missing", "bars": []}
        if stored["turnover_rank"] is not None:
            result["turnover_rank"] = stored["turnover_rank"]
            result["ranking_basis"] = stored["ranking_basis"]
            result["volume_rank"] = stored["turnover_rank"] if stored["ranking_basis"] == "volume" else None
            result["ranked_volume"] = stored["volume"]
            result["turnover"] = stored["turnover"]
            result["ranking_fetched_at"] = stored["ranking_fetched_at"]
        try:
            if errors and item.id in errors:
                raise ValueError(errors[item.id])
            data = store.decode_snapshot(item, stored["snapshot_data"]) if stored["snapshot_data"] is not None else None
            if data:
                from dockdack.autotrade import AutoTrader
                AutoTrader._validate_snapshot(item, data)
                bars = data.history.bars[-item.days:]
                today = market_time(inst.market, now).date()
                age = (now - data.fetched_at).total_seconds()
                if age < 0:
                    raise ValueError("저장된 조회 시각이 미래입니다.")
                if age > 86400:
                    raise ValueError("저장된 조회값이 24시간 이상 오래되었습니다. 재조회하세요.")
                result.update(status="ok", price=str(data.quote.price), quote_fetched_at=data.fetched_at.isoformat(),
                              history_fetched_at=stored["history_fetched_at"],
                              quote_age_seconds=round(age, 3), quote_stale=age > 15,
                              available_days=len(bars), complete=len(bars) >= item.days,
                              bars=[{"date": bar.day.isoformat(), **{key: str(getattr(bar, key)) for key in ("open", "high", "low", "close", "volume")},
                                     "is_current_day": bar.day == today} for bar in bars])
                members.append(item.id)
        except Exception as exc:
            result.update(status="error", error=str(exc))
        rows.append(result)
    mode = TradingMode(getattr(store, "mode", TradingMode.DEMO))
    payload = {"schema_version": 1, "export_id": uuid4().hex, "created_at": now.isoformat(),
               "source": f"kiwoom_{mode.value}", "trading_mode": mode.value,
               "adjusted_prices": True, "history_cache_max_seconds": 300,
               "completed_bars_persisted": True, "intraday_history_refresh_seconds": 300,
               "scope": "watchlist" if watch_ids is None else "instrument",
               "quote_time_is_fetch_time": True, "stocks": rows}
    atomic_json(path, payload)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    with store.connection() as db:
        db.execute("INSERT INTO chart_exports VALUES (?, ?, ?, ?)",
                   (payload["export_id"], now.isoformat(), str(Path(path).resolve()), hashlib.sha256(encoded).hexdigest()))
        db.executemany("INSERT INTO chart_export_members VALUES (?, ?)", [(payload["export_id"], key) for key in members])
        if watch_ids is None:
            store._insert_event(db, "SYSTEM", f"차트 JSON 내보내기 · 유효 {len(members)}/{len(rows)}종목 · {payload['export_id']}", category="monitor")
        else:
            for row in rows:
                store._insert_event(db, row["watch_id"],
                                    f"차트 JSON 내보내기 · {'전송 파일 저장 완료' if row['status'] == 'ok' else '유효 데이터 없음'} · {payload['export_id']}",
                                    category="monitor")
    return payload


def read_signal_file(path: Path | str) -> dict:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SIGNAL_BYTES + 1)
    if len(raw) > MAX_SIGNAL_BYTES:
        raise ValueError("신호 파일은 2MB 이하여야 합니다.")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("신호 JSON에 중복 필드가 있습니다.")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"JSON에 {value} 상수를 사용할 수 없습니다.")
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("신호 파일은 완성된 UTF-8 JSON이어야 합니다. 임시 파일을 원자적으로 교체하세요.") from exc


def ingest_signals(store: WatchStore, payload: dict, policy: ExternalPolicy, *, now: datetime | None = None) -> dict:
    """All-or-nothing ingestion. No Python code is loaded and no order is sent here."""
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError("신호 수신 시각에 시간대가 필요합니다.")
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("신호 schema_version은 1이어야 합니다.")
    if set(payload) - {"schema_version", "source_id", "signals", "trading_mode"} or not {"schema_version", "source_id", "signals"}.issubset(payload):
        raise ValueError("신호 최상위 필드는 schema_version, source_id, signals와 선택적 trading_mode만 허용합니다.")
    source = identifier(payload.get("source_id"), "source_id")
    if "trading_mode" in payload and (not isinstance(payload["trading_mode"], str) or payload["trading_mode"] not in {"demo", "real"}):
        raise ValueError("trading_mode는 demo 또는 real이어야 합니다.")
    _signal_environment(store, source, payload)
    if source != policy.source_id:
        raise ValueError("허용된 source_id가 아닙니다.")
    entries = payload.get("signals")
    if not isinstance(entries, list) or len(entries) > 500:
        raise ValueError("signals는 최대 500개인 배열이어야 합니다.")
    parsed, seen = [], set()
    for entry in entries:
        required = {"signal_id", "export_id", "market", "symbol", "exchange", "action", "generated_at", "expires_at"}
        if not isinstance(entry, dict) or not required.issubset(entry) or set(entry) - required - {"quantity", "max_notional", "order_type", "min_sell_price", "cost_profit_pct", "cost_loss_pct", "take_profit_price", "stop_loss_price"}:
            raise ValueError("외부 신호의 필수/허용 필드를 확인하세요.")
        sid, export_id = identifier(entry["signal_id"], "signal_id"), identifier(entry["export_id"], "export_id")
        inst = Instrument(Market(entry["market"]), entry["symbol"], entry["exchange"])
        watch_id = instrument_key(inst)
        _signal_environment(store, source, {**entry, "trading_mode": payload.get("trading_mode")}, watch_id=watch_id)
        if watch_id in seen:
            raise ValueError("한 파일에는 종목별 신호를 하나만 넣으세요.")
        seen.add(watch_id)
        action = entry["action"]
        if action not in {"buy", "sell", "hold"}:
            raise ValueError("action은 buy, sell, hold 중 하나여야 합니다.")
        created, expiry = timestamp(entry["generated_at"], "generated_at"), timestamp(entry["expires_at"], "expires_at")
        if not created < expiry <= created + timedelta(minutes=10):
            raise ValueError("신호 유효기간은 생성 이후 최대 10분입니다.")
        if created > now + timedelta(seconds=5):
            raise ValueError("미래에 생성된 신호는 사용할 수 없습니다.")
        rule = None
        if action != "hold":
            qty = entry.get("quantity")
            amount = decimal_string(entry.get("max_notional"), "max_notional")
            rule = TriggerRule(uuid4().hex, watch_id, TriggerKind.EXTERNAL, OrderSide(action), qty, amount)
            policy.validate(source, inst.market, qty, amount, entry.get("order_type", "limit"))
            validate_exit_conditions(entry)
        elif any(key in entry for key in ("quantity", "max_notional", "order_type", "min_sell_price", "cost_profit_pct", "cost_loss_pct", "take_profit_price", "stop_loss_price")):
            raise ValueError("hold 신호에는 수량/주문금액을 넣지 마세요.")
        # Persist the explicit REAL declaration so a later restart/final paced
        # send validates the same environment, not just an editable GUI label.
        canonical_entry = {**entry, "trading_mode": "real"} if getattr(store, "mode", TradingMode.DEMO) is TradingMode.REAL else entry
        canonical = json.dumps(canonical_entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        parsed.append((entry, sid, export_id, watch_id, created, expiry, rule, canonical))
    result = {"queued": 0, "duplicates": 0, "hold": 0, "expired": 0}
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for entry, sid, export_id, watch_id, created, expiry, rule, canonical in parsed:
            old = db.execute("SELECT payload FROM external_signals WHERE source_id=? AND signal_id=?", (source, sid)).fetchone()
            if old:
                if old[0] != canonical:
                    raise ValueError(f"이미 받은 signal_id의 내용이 바뀌었습니다: {sid}")
                result["duplicates"] += 1
                continue
            export = db.execute("SELECT created_at FROM chart_exports WHERE id=?", (export_id,)).fetchone()
            member = db.execute("SELECT 1 FROM chart_export_members WHERE export_id=? AND watch_id=?", (export_id, watch_id)).fetchone()
            if not export or not member:
                raise ValueError("알려진 차트 내보내기 또는 내보낸 종목이 아닙니다.")
            if not db.execute("SELECT 1 FROM watchlist WHERE id=? AND active=1", (watch_id,)).fetchone():
                raise ValueError("현재 관심종목에 없는 신호입니다.")
            exported = timestamp(export[0], "export created_at")
            if exported > created + timedelta(seconds=5):
                raise ValueError("차트 내보내기보다 먼저 생성된 신호입니다.")
            expired = expiry <= now or now - created > timedelta(minutes=5) or now - exported > timedelta(hours=24)
            status = "expired" if expired else ("hold" if rule is None else "queued")
            # Older decisions must not undo a newer decision even when a new ID is used.
            latest = db.execute("SELECT generated_at FROM external_signals WHERE source_id=? AND watch_id=? ORDER BY generated_at DESC LIMIT 1",
                                (source, watch_id)).fetchone()
            if latest and created <= timestamp(latest[0], "previous generated_at"):
                status, expired = "expired", True
            if not expired:
                db.execute("""UPDATE external_signals SET status='superseded' WHERE source_id=? AND watch_id=?
                           AND rule_id IN (SELECT id FROM rules WHERE status='ready')""", (source, watch_id))
                db.execute("""UPDATE rules SET status='superseded' WHERE status='ready' AND id IN
                           (SELECT rule_id FROM external_signals WHERE source_id=? AND watch_id=?)""", (source, watch_id))
            if rule is not None and not expired:
                db.execute("INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready')",
                           (rule.id, watch_id, rule.kind.value, rule.side.value, rule.quantity, str(rule.max_notional), None, rule.period))
            db.execute("INSERT INTO external_signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (source, sid, canonical, rule.id if rule is not None and not expired else None, watch_id,
                        created.isoformat(), expiry.isoformat(), export_id, now.isoformat(), entry["action"], status))
            result["expired" if expired else "hold" if rule is None else "queued"] += 1
            decision = {"hold": "HOLD · 매매하지 않음", "buy": "BUY · 매수 후보", "sell": "SELL · 매도 후보"}[entry["action"]]
            if "cost_profit_pct" in entry:
                decision += f" · 익절 +{entry['cost_profit_pct']}%"
            elif "cost_loss_pct" in entry:
                decision += f" · 손절 -{entry['cost_loss_pct']}%"
            detail = "유효기간 경과/이전 신호 · 실행 대상 아님" if expired else (
                "주문 생성 없음" if rule is None else "조건 확인 대기 · 주문 접수/체결 아님")
            store._insert_event(db, watch_id, f"외부 신호 수신 · {source} · {decision} · {detail}",
                                category="signal", at=now)
    return result


def validate_external_rule(store: WatchStore, rule: TriggerRule, policy: ExternalPolicy | None, now: datetime):
    record = store.external_for_rule(rule.id)
    if policy is None or not record:
        raise ValueError("외부 신호 정책 또는 수신 기록이 없습니다.")
    with store.connection() as db:
        latest = db.execute("SELECT signal_id FROM external_signals WHERE source_id=? AND watch_id=? AND status!='expired' ORDER BY generated_at DESC LIMIT 1",
                            (record["source_id"], record["watch_id"])).fetchone()
    if not latest or latest[0] != record["signal_id"]:
        raise ValueError("동일 출처의 더 최근 매매/HOLD 신호가 도착해 이전 신호를 사용하지 않습니다.")
    if timestamp(record["expires_at"], "expires_at") <= now:
        store.expire_external(now)
        raise ValueError("외부 매매 신호가 만료되었습니다.")
    metadata = json.loads(record["payload"])
    _signal_environment(store, record["source_id"], metadata, watch_id=record["watch_id"])
    policy.validate(record["source_id"], Market(record["watch_id"].split(":")[0]), rule.quantity, rule.max_notional, metadata.get("order_type", "limit"))
    validate_exit_conditions(metadata)
    return metadata


class SignalFileReader:
    def __init__(self, store: WatchStore, path: Path | str, policy: ExternalPolicy, clock=utc_now):
        self.store, self.path, self.policy, self.clock = store, Path(path).resolve(), policy, clock
        self.signature = None
        self.reader_state = "waiting"
        self.reader_error = ""
        self.last_read_at = None
        self.last_accepted_at = None
        self.received_counts = None

    def status(self):
        """Cached connection telemetry only; does not stat/read the file or query SQLite."""
        return {"reader_state": self.reader_state, "reader_error": self.reader_error,
                "last_read_at": self.last_read_at, "last_accepted_at": self.last_accepted_at,
                "received_counts": dict(self.received_counts) if self.received_counts is not None else None}

    def __call__(self):
        try:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self.reader_state, self.reader_error = "missing", ""
                return None  # A producer may not have published its first signal file yet.
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self.signature:
                self.reader_state, self.reader_error = "unchanged", ""
                return None
            payload = read_signal_file(self.path)
            self.last_read_at = self.clock()
            result = ingest_signals(self.store, payload, self.policy, now=self.last_read_at)
            self.signature = signature
            self.last_accepted_at = self.clock()
            self.received_counts = dict(result)
            self.reader_state, self.reader_error = "ok", ""
            return result
        except Exception as exc:
            self.reader_state, self.reader_error = "error", str(exc) or type(exc).__name__
            raise  # The engine isolates this source; another producer may remain healthy.
