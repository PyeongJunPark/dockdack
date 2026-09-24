"""Render existing provenance widgets with invented rows; no DB, broker or network."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from dockdack.activity_snapshot import LedgerSnapshot
from dockdack.execution_policy import holding_exit_targets
from dockdack.gui import STYLE
from dockdack.models import AccountSnapshot, Market, Position, TradingMode
from dockdack.operations_gui import OrderHistoryPanel
from dockdack.portfolio import PortfolioMarketState
from dockdack.portfolio_gui import PortfolioPanel
from dockdack.signal_bridge import prototype_family
from dockdack.trade_journal import daily_trade_journal


def fixture(index, source=None, *, mismatch=False):
    now = datetime.now(timezone.utc) - timedelta(minutes=5 - index)
    symbol, name = f"90000{index}", f"가상종목 {chr(64 + index)}"
    key = f"domestic:KRX:{symbol}"
    price = D("10000")
    position = Position(Market.DOMESTIC, symbol, name, "KRX", "KRW", D(1), D(1), price,
                        price, price, D(0), D(0))
    row = {"rule_id": f"fixture-{index}", "watch_id": key, "started_at": now.isoformat(),
           "status": "filled", "order_number": f"FAKE-{index}", "market": "domestic",
           "exchange": "KRX", "symbol": symbol, "name": name, "currency": "KRW", "side": "buy",
           "quantity": 1, "filled_quantity": "1", "fill_price": str(price), "remaining_quantity": "0",
           "observed_at": now.isoformat(), "reference_price": str(price), "message": "가상 화면 검증 행"}
    family = prototype_family(source)
    if family is None:
        return position, row, None, None
    signal_id = ("mark1-1-prototype" if mismatch else family.id) + f":preview-{index}"
    payload = {"signal_id": signal_id, "market": "domestic", "exchange": "KRX", "symbol": symbol,
               "action": "buy", "trading_mode": "demo"}
    record = {"source_id": source, "signal_id": signal_id, "watch_id": key, "decision": "buy",
              "payload": json.dumps(payload), "rule_id": row["rule_id"]}
    row.update(external_source_id=source, external_signal_id=signal_id, external_watch_id=key,
               external_decision="buy", external_payload=record["payload"])
    saved = {"rule_id": row["rule_id"], "source": source, "take_profit_price": price * (1 + family.take_profit),
             "stop_loss_price": price * (1 - family.stop_loss)}
    return position, row, record, saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "outputs/mark1/dual-prototype-provenance-preview-20260924.png")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([])
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/malgun.ttf")
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError("Korean preview font could not be loaded")
    app.setFont(QFont(families[0], 10))
    data = [fixture(1, "mark1-prototype-demo-trigger"), fixture(2, "mark1-1-prototype-demo-trigger"),
            fixture(3), fixture(4, "mark1-prototype-demo-trigger", mismatch=True)]
    records = {row["watch_id"]: (record, saved) for _, row, record, saved in data}
    store = SimpleNamespace(mode=TradingMode.DEMO,
                            exit_targets=lambda key: records[key][1],
                            external_for_rule=lambda rule: next(record for record, _ in records.values()
                                                                if record and record["rule_id"] == rule))
    targets = {}
    for position, row, _, _ in data[:3]:
        targets[row["watch_id"]] = holding_exit_targets(store, position)
    root = QWidget()
    root.setObjectName("offlineProof")
    root.setStyleSheet(STYLE + f"\nQWidget {{ font-family: '{families[0]}'; }}\n"
                      "QWidget#offlineProof { background: #0f1724; }\n"
                      "QHeaderView { background: #1b293a; }")
    layout = QVBoxLayout(root)
    heading = QLabel("똑딱 · 모델별 매수 출처 표시 검증")
    heading.setStyleSheet("font-size: 24px; font-weight: 700; color: #e7edf8;")
    layout.addWidget(heading)
    note = QLabel("전부 가상 종목·체결입니다. 실제 계좌 / API / DB 접속 없음 · 감시 및 주문 시작 안 함")
    note.setStyleSheet("background: #203c57; color: #d9edff; padding: 10px; font-size: 14px;")
    layout.addWidget(note)
    holdings = PortfolioPanel()
    holdings.heading.setText("현재 보유종목 · 가상계좌 (원래 매수 모델 기준)")
    holdings.set_exit_targets(targets)
    current = datetime.now(timezone.utc)
    account = AccountSnapshot(Market.DOMESTIC, "KRW", tuple(row[0] for row in data[:3]),
                              cash=D("1000000"), available_to_order=D("1000000"))
    holdings.apply({Market.DOMESTIC: PortfolioMarketState(Market.DOMESTIC, account, current, current)}, now=current)
    layout.addWidget(holdings, 1)
    orders = OrderHistoryPanel()
    for column, width in ((3, 80), (5, 90), (13, 190)):
        orders.table.setColumnWidth(column, width)
    orders.heading.setText("주문·체결 내역 · 가상 검증 장부")
    ledger = tuple(row[1] for row in data)
    orders.apply_snapshot(LedgerSnapshot("offline-preview", ledger, daily_trade_journal(ledger)))
    layout.addWidget(orders, 1)
    footer = QLabel("A: mark1 (+1% / -0.9%)   ·   B: mark1.1 (+0.5% / -0.4%)   ·   C: 출처 미확인   ·   D: 불일치 감지")
    footer.setStyleSheet("color: #9cacc1; padding: 6px;")
    layout.addWidget(footer)
    root.resize(1600, 1280)
    with patch("requests.sessions.Session.request", side_effect=AssertionError("Offline preview: no network")) as network:
        root.show()
        app.processEvents()
        app.processEvents()
        assert holdings.table.item(0, 11).text() == "mark1 prototype"
        assert holdings.table.item(1, 11).text() == "mark1.1 prototype"
        assert "미확인" in holdings.table.item(2, 11).text()
        assert "불일치" in orders.table.item(0, 13).text()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not root.grab().save(str(args.output)):
            raise RuntimeError("Could not save offscreen preview")
        network.assert_not_called()
        root.close()
    print(str(args.output))


if __name__ == "__main__":
    main()
