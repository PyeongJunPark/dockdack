"""Read-only operational views. Signal receipts are never treated as orders."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QPushButton, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.gui import label, number, table
from dockdack.watchlist import STATUS_LABELS
from dockdack.performance import realized_performance
from dockdack.activity_snapshot import LedgerCollector, MAX_VISIBLE_ROWS, collect_event_logs


def local_time(value):
    if not value:
        return "—"
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return timestamp.astimezone().strftime("%m/%d %H:%M:%S")


def populate(widget, rows, keys):
    """Avoid repainting unchanged history or dragging a reader back to the top."""
    signature = (tuple(tuple(row) for row in rows), tuple(keys))
    if getattr(widget, "_contents", None) == signature:
        return False
    widget._contents = signature
    scrollbar = widget.verticalScrollBar()
    old_scroll = scrollbar.value()
    first = widget.item(widget.rowAt(0), 0)
    anchor = first.data(Qt.ItemDataRole.UserRole) if first and old_scroll else None
    enabled = widget.updatesEnabled()
    widget.setUpdatesEnabled(False)
    try:
        widget.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                text = str(value)
                cell = widget.item(row, column)
                if cell is None:
                    cell = QTableWidgetItem(text)
                    widget.setItem(row, column, cell)
                elif cell.text() != text:
                    cell.setText(text)
                cell.setToolTip(text)
                cell.setData(Qt.ItemDataRole.ForegroundRole, None)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, keys[row])
        if anchor in keys:
            widget.scrollToItem(widget.item(keys.index(anchor), 0), widget.ScrollHint.PositionAtTop)
        else:
            scrollbar.setValue(old_scroll)
    finally:
        widget.setUpdatesEnabled(enabled)
    return True


class EventLog(QWidget):
    def __init__(self, category, description, parent=None):
        super().__init__(parent)
        self.category = category
        layout = QVBoxLayout(self)
        layout.addWidget(label(description, "muted", wrap=True))
        self.count_label = label("", "muted")
        layout.addWidget(self.count_label)
        self.table = table(["시각 (로컬)", "종목 / 구분", "기록"])
        for column in (0, 1):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self.latest = None
        self._head = None

    def reload(self, store, *, head=None, force=False):
        if not force and head is not None and head == self._head:
            return
        events = store.events(limit=500, category=self.category)
        self.apply_events(events, head=head)

    def apply_events(self, events, *, head=None):
        """Qt-only bounded paint; collection has already completed elsewhere."""
        events = events[:MAX_VISIBLE_ROWS]
        self.latest = events[0]["time"] if events else None
        populate(self.table, [(local_time(r["time"]), r["symbol"].split(":")[-1], r["message"])
                              for r in events], [r["id"] for r in events])
        self.count_label.setText(f"로컬 저장 기록 · 최근 {len(events)}건 (최대 500건)" if events else "아직 이 분류의 기록이 없습니다.")
        self._head = head


class OperationsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(label("서버 상태와 데이터 흐름", "section"))
        self.runtime = label("감시 중지 · API 상태 미확인", "muted", wrap=True)
        self.flow = label("최근 감시 기록 — · 최근 신호 수신 —", "muted", wrap=True)
        layout.addWidget(self.runtime)
        layout.addWidget(self.flow)
        self.tabs = QTabWidget()
        self.logs = {}
        for category, title, text in (
            ("system", "서버·오류", "시작·중지, 순회 상태, 자동주문 허용 상태와 운영 오류입니다. 앱 응답 표시만으로 API 정상 여부를 보장하지 않습니다."),
            ("monitor", "시세·차트 감시", "종목별 조회 성공·실패, 일봉 캐시와 차트 전달 기록입니다. 매수·매도 내역이 아닙니다."),
            ("signal", "매매 신호 (주문 아님)", "BUY / SELL은 매매 제안이며 주문·체결이 아닙니다. HOLD는 매매하지 않음입니다. 자동주문 OFF여도 신호 수신은 계속됩니다."),
        ):
            view = EventLog(category, text)
            self.logs[category] = view
            self.tabs.addTab(view, title)
        layout.addWidget(self.tabs, 1)

    def reload(self, store, *, visible_only=False, force=False):
        categories = (self.tabs.currentWidget().category,) if visible_only else tuple(self.logs)
        self.apply_logs(collect_event_logs(store, categories,
            previous_heads={key: view._head for key, view in self.logs.items()}, force=force))

    def apply_logs(self, payload):
        """Apply worker-collected events without DB reads or a full-log rebuild."""
        heads = payload["heads"]
        for view in self.logs.values():
            if view.category in payload["events"]:
                view.apply_events(payload["events"][view.category], head=heads[view.category])
            view.latest = heads[view.category]["time"]
        self.flow.setText(f"최근 감시 기록 {local_time(self.logs['monitor'].latest)}  ·  최근 신호 기록 {local_time(self.logs['signal'].latest)}")


class OrderHistoryPanel(QWidget):
    """Durable local order attempts, not inferred from human log messages."""

    request_refresh = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = ()
        self._ledger = ()
        self.performance = realized_performance(())
        self._collector = None
        self._applied_snapshot = None
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        self.heading = label("실제 주문·체결 내역 · 모의계좌", "section")
        heading.addWidget(self.heading, 1)
        self.refresh_button = QPushButton("체결가 다시 확인")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.setToolTip("증권사 체결 내역을 조회해 누락된 가격을 보완합니다. 주문하지 않습니다. 재조회는 최소 60초 간격입니다.")
        self.refresh_button.clicked.connect(self.request_refresh.emit)
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
        layout.addWidget(label("이 앱이 기록한 자동·수동 주문을 표시합니다. HOLD·신호는 제외하며 ‘접수’는 체결이 아닙니다. 체결가는 증권사 확인값만 사용합니다. 기록 시작 전이나 다른 앱의 주문은 포함되지 않습니다.", "muted", wrap=True))
        self.summary = label("주문 기록 없음", "section", wrap=True)
        layout.addWidget(self.summary)
        self.performance_label = label("실현손익 미확인", "section", wrap=True)
        layout.addWidget(self.performance_label)
        self.recovery_label = label("체결가 확인 전 · 현재가나 주문가로 체결가를 대신 채우지 않습니다.", "muted", wrap=True)
        layout.addWidget(self.recovery_label)
        self.tabs = tabs = QTabWidget()
        records_box = QWidget()
        records_layout = QVBoxLayout(records_box)
        filter_row = QHBoxLayout()
        self.filter = QComboBox()
        for text, key in (("전체 주문", "all"), ("체결 확인 (부분체결 포함)", "filled"),
                          ("접수·확인 대기", "pending"), ("거절·취소·미전송·수동 확인", "other")):
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(self.render)
        filter_row.addWidget(self.filter)
        self.count_label = label("", "muted")
        filter_row.addWidget(self.count_label, 1)
        records_layout.addLayout(filter_row)
        self.table = table(["요청시각 (로컬)", "종목", "시장", "매수/매도", "주문 수량", "체결 수량",
                            "잔량", "상태", "실제 체결가", "주문번호", "매도 대응 원가", "실현손익", "실현 수익률"])
        self.table.verticalHeader().setDefaultSectionSize(54)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in ((0, 126), (1, 150), (3, 54), (5, 58), (7, 114), (8, 118), (10, 120), (11, 115), (12, 87)):
            self.table.setColumnWidth(column, width)
        for column in (2, 4, 6, 9):
            self.table.setColumnHidden(column, True)
        for visual, logical in enumerate((0, 1, 3, 8, 5, 10, 11, 12, 7, 2, 4, 6, 9)):
            header.moveSection(header.visualIndex(logical), visual)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip("체결가·원가·손익은 통화별 값입니다. 행에 마우스를 올리면 주문번호·주문 수량·잔량·가격 출처를 볼 수 있습니다.")
        records_layout.addWidget(self.table, 1)
        tabs.addTab(records_box, "주문 장부")
        self.audit = EventLog("order", "주문 전송 의도·접수·체결 확인의 상태 변화 기록입니다. ‘전송 의도’나 ‘접수 여부 확인 필요’를 체결로 해석하지 마세요.")
        tabs.addTab(self.audit, "주문 처리 로그")
        layout.addWidget(tabs, 1)
        layout.addWidget(label("전체 앱 주문 장부의 주문순서 FIFO · 수수료·세금 제외 · 계좌 전체 수익률 아님 · 매수 기록이 없으면 원가/손익 미확인", "muted", wrap=True))

    @staticmethod
    def has_fill(record):
        return record["status"] == "filled" or Decimal(str(record.get("filled_quantity") or 0)) > 0

    def reload(self, store, *, visible_only=False, force=False):
        # Compatibility path for synchronous callers. The dashboard uses a
        # worker-owned LedgerCollector then apply_snapshot to keep I/O/FIFO off
        # the UI thread. The revision includes fill snapshots, not only logs.
        if self._collector is None or self._collector.store is not store:
            self._collector = LedgerCollector(store)
        snapshot = self._collector.collect(force=force or self._ledger is None)
        if snapshot is not None:
            self.apply_snapshot(snapshot, force=force)
        if not visible_only or self.tabs.currentWidget() is self.audit:
            self.audit.reload(store, head=store.event_heads()["order"], force=force)

    def apply_snapshot(self, snapshot, *, force=False):
        """Paint precomputed results only; never reads SQLite or recomputes FIFO."""
        if snapshot is self._applied_snapshot and self._ledger is not None and not force:
            return False
        self._applied_snapshot = snapshot
        self._ledger = snapshot.ledger
        self.performance = snapshot.performance
        self.records = tuple(reversed(snapshot.ledger[-MAX_VISIBLE_ROWS:]))
        self.render()
        return True

    def apply_logs(self, payload):
        if "order" in payload["events"]:
            self.audit.apply_events(payload["events"]["order"], head=payload["heads"]["order"])

    def set_recovery_status(self, status):
        text = status.get("message") or "체결가 확인 전"
        if status.get("last_checked_at"):
            text += f" · 최근 확인 {local_time(status['last_checked_at'])}"
        self.recovery_label.setText(text)
        self.recovery_label.setToolTip(str(status.get("errors") or "주문번호·종목·방향·수량·거래일이 맞는 증권사 체결가만 보완합니다."))
        self.refresh_button.setEnabled(status.get("state") != "running")

    def _performance_summary(self):
        parts = []
        for market, currency, name in (("domestic", "KRW", "한국"), ("us", "USD", "미국")):
            result = self.performance["summaries"].get((market, currency))
            if not result:
                parts.append(f"{name} · 매도 기록 없음")
                continue
            value = result["known_realized_profit"]
            rate = result["known_return_pct"]
            if value is None:
                parts.append(f"{name} · 실현손익 미확인 ({result['unknown_sell_count']}건)")
                continue
            amount = f"{value:+,.0f}" if currency == "KRW" else f"{value:+,.2f}"
            text = f"{name} 실현손익 {amount} {currency} · {rate:+.2f}%"
            if not result["complete"]:
                text += f" (확인분만 · 미확인 {result['unknown_sell_count']}건 별도)"
            parts.append(text)
        self.performance_label.setText("  /  ".join(parts))

    def render(self, *_):
        self._performance_summary()
        records = self.records
        buys = sum(self.has_fill(r) and r["side"] == "buy" for r in records)
        sells = sum(self.has_fill(r) and r["side"] == "sell" for r in records)
        pending = sum(r["status"] in {"accepted", "submitting", "unknown"} for r in records)
        self.summary.setText(f"체결 확인  매수 {buys}건 / 매도 {sells}건    ·    접수·확인 대기 {pending}건")
        selected = self.filter.currentData()
        if selected == "filled":
            records = tuple(r for r in records if self.has_fill(r))
        elif selected == "pending":
            records = tuple(r for r in records if r["status"] in {"accepted", "submitting", "unknown"})
        elif selected == "other":
            records = tuple(r for r in records if r["status"] not in {"accepted", "submitting", "unknown", "filled"})
        self.count_label.setText(f"최근 주문 최대 500건 기준 · 현재 {len(records)}건" if self.records else "아직 자동주문 기록이 없습니다. 신호만 수신해도 여기는 늘어나지 않습니다.")
        rows = []
        metrics = []
        for r in records:
            filled = r.get("filled_quantity")
            remaining = r.get("remaining_quantity")
            # Legacy 'filled' was recorded only after a full-quantity broker confirmation.
            if r["status"] == "filled":
                filled = r["quantity"] if filled is None else filled
                remaining = 0 if remaining is None else remaining
            metric = self.performance["by_rule_id"].get(r["rule_id"], {})
            fill_price = metric.get("effective_fill_price")
            fill_text = "— (미확인)" if fill_price is None or Decimal(str(fill_price)) <= 0 else f"{number(Decimal(str(fill_price)), 0 if r['currency'] == 'KRW' else 4)} {r['currency']}"
            status = STATUS_LABELS.get(r["status"], r["status"])
            if r["status"] == "accepted" and self.has_fill(r):
                status = "부분체결 · 잔량 확인"
            elif r["status"] == "cancelled" and self.has_fill(r):
                status = "부분체결 후 잔량 취소"
            metrics.append(metric)
            def money(value):
                if value is None:
                    return "미확인" if r["side"] == "sell" and self.has_fill(r) else "—"
                return number(value, 0 if r["currency"] == "KRW" else 2)
            rate = metric.get("return_pct")
            rows.append((local_time(r["started_at"]), f"{r['name'] or r['symbol']}\n{r['symbol']}",
                         "한국" if r["market"] == "domestic" else "미국",
                         "매수" if r["side"] == "buy" else "매도", r["quantity"],
                         "—" if filled is None else str(filled), "—" if remaining is None else str(remaining),
                         status, fill_text, r["order_number"] or "—", money(metric.get("cost_basis")),
                         money(metric.get("realized_profit")), f"{rate:+.2f}%" if rate is not None else money(None)))
        if populate(self.table, rows, [r["rule_id"] for r in records]):
            for index, r in enumerate(records):
                self.table.item(index, 3).setForeground(QColor("#ed7892" if r["side"] == "buy" else "#7aa2ff"))
                self.table.item(index, 7).setForeground(QColor("#78e6c7" if r["status"] == "filled" else "#ffda91"))
                self.table.item(index, 7).setToolTip(f"{r['message']}\n체결 정보 확인 시각: {local_time(r.get('observed_at'))}\n주문 직전 참고 현재가: {r.get('reference_price', '—')} {r['currency']} (체결가 아님)")
                remaining = r.get("remaining_quantity")
                if remaining is None:
                    remaining = "0" if r["status"] == "filled" else "미확인"
                details = (f"주문번호 {r['order_number'] or '—'} · 주문 {r['quantity']}주 · 잔량 {remaining}\n"
                           f"가격 재조회: {r.get('recovery_message') or '아직 보완 조회하지 않음'}\n"
                           f"가격 출처: {r.get('recovery_source_api') or '증권사 체결 응답'} · {r.get('recovery_price_basis') or '저장된 체결가'}\n"
                           f"가격 검증: {metrics[index].get('price_reason') or '확인됨'} · 원본 응답 가격 {r.get('fill_price') or '미확인'}\n"
                           f"손익: {metrics[index].get('reason') or '앱 장부의 주문순서 FIFO · 수수료·세금 제외'}")
                for column in (0, 1, 5, 8, 10, 11, 12):
                    self.table.item(index, column).setToolTip(details)
                profit = metrics[index].get("realized_profit")
                if profit is not None:
                    color = QColor("#ed7892" if profit > 0 else "#7aa2ff" if profit < 0 else "#c4cfdf")
                    self.table.item(index, 11).setForeground(color)
                    self.table.item(index, 12).setForeground(color)
