"""Read-only daily trading journal. No broker calls, timers or order actions."""
from __future__ import annotations

from datetime import datetime, timezone
from time import monotonic
from zoneinfo import ZoneInfo

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDateEdit, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.gui import card, label, table
from dockdack.operations_gui import populate
from dockdack.trade_journal import DATE_DESCRIPTION, MARKETS, RETURN_DESCRIPTION, daily_trade_journal, empty_day
from dockdack.watchlist import STATUS_LABELS


def _money(value, currency, *, signed=False, price=False):
    if value is None:
        return "미확인"
    places = 0 if currency == "KRW" else 4 if price else 2
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:,.{places}f} {currency}"


class DailyTradeJournalPanel(QWidget):
    """Caller owns refresh cadence; unchanged ledgers never rebuild tables.

    Pass an existing complete ``records`` ledger to share an upstream read, or
    a reliable ``head`` revision to skip reads. Without either, DB reads are
    throttled to five seconds; force=True is an explicit local-only refresh.
    No event-message content is ever used as execution evidence.
    """

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._ledger = None
        self._head = object()
        self._last_read = float("-inf")
        self.journal = daily_trade_journal(())
        self.pages, self.dates, self.tables, self.values, self.summaries = {}, {}, {}, {}, {}
        self.scroll_areas, self.metric_grids, self.metric_cards = {}, {}, {}
        self._metric_columns = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        heading = QHBoxLayout()
        self.title = label("매매일지", "section")
        heading.addWidget(self.title, 1)
        self.mode_badge = label("", "badge")
        heading.addWidget(self.mode_badge)
        self.refresh_button = QPushButton("일지 새로고침")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.setToolTip("저장된 주문 장부만 다시 읽습니다. API 조회나 매수·매도는 하지 않습니다.")
        self.refresh_button.clicked.connect(lambda: self.refresh(force=True))
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
        layout.addWidget(label(DATE_DESCRIPTION, "muted", wrap=True))
        layout.addWidget(label(RETURN_DESCRIPTION + " · 수수료·세금 제외", "muted", wrap=True))
        self.market_tabs = QTabWidget()
        self.market_tabs.setDocumentMode(True)
        self.market_tabs.tabBar().setDrawBase(False)
        for market, title in (("domestic", "국내 · KRW"), ("us", "해외 (미국) · USD")):
            currency, zone = MARKETS[market]
            page = QWidget()
            page.setObjectName("journalMarketContent")
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            page.setStyleSheet("QWidget#journalMarketContent { background: #121b2a; }")
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 6, 8, 6)
            page_layout.setSpacing(6)
            controls = QHBoxLayout()
            controls.addWidget(label("주문일", "muted"))
            date_edit = QDateEdit()
            date_edit.setCalendarPopup(True)
            date_edit.setDisplayFormat("yyyy-MM-dd")
            date_edit.setStyleSheet("QDateEdit { background: #0d1523; color: #e7edf8; border: 1px solid #33435d; border-radius: 7px; padding: 8px; min-height: 22px; }")
            date_edit.calendarWidget().setStyleSheet("QCalendarWidget QWidget { background: #172235; color: #e7edf8; } QCalendarWidget QAbstractItemView { selection-background-color: #267a76; }")
            date_edit.setAccessibleName(f"{title} 매매일지 주문일")
            today = datetime.now(timezone.utc).astimezone(ZoneInfo(zone)).date()
            date_edit.setDate(QDate(today.year, today.month, today.day))
            date_edit.dateChanged.connect(lambda _, m=market: self.render_market(m))
            self.dates[market] = date_edit
            controls.addWidget(date_edit)
            today_button = QPushButton("오늘")
            today_button.setAutoDefault(False)
            today_button.clicked.connect(lambda _, m=market: self.select_today(m))
            controls.addWidget(today_button)
            zone_text = "서울 날짜" if market == "domestic" else "뉴욕 날짜 · 서머타임 반영"
            scroll_hint = label(zone_text, "muted", wrap=True)
            controls.addWidget(scroll_hint, 1)
            page_layout.addLayout(controls)
            metrics = QGridLayout()
            metrics.setSpacing(6)
            self.metric_grids[market] = metrics
            self.metric_cards[market] = []
            values = {}
            for column, (key, caption) in enumerate((("buy", "체결 매수금액"), ("sell", "체결 매도금액"),
                                                    ("profit", "매도 실현손익 (세전)"), ("return", "매도 실현 수익률"))):
                frame, tile = card()
                tile.setContentsMargins(12, 9, 12, 9)
                tile.setSpacing(4)
                tile.addWidget(label(caption, "muted"))
                value = label("—", "metric", wrap=True)
                value.setStyleSheet("font-size: 20px; font-weight: 600;")
                value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
                value.setAccessibleName(caption)
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                tile.addWidget(value)
                values[key] = value
                self.metric_cards[market].append(frame)
                metrics.addWidget(frame, 0, column)
            self.values[market] = values
            page_layout.addLayout(metrics)
            self.summaries[market] = label("", "muted", wrap=True)
            page_layout.addWidget(self.summaries[market])
            view = table(["주문시각", "종목", "매수/매도", "체결 수량", "체결 평균가", "체결금액",
                          "실현손익", "실현 수익률", "주문 상태"])
            view.setWordWrap(False)
            # Never squeeze the detail ledger down to a clipped header/one row.
            # Short windows scroll the market page instead of hiding its text.
            view.setMinimumHeight(190)
            view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            # Preserve the stock name/code on small windows; horizontal
            # scrolling is preferable to squeezing the symbol to 30 pixels.
            view.horizontalHeader().setStretchLastSection(True)
            for column, width in ((0, 92), (1, 180), (2, 75), (3, 85), (4, 125), (5, 135), (6, 125), (7, 105), (8, 145)):
                view.setColumnWidth(column, width)
            self.tables[market] = view
            self.pages[market] = page
            page_layout.addWidget(view, 1)
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
            scroll.setWidget(page)
            scroll.verticalScrollBar().rangeChanged.connect(
                lambda minimum, maximum, hint=scroll_hint, caption=zone_text:
                hint.setText(caption + (" · 아래로 스크롤하면 상세내역" if maximum > 0 else "")))
            self.scroll_areas[market] = scroll
            self.market_tabs.addTab(scroll, title)
        layout.addWidget(self.market_tabs, 1)
        self.warning = label("", "muted", wrap=True)
        layout.addWidget(self.warning)
        layout.addWidget(label("앱이 기록한 주문만 포함 · 접수/미체결 금액은 제외 · 다른 앱의 거래·입출금·평가손익은 미포함 · KRW와 USD는 합산하지 않습니다.", "muted", wrap=True))
        self._set_mode()
        self._arrange_metrics()
        for market in MARKETS:
            self.render_market(market)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrange_metrics()

    def _arrange_metrics(self):
        # Keep normal desktop dashboards compact; narrow embedded panels use
        # two rows with scrolling rather than shrinking amounts or caveats.
        columns = 4 if self.width() >= 1000 else 2
        if columns == self._metric_columns:
            return
        self._metric_columns = columns
        for market, grid in self.metric_grids.items():
            for frame in self.metric_cards[market]:
                grid.removeWidget(frame)
            for column in range(4):
                grid.setColumnStretch(column, 1 if column < columns else 0)
            for index, frame in enumerate(self.metric_cards[market]):
                grid.addWidget(frame, index // columns, index % columns)

    def _set_mode(self):
        mode = str(getattr(getattr(self.store, "mode", "demo"), "value", getattr(self.store, "mode", "demo")))
        real = mode in {"real", "live"}
        self.mode_badge.setText("실제투자 · REAL" if real else "모의투자 · DEMO")
        self.mode_badge.setStyleSheet("color: #ffd2d8; background: #4b2432; border: 1px solid #a65066;" if real else "")
        self.mode_badge.setToolTip("선택한 투자 환경의 로컬 장부만 표시합니다. 모의·실제 거래를 합산하지 않습니다.")

    def select_today(self, market):
        day = datetime.now(timezone.utc).astimezone(ZoneInfo(MARKETS[market][1])).date()
        self.dates[market].setDate(QDate(day.year, day.month, day.day))

    def refresh(self, *, records=None, head=None, force=False):
        if records is None:
            if not force and ((head is not None and head == self._head)
                              or (head is None and monotonic() - self._last_read < 5)):
                return False
            records = self.store.order_history(limit=None)
            self._last_read = monotonic()
            self._head = head
        ledger = tuple(dict(record) for record in records)
        self._set_mode()
        if ledger == self._ledger and not force:
            return False
        self._ledger = ledger
        self.journal = daily_trade_journal(ledger)
        for market in MARKETS:
            self.render_market(market)
        count = len(self.journal["undated"])
        self.warning.setText(f"날짜·시장 미확인 {count}건은 일별 합계에서 제외되었습니다. 원본 주문 장부에서 확인하세요." if count else "")
        self.warning.setVisible(bool(count))
        return True

    def render_market(self, market):
        if market not in self.values:
            return
        day = self.dates[market].date().toPython()
        summary = self.journal["days"].get((market, day), empty_day(market, day))
        currency = MARKETS[market][0]
        values = self.values[market]
        for side, count_key in (("buy", "buy"), ("sell", "sell_amount")):
            unknown = summary[f"unknown_{count_key}_count"]
            text = _money(summary[f"{side}_amount"], currency)
            if unknown:
                text += f"\n확인분 {_money(summary[f'known_{side}_amount'], currency)}"
            values[side].setText(text)
            values[side].setToolTip(f"실제 체결가격 확인 {summary[f'known_{count_key}_count']}건 / 금액 미확인 {unknown}건 · 주문가/현재가 대체 없음")
        profit, rate = summary["known_realized_profit"], summary["known_return_pct"]
        has_sales = summary["sell_count"] > 0 or summary["unknown_profit_count"] > 0
        values["profit"].setText(_money(profit, currency, signed=True) if has_sales else "매도 체결 없음")
        values["return"].setText(f"{rate:+.2f}%" if rate is not None else "미확인" if has_sales else "—")
        if summary["unknown_profit_count"] and profit is not None:
            values["profit"].setText(values["profit"].text() + "\n(확인분만)")
            values["return"].setText(values["return"].text() + "\n(확인분만)")
        for value in values.values():
            # QLabel's word-wrap height hint can be sacrificed by a crowded
            # parent layout. Explicit newlines carry accounting caveats and
            # must always receive their full line height.
            value.setMinimumHeight(value.fontMetrics().lineSpacing() * len(value.text().splitlines()) + 4)
        values["profit"].setToolTip("전체 과거 매수를 사용한 주문순서 FIFO · 수수료·세금 제외 · 미확인 손익은 0원이 아닙니다.")
        values["return"].setToolTip(RETURN_DESCRIPTION)
        self.summaries[market].setText(
            f"매수 체결 {summary['buy_count']}건 · 매도 체결 {summary['sell_count']}건  |  "
            f"매도 손익 확인 {summary['known_profit_count']}건 / 미확인 {summary['unknown_profit_count']}건\n"
            f"접수·확인 대기 {summary['pending_count']}건 · 거절 {summary['rejected_count']}건 · 취소/기타 {summary['other_count']}건"
            + (f" · 장부 정보 불일치 {summary['invalid_count']}건" if summary["invalid_count"] else ""))
        rows = tuple(reversed(summary["rows"]))
        cells = []
        for row in rows:
            metric, filled = row["metric"], row["has_fill"]
            status = STATUS_LABELS.get(row.get("status"), row.get("status", "미확인"))
            if filled and row.get("status") == "accepted":
                status = "부분체결 · 잔량 대기"
            elif filled and row.get("status") == "cancelled":
                status = "부분체결 후 취소"
            rate = metric.get("return_pct")
            sale = row.get("side") == "sell" and filled
            cells.append((row["local_order_time"], f"{row.get('name') or row.get('symbol')} ({row.get('symbol')})",
                          {"buy": "매수", "sell": "매도"}.get(row.get("side"), "미확인"),
                          str(row["confirmed_quantity"]) if filled and not row["invalid"] else "미확인" if row["invalid"] else "—",
                          _money(row["effective_fill_price"], currency, price=True) if filled else "—",
                          _money(row["amount"], currency) if filled else "—",
                          _money(metric.get("realized_profit"), currency, signed=True) if sale else "—",
                          f"{rate:+.2f}%" if sale and rate is not None else "미확인" if sale else "—", status))
        view = self.tables[market]
        if populate(view, cells, [row["rule_id"] for row in rows]):
            for index, row in enumerate(rows):
                tooltip = (f"주문번호: {row.get('order_number') or '미확인'}\n{DATE_DESCRIPTION}\n"
                           f"증권사 주문일 원문: {row.get('recovery_order_date') or '없음'} · 체결시각 원문: {row.get('recovery_fill_time') or '없음'}\n"
                           f"체결 정보 조회시각(체결시각 아님): {row.get('observed_at') or '없음'}\n"
                           f"가격: {row['metric'].get('price_reason') or '증권사 체결가격'}\n"
                           f"손익: {row['metric'].get('reason') or '전체 로컬 주문순서 FIFO · 수수료·세금 제외'}\n"
                           f"{row.get('message') or ''}")
                for column in range(view.columnCount()):
                    view.item(index, column).setToolTip(tooltip)
                color = "#ed7892" if row.get("side") == "buy" else "#7aa2ff"
                view.item(index, 2).setForeground(QColor(color))
