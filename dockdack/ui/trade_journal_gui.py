"""Read-only daily, monthly and yearly journal. No broker calls or orders."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from time import monotonic
from zoneinfo import ZoneInfo

from PySide6.QtCore import QDate, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QPushButton,
    QScrollArea, QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.fx_reference import UsdKrwReference, fetch_ecb_usd_krw
from dockdack.gui import Worker, card, label, table
from dockdack.operations_gui import populate
from dockdack.trade_journal import (
    DATE_DESCRIPTION, MARKETS, RETURN_DESCRIPTION, daily_trade_journal, period_trade_journal,
)
from dockdack.watchlist import STATUS_LABELS
from dockdack.activity_snapshot import LedgerSnapshot, MAX_VISIBLE_ROWS
from dockdack.signal_bridge import prototype_order_label


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

    request_refresh = Signal()

    def __init__(self, store, parent=None, *, fx_fetcher=None):
        super().__init__(parent)
        self.store = store
        self._fx_fetcher = fx_fetcher or fetch_ecb_usd_krw
        self._fx_reference: UsdKrwReference | None = None
        self._fx_request_id = 0
        self._fx_workers = {}
        self._ledger = None
        self._head = object()
        self._last_read = float("-inf")
        self._applied_snapshot = None
        self.journal = daily_trade_journal(())
        self.pages, self.dates, self.periods, self.tables = {}, {}, {}, {}
        self.previous_buttons, self.next_buttons = {}, {}
        self.values, self.summaries = {}, {}
        self.scroll_areas, self.metric_grids, self.metric_cards = {}, {}, {}
        self._metric_columns = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        heading = QHBoxLayout()
        self.title = label("매매일지", "section")
        self.title.setToolTip(DATE_DESCRIPTION + "\n" + RETURN_DESCRIPTION + " · 수수료·세금 제외\n"
                              "앱이 기록한 주문만 포함하며 다른 앱의 거래·입출금·평가손익은 제외합니다.")
        heading.addWidget(self.title, 1)
        self.mode_badge = label("", "badge")
        heading.addWidget(self.mode_badge)
        self.refresh_button = QPushButton("일지 새로고침")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.setToolTip("저장된 주문 장부만 다시 읽습니다. API 조회나 매수·매도는 하지 않습니다.")
        self.refresh_button.clicked.connect(self.request_refresh.emit)
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
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
            controls.addWidget(label("주문 기간", "muted"))
            period_edit = QComboBox()
            for caption, period in (("일별", "day"), ("월별", "month"), ("연별", "year")):
                period_edit.addItem(caption, period)
            period_edit.setAccessibleName(f"{title} 매매일지 조회 기간")
            period_edit.setToolTip("주문일을 기준으로 일·월·연 합계를 전환합니다. 체결일을 추정하지 않습니다.")
            self.periods[market] = period_edit
            controls.addWidget(period_edit)
            previous = QPushButton("◀ 이전")
            previous.setAutoDefault(False)
            previous.setAccessibleName(f"{title} 이전 조회 기간")
            previous.clicked.connect(lambda _, m=market: self.shift_period(m, -1))
            self.previous_buttons[market] = previous
            controls.addWidget(previous)
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
            following = QPushButton("다음 ▶")
            following.setAutoDefault(False)
            following.setAccessibleName(f"{title} 다음 조회 기간")
            following.clicked.connect(lambda _, m=market: self.shift_period(m, 1))
            self.next_buttons[market] = following
            controls.addWidget(following)
            period_edit.currentIndexChanged.connect(lambda _, m=market: self.change_period(m))
            today_button = QPushButton("오늘")
            today_button.setAutoDefault(False)
            today_button.clicked.connect(lambda _, m=market: self.select_today(m))
            controls.addWidget(today_button)
            zone_text = "서울 날짜" if market == "domestic" else "뉴욕 날짜 · 서머타임 반영"
            scroll_hint = label(zone_text, "muted", wrap=True)
            controls.addWidget(scroll_hint, 1)
            page_layout.addLayout(controls)
            if market == "us":
                fx_controls = QHBoxLayout()
                self.fx_toggle = QCheckBox("원화 환산")
                self.fx_toggle.setAccessibleName("미국 매매일지 원화 환산")
                self.fx_toggle.setToolTip("화면 표시만 바꿉니다. USD 장부·주문·체결·실제 원화 손익은 변경하지 않습니다.")
                self.fx_toggle.toggled.connect(self._fx_toggled)
                fx_controls.addWidget(self.fx_toggle)
                self.fx_refresh_button = QPushButton("환율 갱신")
                self.fx_refresh_button.setAutoDefault(False)
                self.fx_refresh_button.setEnabled(False)
                self.fx_refresh_button.clicked.connect(self._request_fx)
                fx_controls.addWidget(self.fx_refresh_button)
                self.fx_status = label("USD", "muted", wrap=True)
                fx_controls.addWidget(self.fx_status, 1)
                page_layout.addLayout(fx_controls)
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
            self.summaries[market].hide()  # Detail counts remain available in the ledger table/tooltips.
            view = table(["주문시각", "종목", "매수/매도", "체결 수량", "체결 평균가", "체결금액",
                          "실현손익", "실현 수익률", "주문 상태", "매수 모델"])
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
            view.setColumnWidth(9, 160)
            view.horizontalHeader().moveSection(9, 3)
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
        self.mode_badge.setText("실전" if real else "모의투자")
        self.mode_badge.setStyleSheet("color: #ffd2d8; background: #4b2432; border: 1px solid #a65066;" if real else "")
        self.mode_badge.setToolTip("선택한 투자 환경의 로컬 장부만 표시합니다. 모의·실제 거래를 합산하지 않습니다.")

    def select_today(self, market):
        day = datetime.now(timezone.utc).astimezone(ZoneInfo(MARKETS[market][1])).date()
        self.dates[market].setDate(QDate(day.year, day.month, day.day))

    def _fx_toggled(self, checked):
        if not checked:
            self._fx_request_id += 1
            self.fx_refresh_button.setEnabled(False)
            self.fx_status.setText("USD")
            self.fx_status.setToolTip("")
            self.render_market("us")
            return
        if self._fx_reference is not None:
            self._show_fx_reference(self._fx_reference)
            self.render_market("us")
            return
        self._request_fx()

    def _show_fx_reference(self, reference):
        self.fx_status.setText(f"1 USD ≈ {reference.krw_per_usd:,.0f}원")
        self.fx_status.setToolTip(
            f"ECB {reference.published_on.isoformat()} 고시 · 표시용 환산, 실제 원화 실현손익 아님\n"
            f"EUR 기준 USD {reference.eur_usd} / KRW {reference.eur_krw}의 같은 고시일 교차환율\n"
            f"{reference.source_url}\n거래·체결일 환율이 아니며 수수료·세금·환전 스프레드가 없습니다.")
        self.fx_refresh_button.setEnabled(True)

    def _request_fx(self):
        if not self.fx_toggle.isChecked():
            return
        self._fx_request_id += 1
        request_id = self._fx_request_id
        if self._fx_reference is None:
            self.fx_status.setText("환율 조회 중")
            self.fx_status.setToolTip("")
        else:
            self.fx_status.setText("환율 갱신 중 · 이전 환율")
            self.fx_status.setToolTip(
                f"ECB {self._fx_reference.published_on.isoformat()} 고시 환율을 갱신하는 동안 표시합니다.")
        self.fx_refresh_button.setEnabled(False)
        self.render_market("us")
        worker = Worker(lambda: self._fx_fetcher(timeout=3.0))
        worker.signals.completed.connect(
            lambda result, error, token=request_id: self._fx_finished(token, result, error))
        self._fx_workers[request_id] = worker
        QThreadPool.globalInstance().start(worker)

    def _fx_finished(self, request_id, reference, error):
        self._fx_workers.pop(request_id, None)
        if request_id != self._fx_request_id or not self.fx_toggle.isChecked():
            return
        valid = (isinstance(reference, UsdKrwReference)
                 and isinstance(reference.published_on, date)
                 and isinstance(reference.krw_per_usd, Decimal)
                 and reference.krw_per_usd.is_finite() and reference.krw_per_usd > 0
                 and isinstance(reference.eur_usd, Decimal)
                 and reference.eur_usd.is_finite() and reference.eur_usd > 0
                 and isinstance(reference.eur_krw, Decimal)
                 and reference.eur_krw.is_finite() and reference.eur_krw > 0)
        if error is not None or not valid:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__ if error else "응답 형식 미확인"
            if self._fx_reference is None:
                self.fx_status.setText("환율 조회 실패 · USD 유지")
                self.fx_status.setToolTip(f"ECB 환율 조회 실패: {reason}")
            else:
                self.fx_status.setText("환율 갱신 실패 · 이전 환율")
                self.fx_status.setToolTip(
                    f"ECB 환율 갱신 실패: {reason}\n"
                    f"이전 환율 ECB {self._fx_reference.published_on.isoformat()} 고시 · "
                    f"1 USD = {self._fx_reference.krw_per_usd:,.2f} KRW")
        else:
            self._fx_reference = reference
            self._show_fx_reference(reference)
        self.fx_refresh_button.setEnabled(True)
        self.render_market("us")

    def closeEvent(self, event):
        self._fx_request_id += 1
        super().closeEvent(event)

    def change_period(self, market):
        period = self.periods[market].currentData()
        self.dates[market].setDisplayFormat(
            {"day": "yyyy-MM-dd", "month": "yyyy-MM", "year": "yyyy"}[period])
        self.tables[market].horizontalHeaderItem(0).setText(
            "주문시각" if period == "day" else "주문일·시각")
        self.tables[market].setColumnWidth(0, 92 if period == "day" else 160)
        self.render_market(market)

    def shift_period(self, market, direction):
        if direction not in {-1, 1}:
            raise ValueError("이전 또는 다음 기간만 선택할 수 있습니다.")
        current = self.dates[market].date()
        period = self.periods[market].currentData()
        if period == "day":
            selected = current.addDays(direction)
        elif period == "month":
            selected = QDate(current.year(), current.month(), 1).addMonths(direction)
        else:
            selected = QDate(current.year(), 1, 1).addYears(direction)
        self.dates[market].setDate(selected)

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
        return self.apply_snapshot(LedgerSnapshot(head, ledger, daily_trade_journal(ledger)), force=force)

    def apply_snapshot(self, snapshot, *, force=False):
        """Apply precomputed worker results; no DB or FIFO work on the UI thread."""
        self._set_mode()
        if snapshot is self._applied_snapshot and not force:
            return False
        self._applied_snapshot = snapshot
        self._ledger = snapshot.ledger
        self.journal = snapshot.journal
        for market in MARKETS:
            self.render_market(market)
        count = len(self.journal["undated"])
        self.warning.setText(f"날짜·시장 미확인 {count}건은 일·월·연 합계에서 제외되었습니다. 주문 장부에서 확인하세요." if count else "")
        self.warning.setVisible(bool(count))
        return True

    def render_market(self, market):
        if market not in self.values:
            return
        day = self.dates[market].date().toPython()
        period = self.periods[market].currentData()
        summary = period_trade_journal(self.journal, market, day, period)
        currency = MARKETS[market][0]
        fx = self._fx_reference if market == "us" and self.fx_toggle.isChecked() else None

        def display_money(amount, *, signed=False, price=False):
            if fx is not None and amount is not None:
                return _money(amount * fx.krw_per_usd, "KRW", signed=signed, price=price)
            return _money(amount, currency, signed=signed, price=price)

        fx_note = (f"ECB {fx.published_on.isoformat()} 고시 1 USD ≈ {fx.krw_per_usd:,.2f} KRW "
                   "표시용 환산 · 체결일 환율/실제 원화 손익 아님" if fx is not None else "")
        values = self.values[market]
        for side, count_key in (("buy", "buy"), ("sell", "sell_amount")):
            unknown = summary[f"unknown_{count_key}_count"]
            amount = summary[f"{side}_amount"]
            text = display_money(amount)
            if unknown:
                text += f"\n확인분 {display_money(summary[f'known_{side}_amount'])}"
            values[side].setText(text)
            values[side].setToolTip(
                f"실제 체결가격 확인 {summary[f'known_{count_key}_count']}건 / 금액 미확인 {unknown}건 · 주문가/현재가 대체 없음"
                + (f"\nUSD {_money(summary[f'known_{side}_amount'], 'USD')} · {fx_note}" if fx is not None else ""))
        profit, rate = summary["known_realized_profit"], summary["known_return_pct"]
        has_sales = summary["sell_count"] > 0 or summary["unknown_profit_count"] > 0
        profit_text = display_money(profit, signed=True) if has_sales else "매도 체결 없음"
        values["profit"].setText(profit_text)
        values["return"].setText(f"{rate:+.2f}%" if rate is not None else "미확인" if has_sales else "—")
        if summary["unknown_profit_count"] and profit is not None:
            values["profit"].setText(values["profit"].text() + "\n(확인분만)")
            values["return"].setText(values["return"].text() + "\n(확인분만)")
        for value in values.values():
            # QLabel's word-wrap height hint can be sacrificed by a crowded
            # parent layout. Explicit newlines carry accounting caveats and
            # must always receive their full line height.
            value.setMinimumHeight(value.fontMetrics().lineSpacing() * max(2, len(value.text().splitlines())) + 4)
        values["profit"].setToolTip(RETURN_DESCRIPTION + " · 수수료·세금 제외 · 미확인 손익은 0원이 아닙니다."
                                    + (f"\nUSD {_money(profit, 'USD', signed=True)} · {fx_note}" if fx is not None else ""))
        values["return"].setToolTip(RETURN_DESCRIPTION + ("\n수익률은 USD 매도손익·매입원가 기준입니다." if fx is not None else ""))
        period_title = (day.isoformat() if period == "day" else
                        day.strftime("%Y-%m") if period == "month" else str(day.year))
        self.summaries[market].setText(
            f"{period_title} 주문일 기준 · 매수 체결 {summary['buy_count']}건 · 매도 체결 {summary['sell_count']}건  |  "
            f"매도 손익 확인 {summary['known_profit_count']}건 / 미확인 {summary['unknown_profit_count']}건\n"
            f"접수·확인 대기 {summary['pending_count']}건 · 거절 {summary['rejected_count']}건 · 취소/기타 {summary['other_count']}건"
            + (f" · 장부 정보 불일치 {summary['invalid_count']}건" if summary["invalid_count"] else ""))
        rows = tuple(reversed(summary["rows"][-MAX_VISIBLE_ROWS:]))
        if len(summary["rows"]) > MAX_VISIBLE_ROWS:
            self.summaries[market].setText(self.summaries[market].text() +
                f"\n상세 표 최근 {MAX_VISIBLE_ROWS}건 / 전체 {len(summary['rows'])}건 · 위 합계는 전체 주문 기준")
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
            order_time = (row["local_order_time"] if period == "day" else
                          f"{row['day']:%Y-%m-%d} {row['local_order_time']}")
            cells.append((order_time, f"{row.get('name') or row.get('symbol')} ({row.get('symbol')})",
                          {"buy": "매수", "sell": "매도"}.get(row.get("side"), "미확인"),
                          str(row["confirmed_quantity"]) if filled and not row["invalid"] else "미확인" if row["invalid"] else "—",
                          display_money(row["effective_fill_price"], price=True) if filled else "—",
                          display_money(row["amount"]) if filled else "—",
                          display_money(metric.get("realized_profit"), signed=True) if sale else "—",
                          f"{rate:+.2f}%" if sale and rate is not None else "미확인" if sale else "—", status,
                          prototype_order_label(row)))
        view = self.tables[market]
        for column, title in ((4, "체결 평균가"), (5, "체결금액"), (6, "실현손익")):
            view.horizontalHeaderItem(column).setText(title)
        if populate(view, cells, [row["rule_id"] for row in rows]):
            for index, row in enumerate(rows):
                fx_details = ""
                if fx is not None:
                    original_profit = (_money(row["metric"].get("realized_profit"), "USD", signed=True)
                                       if row.get("side") == "sell" and row["has_fill"] else "—")
                    fx_details = (f"USD 체결가 {_money(row['effective_fill_price'], 'USD', price=True)} · "
                                  f"체결금액 {_money(row['amount'], 'USD')} · "
                                  f"실현손익 {original_profit}\n{fx_note}\n")
                tooltip = (f"주문번호: {row.get('order_number') or '미확인'}\n{DATE_DESCRIPTION}\n"
                           f"매수 모델: {prototype_order_label(row)}\n"
                           f"저장된 신호: {row.get('external_signal_id') or '미확인'} · 현재 선택한 모델과 무관\n"
                           f"증권사 주문일 원문: {row.get('recovery_order_date') or '없음'} · 체결시각 원문: {row.get('recovery_fill_time') or '없음'}\n"
                           f"체결 정보 조회시각(체결시각 아님): {row.get('observed_at') or '없음'}\n"
                           f"가격: {row['metric'].get('price_reason') or '증권사 체결가격'}\n"
                           f"손익: {row['metric'].get('reason') or RETURN_DESCRIPTION + ' · 수수료·세금 제외'}\n"
                           + fx_details + f"{row.get('message') or ''}")
                for column in range(view.columnCount()):
                    view.item(index, column).setToolTip(tooltip)
                color = "#ed7892" if row.get("side") == "buy" else "#7aa2ff"
                view.item(index, 2).setForeground(QColor(color))
