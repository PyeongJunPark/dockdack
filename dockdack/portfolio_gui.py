"""Read-only holdings dashboard; fetching is owned by the parent GUI worker."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Mapping

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.models import Market
from dockdack.portfolio import PortfolioMarketState, utc_now


def _label(text: str, name: str = "") -> QLabel:
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setObjectName(name)
    return result


def _money(value: Decimal | None, currency: str, *, price: bool = False, signed: bool = False) -> str:
    if value is None:
        return "미확인"
    places = 0 if currency == "KRW" else 4 if price else 2
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:,.{places}f} {currency}"


def _quantity(value: Decimal) -> str:
    return f"{value:,f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"


def _table_money(value: Decimal, currency: str, *, price: bool = False, signed: bool = False) -> str:
    # Currency belongs in the market column, keeping the important P/L columns
    # visible even when the dashboard is used in a smaller desktop window.
    return _money(value, currency, price=price, signed=signed).removesuffix(f" {currency}")


def _time(value: datetime | None) -> str:
    return "조회 전" if value is None else value.astimezone().strftime("%m/%d %H:%M:%S")


class PortfolioPanel(QWidget):
    """Shows successful empty balances differently from unavailable balances.

    All values are account snapshots, not independent live quote requests.
    Calling ``apply`` or clicking refresh never changes order authorization.
    """

    request_refresh = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._payload: dict[Market, PortfolioMarketState] = {}
        self._row_signatures = {}
        self._rendered_states = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.heading = _label("현재 보유종목 · 모의계좌", "section")
        header.addWidget(self.heading)
        header.addStretch()
        self.refresh_button = QPushButton("보유종목 새로고침")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.setToolTip("잔고만 조회합니다. 주문을 켜거나 매수·매도하지 않습니다. 시장별 최소 60초 간격입니다.")
        self.refresh_button.clicked.connect(self.request_refresh.emit)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        self.market_tabs = QTabWidget()
        self.market_tabs.setDocumentMode(True)
        self.market_tabs.tabBar().setDrawBase(False)
        self.tables: dict[Market, QTableWidget] = {}
        self.market_cards: dict[Market, QFrame] = {}
        self.summaries: dict[Market, QLabel] = {}
        self.market_labels: dict[Market, dict[str, QLabel]] = {}
        for market, title in ((Market.DOMESTIC, "한국 · KRW"), (Market.US, "미국 · USD")):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(8, 8, 8, 8)
            page_layout.setSpacing(8)
            frame = QFrame()
            frame.setObjectName("card")
            grid = QGridLayout(frame)
            grid.setContentsMargins(14, 12, 14, 12)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(10)
            labels = {key: _label("") for key in ("status", "holdings", "evaluation", "profit", "cash", "available", "updated")}
            account_heading = QHBoxLayout()
            account_heading.setSpacing(14)
            account_heading.addWidget(_label(title, "section"))
            labels["status"].setObjectName("portfolioStatus")
            account_heading.addWidget(labels["status"])
            account_heading.addStretch(1)
            labels["holdings"].setObjectName("portfolioHoldings")
            account_heading.addWidget(labels["holdings"])
            grid.addLayout(account_heading, 0, 0, 1, 4)
            for column, (key, caption) in enumerate((("evaluation", "총 평가금액"), ("profit", "평가손익"),
                                                     ("cash", "예수금"), ("available", "주문가능금액"))):
                tile = QFrame()
                tile.setObjectName("portfolioMetric")
                tile.setMinimumHeight(76)
                tile_layout = QVBoxLayout(tile)
                tile_layout.setContentsMargins(12, 10, 12, 10)
                tile_layout.setSpacing(7)
                tile_layout.addWidget(_label(caption, "muted"))
                labels[key].setObjectName("portfolioValue")
                labels[key].setMinimumHeight(28)
                labels[key].setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                labels[key].setAccessibleName(caption)
                tile_layout.addWidget(labels[key])
                grid.addWidget(tile, 1, column)
                grid.setColumnStretch(column, 1)
            labels["updated"].setObjectName("muted")
            labels["updated"].setWordWrap(True)
            grid.addWidget(labels["updated"], 2, 0, 1, 4)
            self.market_labels[market] = labels
            self.market_cards[market] = frame
            page_layout.addWidget(frame)

            summary = _label("잔고 조회 전 · 미확인은 보유종목 0개를 의미하지 않습니다.", "muted")
            summary.setWordWrap(True)
            self.summaries[market] = summary
            page_layout.addWidget(summary)
            self.tables[market] = self._make_table()
            page_layout.addWidget(self.tables[market], 1)
            self.market_tabs.addTab(page, title)
        layout.addWidget(self.market_tabs, 1)
        footer = _label("예수금 ≠ 주문가능금액 ≠ 주문당 상한 · KRW와 USD는 합산하지 않습니다.", "muted")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.apply({})

    @staticmethod
    def _make_table() -> QTableWidget:
        result = QTableWidget(0, 9)
        result.setHorizontalHeaderLabels([
            "시장 / 통화", "종목명 / 코드", "보유", "매도 가능", "평균 매수가", "현재가",
            "평가금액", "평가손익", "수익률",
        ])
        result.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        result.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        result.setAlternatingRowColors(True)
        result.setWordWrap(False)
        result.setTextElideMode(Qt.TextElideMode.ElideRight)
        result.setShowGrid(False)
        result.verticalHeader().hide()
        result.verticalHeader().setDefaultSectionSize(42)
        header = result.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(54)
        for column, width in enumerate((86, 220, 62, 78, 111, 111, 129, 117, 95)):
            result.setColumnWidth(column, width)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Market/currency are already prominent in the selected tab. Keep the
        # data column for selection keys and copy/export compatibility, without
        # sacrificing scarce horizontal space to duplicate information.
        result.setColumnHidden(0, True)
        # Two rows remain usable in a 1080 x 780 window; keep account values
        # readable instead of crushing the metric cards to preserve blank rows.
        result.setMinimumHeight(100)
        result.setToolTip("현재가·평가손익은 표시된 잔고 조회 시각 기준입니다. 시세의 연속 갱신과 다를 수 있습니다.")
        return result

    @property
    def current_market(self) -> Market:
        return Market.US if self.market_tabs.currentIndex() == 1 else Market.DOMESTIC

    @property
    def table(self) -> QTableWidget:
        """Compatibility accessor for the selected market's holdings table."""
        return self.tables[self.current_market]

    @property
    def summary_label(self) -> QLabel:
        return self.summaries[self.current_market]

    def brief_summary(self) -> str:
        parts = []
        for market, title in ((Market.DOMESTIC, "한국"), (Market.US, "미국")):
            state = self._payload.get(market, PortfolioMarketState(market))
            parts.append(f"{title} {len(state.positions)}종목" if state.snapshot is not None else f"{title} 미확인")
        return " / ".join(parts)

    def apply(self, payload: Mapping[Market, PortfolioMarketState], now: datetime | None = None) -> None:
        self._payload = dict(payload)
        now = now or utc_now()
        for market, title in ((Market.DOMESTIC, "한국"), (Market.US, "미국")):
            state = self._payload.get(market, PortfolioMarketState(market))
            status = state.status(now)
            render_state = (state, status)
            if self._rendered_states.get(market) == render_state:
                continue
            self._rendered_states[market] = render_state
            rows = []
            labels = self.market_labels[market]
            currency = "KRW" if market is Market.DOMESTIC else "USD"
            text = {"unknown": "미확인 · 조회 전", "ok": "잔고 확인", "empty": "보유종목 없음",
                    "stale": "오래된 잔고 · 재조회 필요", "error": "조회 오류 · 이전 잔고 유지" if state.snapshot else "조회 오류 · 보유 여부 미확인"}[status]
            labels["status"].setText(text)
            labels["status"].setStyleSheet("color: #ffb586;" if status in {"error", "stale"} else "color: #95a4bb;" if status == "unknown" else "color: #78e6c7;")
            labels["holdings"].setText(f"{len(state.positions)}종목" if state.snapshot is not None else "보유종목 미확인")
            account = state.snapshot
            evaluation = None if account is None else account.total_evaluation
            profit = None if account is None else account.total_profit_loss
            if account is not None:
                if evaluation is None:
                    evaluation = sum((p.evaluation_amount for p in state.positions), Decimal(0))
                if profit is None:
                    profit = sum((p.profit_loss for p in state.positions), Decimal(0))
            labels["evaluation"].setText(_money(evaluation, currency))
            labels["profit"].setText(_money(profit, currency, signed=True))
            labels["profit"].setStyleSheet("color: #f08098;" if profit is not None and profit > 0 else "color: #84a7ff;" if profit is not None and profit < 0 else "")
            labels["cash"].setText(_money(None if account is None else account.cash, currency))
            labels["available"].setText(_money(None if account is None else account.available_to_order, currency))
            updated = f"잔고 기준 {_time(state.fetched_at)}"
            if status == "error":
                updated += f" · 최근 시도 {_time(state.last_attempt)} · {state.error}"
            labels["updated"].setText(updated)
            self.summaries[market].setText(
                f"{title}: {text}" if status in {"unknown", "error", "stale", "empty"}
                else f"{title} {len(state.positions)}종목 · {currency} 계좌 잔고 기준 / 자동 갱신 최소 60초"
            )
            for position in sorted(state.positions, key=lambda item: (-item.evaluation_amount, item.symbol)):
                values = (
                    f"{title} · {currency}", f"{position.name or position.symbol} · {position.symbol}", _quantity(position.quantity),
                    _quantity(position.sellable_quantity), _table_money(position.average_price, currency, price=True),
                    _table_money(position.current_price, currency, price=True), _table_money(position.evaluation_amount, currency),
                    _table_money(position.profit_loss, currency, signed=True),
                    f"{'+' if position.profit_rate > 0 else ''}{position.profit_rate:,.2f}%",
                )
                rows.append((values, position, status, state.fetched_at))
            self._apply_rows(market, rows)

    def _apply_rows(self, market: Market, rows) -> None:
        signature = tuple((values, position.profit_loss, position.profit_rate, status, fetched_at)
                          for values, position, status, fetched_at in rows)
        if signature == self._row_signatures.get(market):
            return
        self._row_signatures[market] = signature
        table = self.tables[market]
        selection = {item.data(Qt.ItemDataRole.UserRole) for item in table.selectedItems() if item.column() == 1}
        vertical = table.verticalScrollBar().value()
        horizontal = table.horizontalScrollBar().value()
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            table.clearSelection()
            for row, (values, position, status, fetched_at) in enumerate(rows):
                key = f"{position.market.value}:{position.symbol}"
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    item.setToolTip(f"{position.name or position.symbol} · {position.symbol} · {position.currency}\n잔고 기준 {_time(fetched_at)}")
                    if column in {2, 3, 4, 5, 6, 7, 8}:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if column in {7, 8}:
                        amount = position.profit_loss if column == 7 else position.profit_rate
                        if amount:
                            item.setForeground(QColor("#f08098" if amount > 0 else "#84a7ff"))
                    if status in {"error", "stale"}:
                        item.setToolTip(f"{position.name or position.symbol} · 잔고 기준 {_time(fetched_at)}\n이전 잔고입니다. 현재 보유 상태와 다를 수 있습니다.")
                        if column in {0, 1}:
                            item.setForeground(QColor("#ffb586"))
                    table.setItem(row, column, item)
                if key in selection:
                    table.selectionModel().select(table.model().index(row, 1),
                                                  QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            table.verticalScrollBar().setValue(vertical)
            table.horizontalScrollBar().setValue(horizontal)
        finally:
            table.setUpdatesEnabled(True)
