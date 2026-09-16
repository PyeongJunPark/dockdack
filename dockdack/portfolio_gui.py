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
from dockdack.execution_policy import account_equity_cash


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


def _target_price(value: Decimal) -> str:
    # A trigger can fall between valid order ticks. Show its actual threshold,
    # not a rounded display price that implies a different sell condition.
    text = f'{value:,f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _time(value: datetime | None) -> str:
    return "조회 전" if value is None else value.astimezone().strftime("%m/%d %H:%M:%S")


def _cash_context(account, currency):
    """Compact visible settlement context, with exact meaning in the tooltip."""
    detail = "증권사가 반환한 현재 예수금을 부호 그대로 표시합니다. 주문가능금액·출금가능금액과 다릅니다."
    if account is None:
        return "", detail, False
    parts = []
    if account.market is Market.DOMESTIC:
        for field, label in (("cash_d1", "D+1 추정"), ("cash_d2", "D+2 추정")):
            value = getattr(account, field, None)
            if isinstance(value, Decimal) and value.is_finite():
                parts.append(f"{label} {_money(value, currency)}")
    receivable = getattr(account, "cash_receivable", None)
    if isinstance(receivable, Decimal) and receivable.is_finite():
        parts.append(f"{'외화' if account.market is Market.US else ''}현금미수 {_money(receivable, currency)}")
    negative = isinstance(account.cash, Decimal) and account.cash.is_finite() and account.cash < 0
    if negative:
        try:
            account_equity_cash(account)
        except ValueError:
            parts.append("예수금 음수 · 비중 매수 보류")
            detail += "\n현재 예수금이 음수이며 확인된 결제 후 예수금이 없어 비중 매수를 보류합니다."
        else:
            parts.append("현재 예수금 음수 · 결제/미수 확인")
            detail += ("\n비중 매수의 자산 기준은 증권사 D+2 추정예수금 + 보유 평가금액을 사용합니다. "
                       "가상 매도 수수료·세금 차감 전이며, 실제 주문은 D+2 추정예수금·주문가능금액·상한 이내로 제한합니다.")
    if parts:
        detail += "\n" + " · ".join(parts)
    if account.market is Market.DOMESTIC:
        detail += "\nD+1/D+2는 결제일 기준 증권사 추정치이며 현재 현금이나 출금 가능액을 뜻하지 않습니다."
    return " · ".join(parts), detail, negative


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
        self._live_quotes = {}
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
        result = QTableWidget(0, 11)
        result.setHorizontalHeaderLabels([
            "시장 / 통화", "종목명 / 코드", "보유", "매도 가능", "평균 매수가", "현재가",
            "평가금액", "평가손익", "수익률", "익절 매도 예정", "손절 매도 예정",
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
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        for column in (9, 10):
            result.setColumnWidth(column, 145)
        header.moveSection(header.visualIndex(9), 2)
        header.moveSection(header.visualIndex(10), 3)
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
            cash_context, cash_detail, negative_cash = _cash_context(account, currency)
            labels["cash"].setToolTip(cash_detail)
            labels["cash"].setStyleSheet("color: #ffb586;" if negative_cash else "")
            labels["available"].setToolTip("증권사가 반환한 주문가능금액입니다. 현재 예수금·D+2 추정예수금·출금가능금액과 다릅니다.")
            updated = f"잔고 기준 {_time(state.fetched_at)}"
            if status == "error":
                updated += f" · 최근 시도 {_time(state.last_attempt)} · {state.error}"
            labels["updated"].setText(updated)
            summary_text = (
                f"{title}: {text}" if status in {"unknown", "error", "stale", "empty"}
                else f"{title} {len(state.positions)}종목 · {currency} 계좌 잔고 기준 / 자동 갱신 최소 60초"
            )
            self.summaries[market].setText(summary_text + ("\n" + cash_context if cash_context else ""))
            self.summaries[market].setToolTip(cash_detail)
            for position in sorted(state.positions, key=lambda item: (-item.evaluation_amount, item.symbol)):
                key = f'{position.market.value}:{position.exchange}:{position.symbol}'
                live = self._live_quotes.get(key)
                live_price = live[0] if live and (state.fetched_at is None or live[1] >= state.fetched_at) else position.current_price
                target = getattr(self, '_exit_targets', {}).get(key, {})
                upper = target.get('take_profit_price')
                lower = target.get('stop_loss_price')
                if not target and position.average_price > 0:
                    upper, lower = position.average_price * Decimal('1.01'), position.average_price * Decimal('0.992')
                values = (
                    f"{title} · {currency}", f"{position.name or position.symbol} · {position.symbol}", _quantity(position.quantity),
                    _quantity(position.sellable_quantity), _table_money(position.average_price, currency, price=True),
                    _table_money(live_price, currency, price=True), _table_money(position.evaluation_amount, currency),
                    _table_money(position.profit_loss, currency, signed=True),
                    f"{'+' if position.profit_rate > 0 else ''}{position.profit_rate:,.2f}%",
                    '확인 필요 · 보류' if target.get('error') else '미확인' if upper is None else f'≥ {_target_price(upper)}',
                    '확인 필요 · 보류' if target.get('error') else '미확인' if lower is None else f'≤ {_target_price(lower)}',
                )
                rows.append((values, position, status, state.fetched_at))
            self._apply_rows(market, rows)

    def apply_holding_quote(self, update):
        """One fresh SELL-pass quote updates just its existing row, no I/O/sort."""
        inst, quote, target = update['instrument'], update['quote'], update['targets']
        key = update['watch_id']
        self._live_quotes[key] = (quote.price, utc_now())
        while len(self._live_quotes) > 1000:
            self._live_quotes.pop(next(iter(self._live_quotes)))
        targets = getattr(self, '_exit_targets', {})
        targets[key] = target
        self._exit_targets = targets
        table = self.tables[inst.market]
        for row in range(table.rowCount()):
            cell = table.item(row, 1)
            if cell is None or cell.data(Qt.ItemDataRole.UserRole) != f'{inst.market.value}:{inst.symbol}':
                continue
            price = table.item(row, 5)
            price.setText(_table_money(quote.price, inst.currency, price=True))
            price.setToolTip(f'독립 매도 감시 현재가 · {_time(self._live_quotes[key][1])}\n평가금액·손익은 별도 잔고 조회 시각 기준입니다.')
            for column, field, sign in ((9, 'take_profit_price', '≥'), (10, 'stop_loss_price', '≤')):
                value = target.get(field)
                table.item(row, column).setText('미확인' if value is None else f'{sign} {_target_price(value)}')
                table.item(row, column).setToolTip(self._target_tooltip(key))
            break

    def _target_tooltip(self, key):
        target = getattr(self, '_exit_targets', {}).get(key, {})
        if target.get('error'):
            return ('해당 종목 자동매도 보류: ' + str(target['error'])
                    + '\n다른 종목 감시는 계속합니다. 거래소/종목 확인 전에는 이 종목을 자동주문하지 않습니다.')
        source = target.get('source', '')
        origin = ('목표가 없는 기존 보유분: 평균매입가 +1% / −0.8%' if not source or source.startswith('평균매입가')
                  else '매입가 확인 필요' if source == '매입가 확인 필요' else '매수 신호에서 받은 목표가격')
        return origin + '\n현재가가 목표에 닿으면 매도 조건을 재확인합니다. 주문 OFF·장외에는 주문하지 않으며 체결을 보장하지 않습니다.'

    def set_exit_targets(self, targets):
        if targets != getattr(self, '_exit_targets', {}):
            self._exit_targets = dict(targets)
            self._rendered_states.clear()
            self.apply(self._payload)

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
                    if column in {9, 10}:
                        item.setToolTip(self._target_tooltip(f'{position.market.value}:{position.exchange}:{position.symbol}'))
                    elif column == 5:
                        live = self._live_quotes.get(f'{position.market.value}:{position.exchange}:{position.symbol}')
                        if live and (fetched_at is None or live[1] >= fetched_at):
                            item.setToolTip(f'독립 매도 감시 현재가 · {_time(live[1])}\n평가금액·손익은 별도 잔고 조회 시각 기준입니다.')
                    table.setItem(row, column, item)
                if key in selection:
                    table.selectionModel().select(table.model().index(row, 1),
                                                  QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            table.verticalScrollBar().setValue(vertical)
            table.horizontalScrollBar().setValue(horizontal)
        finally:
            table.setUpdatesEnabled(True)
