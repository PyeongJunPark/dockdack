"""Read-only holdings dashboard; fetching is owned by the parent GUI worker."""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import replace
from decimal import Decimal
from typing import Mapping

from PySide6.QtCore import QEvent, QItemSelectionModel, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.fx_reference import UsdKrwReference, fetch_ecb_usd_krw
from dockdack.gui import Worker
from dockdack.models import Market
from dockdack.portfolio import PortfolioMarketState, utc_now
from dockdack.execution_policy import account_equity_cash
from dockdack.trading.model_exit_schedule import model_exit_schedule, planned_model_exit


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


def _planned_exit_label(target: dict, market: Market, upper=None, lower=None) -> str:
    """Describe the model's earliest time exit without implying an order filled."""
    if "lots" in target and not target.get("reconciled"):
        return "장부 대조 필요"
    if target.get("error"):
        return "매도 확인 필요"
    strategy_id = target.get("strategy_id") or target.get("model_id")
    if model_exit_schedule(strategy_id) is None:
        return "가격 조건 시" if upper is not None or lower is not None else "예정일 없음"
    try:
        planned = planned_model_exit({**target, "strategy_id": strategy_id}, market)
    except (ValueError, OverflowError):
        return "일정 확인 필요"
    if planned is None:
        return "체결일 미확인"
    timing = "마감 5분 전" if planned.timing == "preclose" else "장중"
    return f"{planned.day:%Y-%m-%d} {timing}"


def _unreconciled_model_history(target: dict) -> tuple[str, str]:
    """Show persisted BUY provenance without allocating the broker position."""
    names = []
    details = []
    for lot in target.get('lots', ()):
        if not isinstance(lot, dict) or not all(
                isinstance(lot.get(field), str) and lot[field].strip()
                for field in ('lot_id', 'strategy_id', 'model_title')):
            continue
        name = lot['model_title'].strip()
        if name not in names:
            names.append(name)
        remaining = lot.get('quantity_remaining', lot.get('quantity'))
        amount = (f' · 앱 장부 잔여 {_quantity(remaining)}주'
                  if isinstance(remaining, Decimal) and remaining.is_finite() and remaining > 0 else '')
        details.append(f'{name}{amount} · 장부 매수분 {lot["lot_id"]}')
    if not names:
        return '모델 장부 대조 필요', ''
    label = (names[0] if len(names) == 1 else f'{names[0]} 외 {len(names) - 1}개 모델') + ' · 대조 필요'
    return label, ('앱의 저장된 매수 모델 출처 (증권사 보유주식의 모델별 배분 아님):\n'
                   + '\n'.join(details) + '\n')


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


# Preserve these readable widths on narrow windows, where the table scrolls.
# On wide windows, extra space is distributed among descriptive/money columns.
_TABLE_COLUMN_WIDTHS = (86, 220, 62, 78, 111, 111, 129, 117, 95, 145, 145, 160, 195)
_TABLE_EXPAND_WEIGHTS = {1: 3, 4: 1, 5: 1, 6: 1, 7: 1, 9: 1, 10: 1, 11: 2, 12: 2}


class PortfolioPanel(QWidget):
    """Shows successful empty balances differently from unavailable balances.

    All values are account snapshots, not independent live quote requests.
    Calling ``apply`` or clicking refresh never changes order authorization.
    """

    request_refresh = Signal()

    def __init__(self, parent: QWidget | None = None, *, fx_fetcher=None):
        super().__init__(parent)
        self._fx_fetcher = fx_fetcher or fetch_ecb_usd_krw
        self._fx_reference: UsdKrwReference | None = None
        self._fx_request_id = 0
        self._fx_workers = {}
        self._payload: dict[Market, PortfolioMarketState] = {}
        self._row_signatures = {}
        self._rendered_states = {}
        self._live_quotes = {}
        self._rows_by_instrument = {}
        self._column_fit_pending: set[Market] = set()
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
        self.detail_buttons: dict[Market, QPushButton] = {}
        self.summaries: dict[Market, QLabel] = {}
        self.market_labels: dict[Market, dict[str, QLabel]] = {}
        self._compact = False
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
            labels["updated"].hide()
            self.market_labels[market] = labels
            self.market_cards[market] = frame
            details = QPushButton("계좌 금액·조회 정보 펼치기")
            details.setCheckable(True)
            details.setAutoDefault(False)
            details.setAccessibleName(f"{title} 계좌 금액과 조회 정보")
            details.setToolTip("보유종목 표는 그대로 두고 계좌 금액과 잔고 조회 정보를 펼칩니다.")
            details.toggled.connect(lambda _checked, selected=market: self._update_card_visibility(selected))
            details.hide()
            self.detail_buttons[market] = details
            page_layout.addWidget(details)
            page_layout.addWidget(frame)

            summary = _label("잔고 조회 전 · 미확인은 보유종목 0개를 의미하지 않습니다.", "muted")
            summary.setWordWrap(True)
            self.summaries[market] = summary
            page_layout.addWidget(summary)
            summary.hide()
            if market is Market.US:
                fx_controls = QHBoxLayout()
                self.fx_toggle = QCheckBox("원화 환산")
                self.fx_toggle.setAccessibleName("미국 보유종목 원화 환산")
                self.fx_toggle.setToolTip("화면 표시만 바꿉니다. USD 잔고·매도 기준·주문은 변경하지 않습니다.")
                self.fx_toggle.toggled.connect(self._fx_toggled)
                fx_controls.addWidget(self.fx_toggle)
                self.fx_refresh_button = QPushButton("환율 갱신")
                self.fx_refresh_button.setAutoDefault(False)
                self.fx_refresh_button.setEnabled(False)
                self.fx_refresh_button.clicked.connect(self._request_fx)
                fx_controls.addWidget(self.fx_refresh_button)
                self.fx_status = _label("USD", "muted")
                fx_controls.addWidget(self.fx_status, 1)
                page_layout.addLayout(fx_controls)
            self.tables[market] = self._make_table()
            page_layout.addWidget(self.tables[market], 1)
            self.market_tabs.addTab(page, title)
        layout.addWidget(self.market_tabs, 1)
        for table in self.tables.values():
            table.viewport().installEventFilter(self)
        self.market_tabs.currentChanged.connect(
            lambda _index: self._schedule_table_fit(self.current_market))
        self.apply({})

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            for market, table in self.tables.items():
                if watched is table.viewport():
                    self._schedule_table_fit(market)
                    break
        return super().eventFilter(watched, event)

    def _schedule_table_fit(self, market: Market) -> None:
        if market in self._column_fit_pending:
            return
        self._column_fit_pending.add(market)
        QTimer.singleShot(0, lambda selected=market: self._fit_pending_table(selected))

    def _fit_pending_table(self, market: Market) -> None:
        self._column_fit_pending.discard(market)
        table = self.tables[market]
        available = table.viewport().width()
        if available <= 0:
            return
        base_total = sum(width for column, width in enumerate(_TABLE_COLUMN_WIDTHS)
                         if not table.isColumnHidden(column))
        extra = max(0, available - base_total)
        weight_total = sum(_TABLE_EXPAND_WEIGHTS.values())
        allocated = 0
        expandable = tuple(_TABLE_EXPAND_WEIGHTS)
        for column, base_width in enumerate(_TABLE_COLUMN_WIDTHS):
            if table.isColumnHidden(column):
                continue
            addition = 0
            if column in _TABLE_EXPAND_WEIGHTS:
                if column == expandable[-1]:
                    addition = extra - allocated
                else:
                    addition = extra * _TABLE_EXPAND_WEIGHTS[column] // weight_total
                    allocated += addition
            target = base_width + addition
            if table.columnWidth(column) != target:
                table.setColumnWidth(column, target)

    def _active_fx(self, market: Market) -> UsdKrwReference | None:
        return self._fx_reference if market is Market.US and self.fx_toggle.isChecked() else None

    def _fx_note(self) -> str:
        reference = self._fx_reference
        if reference is None:
            return ""
        return (f"ECB {reference.published_on.isoformat()} 고시 · "
                f"1 USD = {reference.krw_per_usd:,.2f} KRW · 표시용 환산, 실제 환전/거래일 환율 아님")

    def _display_money(self, value, market: Market, *, price=False, signed=False, table=False) -> str:
        reference = self._active_fx(market)
        currency = "KRW" if market is Market.DOMESTIC else "USD"
        if reference is not None and value is not None:
            amount = value * reference.krw_per_usd
            return (_table_money(amount, "KRW", signed=signed) if table
                    else _money(amount, "KRW", signed=signed))
        return (_table_money(value, currency, price=price, signed=signed) if table
                else _money(value, currency, price=price, signed=signed))

    def _display_target(self, value: Decimal, market: Market, sign: str) -> str:
        reference = self._active_fx(market)
        if reference is None:
            return f"{sign} {_target_price(value)}"
        # FX is illustrative, never an executable tick-size threshold. Limit
        # display precision to whole KRW even when the cross-rate repeats.
        return f"{sign} {_table_money(value * reference.krw_per_usd, 'KRW')}"

    def _fx_toggled(self, checked: bool) -> None:
        if not checked:
            self._fx_request_id += 1
            self.fx_refresh_button.setEnabled(False)
            self.fx_status.setText("USD")
            self.fx_status.setToolTip("")
            self.market_tabs.setTabText(1, "미국 · USD")
            self._rendered_states.pop(Market.US, None)
            self.apply(self._payload)
            return
        if self._fx_reference is not None:
            # OFF switches presentation only. Re-enable the last dated rate
            # immediately; the explicit refresh button requests a new quote.
            self._show_fx_reference(self._fx_reference)
            self._rendered_states.pop(Market.US, None)
            self.apply(self._payload)
            return
        self._request_fx()

    def _show_fx_reference(self, reference: UsdKrwReference) -> None:
        self.fx_status.setText(f"1 USD = {reference.krw_per_usd:,.0f}원")
        self.fx_status.setToolTip(self._fx_note() + f"\n{reference.source_url}\nUSD 잔고·매도 기준·주문은 변경되지 않습니다.")
        self.market_tabs.setTabText(1, "미국 · KRW")
        self.fx_refresh_button.setEnabled(True)

    def _request_fx(self) -> None:
        if not self.fx_toggle.isChecked():
            return
        self._fx_request_id += 1
        token = self._fx_request_id
        self.fx_refresh_button.setEnabled(False)
        if self._fx_reference is None:
            self.fx_status.setText("환율 조회 중 · USD 유지")
            self.fx_status.setToolTip("")
            self.market_tabs.setTabText(1, "미국 · USD")
        else:
            self.fx_status.setText("환율 갱신 중 · 이전 환율")
            self.fx_status.setToolTip(self._fx_note())
            self.market_tabs.setTabText(1, "미국 · KRW")
        self._rendered_states.pop(Market.US, None)
        self.apply(self._payload)
        worker = Worker(lambda: self._fx_fetcher(timeout=3.0))
        worker.signals.completed.connect(
            lambda result, error, request_id=token: self._fx_finished(request_id, result, error))
        self._fx_workers[token] = worker
        QThreadPool.globalInstance().start(worker)

    def _fx_finished(self, request_id, reference, error) -> None:
        self._fx_workers.pop(request_id, None)
        if request_id != self._fx_request_id or not self.fx_toggle.isChecked():
            return
        valid = (isinstance(reference, UsdKrwReference)
                 and isinstance(reference.published_on, date)
                 and isinstance(reference.krw_per_usd, Decimal)
                 and reference.krw_per_usd.is_finite() and reference.krw_per_usd > 0)
        if error is not None or not valid:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__ if error else "응답 형식 미확인"
            if self._fx_reference is None:
                self.fx_status.setText("환율 조회 실패 · USD 유지")
                self.fx_status.setToolTip(f"ECB 환율 조회 실패: {reason}")
                self.market_tabs.setTabText(1, "미국 · USD")
            else:
                self.fx_status.setText("환율 갱신 실패 · 이전 환율")
                self.fx_status.setToolTip(f"ECB 환율 갱신 실패: {reason}\n" + self._fx_note())
                self.market_tabs.setTabText(1, "미국 · KRW")
        else:
            self._fx_reference = reference
            self._show_fx_reference(reference)
        if error is not None or not valid:
            self.fx_refresh_button.setEnabled(True)
        self._rendered_states.pop(Market.US, None)
        self.apply(self._payload)

    def closeEvent(self, event):
        self._fx_request_id += 1
        super().closeEvent(event)

    def set_compact(self, compact: bool) -> None:
        """Use the available height for holdings rows; account details can expand."""
        compact = bool(compact)
        if self._compact == compact:
            return
        self._compact = compact
        for market, table in self.tables.items():
            table.verticalHeader().setDefaultSectionSize(34 if compact else 42)
            self._update_card_visibility(market)

    def _update_card_visibility(self, market: Market) -> None:
        button = self.detail_buttons[market]
        button.setVisible(self._compact)
        self.market_cards[market].setVisible(not self._compact or button.isChecked())
        button.setText("계좌 금액·조회 정보 접기" if button.isChecked() else "계좌 금액·조회 정보 펼치기")

    @staticmethod
    def _make_table() -> QTableWidget:
        result = QTableWidget(0, 13)
        result.setHorizontalHeaderLabels([
            "시장 / 통화", "종목명 / 코드", "보유", "매도 가능", "평균 매수가", "현재가",
            "평가금액", "평가손익", "수익률", "익절 매도 예정", "손절 매도 예정", "매수 모델",
            "기간 매도 예정",
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
        for column, width in enumerate(_TABLE_COLUMN_WIDTHS[:9]):
            result.setColumnWidth(column, width)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        for column in (9, 10):
            result.setColumnWidth(column, _TABLE_COLUMN_WIDTHS[column])
        header.moveSection(header.visualIndex(9), 2)
        header.moveSection(header.visualIndex(10), 3)
        result.setColumnWidth(11, _TABLE_COLUMN_WIDTHS[11])
        header.moveSection(header.visualIndex(11), 2)
        result.setColumnWidth(12, _TABLE_COLUMN_WIDTHS[12])
        header.moveSection(header.visualIndex(12), 3)
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
            # Successful refresh is already represented by holdings and time;
            # keep only actionable/unknown states in the compact header.
            labels["status"].setText('' if status == 'ok' else text)
            labels["status"].setStyleSheet("color: #ffb586;" if status in {"error", "stale"} else "color: #95a4bb;" if status == "unknown" else "color: #78e6c7;")
            labels["status"].setToolTip(
                f"최근 잔고 조회 {_time(state.last_attempt)} · {state.error}"
                if status == "error" else "")
            labels["holdings"].setText(f"{len(state.positions)}종목" if state.snapshot is not None else "보유종목 미확인")
            account = state.snapshot
            evaluation = None if account is None else account.total_evaluation
            profit = None if account is None else account.total_profit_loss
            if account is not None:
                if evaluation is None:
                    evaluation = sum((p.evaluation_amount for p in state.positions), Decimal(0))
                if profit is None:
                    profit = sum((p.profit_loss for p in state.positions), Decimal(0))
            fx = self._active_fx(market)
            labels["evaluation"].setText(self._display_money(evaluation, market))
            labels["profit"].setText(self._display_money(profit, market, signed=True))
            labels["profit"].setStyleSheet("color: #f08098;" if profit is not None and profit > 0 else "color: #84a7ff;" if profit is not None and profit < 0 else "")
            labels["cash"].setText(self._display_money(None if account is None else account.cash, market))
            labels["available"].setText(self._display_money(None if account is None else account.available_to_order, market))
            cash_context, cash_detail, negative_cash = _cash_context(account, currency)
            fx_note = self._fx_note() if fx is not None else ""
            labels["evaluation"].setToolTip((f"원본 {_money(evaluation, 'USD')}\n{fx_note}" if fx is not None else ""))
            labels["profit"].setToolTip((f"원본 {_money(profit, 'USD', signed=True)}\n{fx_note}" if fx is not None else ""))
            labels["cash"].setToolTip(cash_detail + (f"\n원본 {_money(account.cash, 'USD')}\n{fx_note}" if fx is not None and account is not None else ""))
            labels["cash"].setStyleSheet("color: #ffb586;" if negative_cash else "")
            labels["available"].setToolTip("증권사가 반환한 주문가능금액입니다. 현재 예수금·D+2 추정예수금·출금가능금액과 다릅니다."
                                          + (f"\n원본 {_money(account.available_to_order, 'USD')}\n{fx_note}" if fx is not None and account is not None else ""))
            updated = f"잔고 기준 {_time(state.fetched_at)}"
            if status == "error":
                updated += f" · 최근 시도 {_time(state.last_attempt)} · {state.error}"
            labels["updated"].setText(updated)
            summary_text = (
                f"{title}: {text}" if status in {"unknown", "error", "stale", "empty"}
                else f"{title} {len(state.positions)}종목 · {'KRW 표시' if fx is not None else currency} 계좌 잔고 기준 / 자동 갱신 최소 60초"
            )
            self.summaries[market].setText(summary_text + ("\n" + cash_context if cash_context else ""))
            self.summaries[market].setToolTip(cash_detail)
            expanded = []
            for held in sorted(state.positions, key=lambda item: (-item.evaluation_amount, item.symbol)):
                key = f'{held.market.value}:{held.exchange}:{held.symbol}'
                group = getattr(self, '_exit_targets', {}).get(key, {})
                if group.get('lots') and group.get('reconciled'):
                    for lot in group['lots']:
                        qty, average = lot['quantity'], lot['average_price']
                        if qty <= 0:
                            continue
                        cost, evaluation = qty * average, qty * held.current_price
                        virtual = replace(held, quantity=qty, sellable_quantity=lot['sellable_quantity'],
                                          average_price=average, evaluation_amount=evaluation,
                                          profit_loss=evaluation-cost, profit_rate=(evaluation-cost)/cost*100,
                                          raw={**held.raw, 'prototype_lot_id': lot['lot_id']})
                        expanded.append((virtual, lot))
                else:
                    if 'lots' in group and not group.get('reconciled'):
                        model_history, _ = _unreconciled_model_history(group)
                        group = {**group, 'error': ' / '.join(map(str, group.get('issues', ()))) or '장부와 증권사 잔고 대조 필요',
                                 'model_title': model_history}
                    expanded.append((held, group))
            for position, target in expanded:
                key = f'{position.market.value}:{position.exchange}:{position.symbol}'
                live = self._live_quotes.get(key)
                live_price = live[0] if live and (state.fetched_at is None or live[1] >= state.fetched_at) else position.current_price
                upper = target.get('take_profit_price')
                lower = target.get('stop_loss_price')
                if not target and position.average_price > 0:
                    upper, lower = position.average_price * Decimal('1.01'), position.average_price * Decimal('0.992')
                values = (
                    f"{title} · {'KRW' if fx is not None else currency}", f"{position.name or position.symbol} · {position.symbol}", _quantity(position.quantity),
                    (f"공유 {_quantity(target['broker_sellable_quantity'])}" if target.get('sellable_is_shared')
                     else _quantity(position.sellable_quantity)), self._display_money(position.average_price, market, price=True, table=True),
                    self._display_money(live_price, market, price=True, table=True), self._display_money(position.evaluation_amount, market, table=True),
                    self._display_money(position.profit_loss, market, signed=True, table=True),
                    f"{'+' if position.profit_rate > 0 else ''}{position.profit_rate:,.2f}%",
                    '확인 필요 · 보류' if target.get('error') else '미확인' if upper is None else self._display_target(upper, market, '≥'),
                    '확인 필요 · 보류' if target.get('error') else '미확인' if lower is None else self._display_target(lower, market, '≤'),
                    target.get('model_title') or '미확인 / 수동·외부',
                    _planned_exit_label(target, position.market, upper, lower),
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
        previous = targets.get(key, {})
        targets[key] = target
        self._exit_targets = targets
        def structure(value):
            return (value.get('reconciled'), tuple(
                (lot.get('lot_id'), lot.get('quantity'), lot.get('average_price'),
                 lot.get('sellable_quantity'), lot.get('broker_sellable_quantity'), lot.get('sellable_is_shared'),
                 lot.get('strategy_id'), lot.get('buy_fill_observed_at'))
                for lot in value.get('lots', ())))
        if ('lots' in target or 'lots' in previous) and structure(previous) != structure(target):
            self._rendered_states.clear()
            self.apply(self._payload)
            return
        table = self.tables[inst.market]
        lots = {lot['lot_id']: lot for lot in target.get('lots', ())} if target.get('reconciled') else {}
        for row, lot_id in self._rows_by_instrument.get(key, ()):
            row_target = lots.get(lot_id, target)
            price = table.item(row, 5)
            price.setText(self._display_money(quote.price, inst.market, price=True, table=True))
            price.setToolTip(f'독립 매도 감시 현재가 · {_time(self._live_quotes[key][1])}\n평가금액·손익은 별도 잔고 조회 시각 기준입니다.'
                             + (f"\n원본 {_money(quote.price, 'USD', price=True)}\n{self._fx_note()}" if self._active_fx(inst.market) is not None else ""))
            for column, field, sign in ((9, 'take_profit_price', '≥'), (10, 'stop_loss_price', '≤')):
                value = row_target.get(field)
                blocked = row_target.get('error') or ('lots' in target and not target.get('reconciled'))
                table.item(row, column).setText('확인 필요 · 보류' if blocked else '미확인' if value is None else self._display_target(value, inst.market, sign))
                original = (f"\n실제 매도 기준 {sign} {_target_price(value)} USD\n{self._fx_note()}"
                            if value is not None and self._active_fx(inst.market) is not None else "")
                table.item(row, column).setToolTip(self._target_tooltip(key, lot_id) + original)
            model_history = (_unreconciled_model_history(target)[0]
                             if 'lots' in target and not target.get('reconciled') else None)
            table.item(row, 11).setText(model_history or row_target.get('model_title') or '미확인 / 수동·외부')
            table.item(row, 11).setToolTip(self._target_tooltip(key, lot_id))
            table.item(row, 12).setText(_planned_exit_label(row_target, inst.market,
                                                              row_target.get('take_profit_price'),
                                                              row_target.get('stop_loss_price')))
            table.item(row, 12).setToolTip(self._target_tooltip(key, lot_id))
        self._row_signatures.pop(inst.market, None)

    def _target_tooltip(self, key, lot_id=None):
        target = getattr(self, '_exit_targets', {}).get(key, {})
        if lot_id is not None:
            target = next((lot for lot in target.get('lots', ()) if lot.get('lot_id') == lot_id), {})
        elif 'lots' in target and not target.get('reconciled'):
            _, model_history = _unreconciled_model_history(target)
            return (model_history
                    + '증권사 보유 수량은 종목별 합산입니다. 모델별 원가·목표가·매도가능수량은 검증되지 않았습니다.\n'
                    '모델별 체결 장부와 증권사 잔고를 대조할 수 없어 이 종목 자동매도를 보류합니다.\n'
                    + '\n'.join(map(str, target.get('issues', ()))))
        if target.get('error'):
            return ('해당 종목 자동매도 보류: ' + str(target['error'])
                    + '\n다른 종목 감시는 계속합니다. 거래소/종목 확인 전에는 이 종목을 자동주문하지 않습니다.')
        source = target.get('source', '')
        origin = (source if target.get('model_title') else
                  '목표가 없는 기존 보유분: 평균매입가 +1% / −0.8%' if not source or source.startswith('평균매입가')
                  else '매입가 확인 필요' if source == '매입가 확인 필요' else '매수 신호에서 받은 목표가격')
        attribution = (f"매수 모델: {target.get('model_title') or '미확인 / 수동·외부'}\n"
                       f"저장된 매수 신호: {target.get('buy_signal_id') or '미확인'}\n"
                       "현재 선택한 모델로 과거 매수 출처를 추정하지 않습니다.\n")
        if lot_id:
            attribution += f'분리 매수분: {lot_id}\n확인된 체결 수량·체결가 기준 가상 구분이며 증권사 잔고는 종목별 합산입니다.\n'
        if target.get('sellable_is_shared'):
            attribution += f"매도가능수량은 계좌 전체 공유 상한 {target['broker_sellable_quantity']}주입니다. 모델별 행의 수량을 더해 팔 수 있다는 뜻이 아닙니다.\n"
        strategy_id = target.get('strategy_id') or target.get('model_id')
        if model_exit_schedule(strategy_id) is not None:
            market = Market(key.split(':', 1)[0])
            planned = _planned_exit_label(target, market)
            return (attribution + origin + f'\n기간 매도 예정: {planned} (해당 거래소 현지 기준).\n'
                    '별도 매도 조건이 먼저 충족되면 일찍 팔릴 수 있고, 주문 OFF·장외·미체결이면 예정일에 매도가 완료되지 않을 수 있습니다. '
                    '기한이 지나면 다음 유효 정규장에 다시 조건을 확인합니다.')
        return attribution + origin + '\n현재가가 목표에 닿으면 매도 조건을 재확인합니다. 주문 OFF·장외에는 주문하지 않으며 체결을 보장하지 않습니다.'

    def set_exit_targets(self, targets):
        if targets != getattr(self, '_exit_targets', {}):
            self._exit_targets = dict(targets)
            self._rendered_states.clear()
            self.apply(self._payload)

    def _apply_rows(self, market: Market, rows) -> None:
        signature = tuple((values, position.profit_loss, position.profit_rate, position.raw.get('prototype_lot_id'), status, fetched_at)
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
            for instrument in [key for key in self._rows_by_instrument if key.startswith(market.value + ':')]:
                del self._rows_by_instrument[instrument]
            for row, (values, position, status, fetched_at) in enumerate(rows):
                key = f"{position.market.value}:{position.symbol}"
                lot_id = position.raw.get('prototype_lot_id')
                instrument = f'{position.market.value}:{position.exchange}:{position.symbol}'
                self._rows_by_instrument.setdefault(instrument, []).append((row, lot_id))
                if lot_id:
                    key += ':' + lot_id
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
                    if column in {9, 10, 11, 12} or (column == 3 and lot_id):
                        item.setToolTip(self._target_tooltip(f'{position.market.value}:{position.exchange}:{position.symbol}', lot_id))
                    elif column == 4 and not lot_id:
                        group = getattr(self, '_exit_targets', {}).get(instrument, {})
                        if 'lots' in group and not group.get('reconciled'):
                            item.setToolTip('증권사 종목 합산 평균매입가입니다. 저장된 각 모델 매수분의 원가는 확인되지 않아 자동매도를 보류합니다.')
                    elif column == 5:
                        live = self._live_quotes.get(f'{position.market.value}:{position.exchange}:{position.symbol}')
                        if live and (fetched_at is None or live[1] >= fetched_at):
                            item.setToolTip(f'독립 매도 감시 현재가 · {_time(live[1])}\n평가금액·손익은 별도 잔고 조회 시각 기준입니다.')
                    if self._active_fx(market) is not None and column in {4, 5, 6, 7, 9, 10}:
                        original = {
                            4: position.average_price,
                            5: (live[0] if (live := self._live_quotes.get(instrument))
                                and (fetched_at is None or live[1] >= fetched_at) else position.current_price),
                            6: position.evaluation_amount,
                            7: position.profit_loss,
                        }.get(column)
                        if column in {9, 10}:
                            group = getattr(self, '_exit_targets', {}).get(instrument, {})
                            target = next((lot for lot in group.get('lots', ()) if lot.get('lot_id') == lot_id), {}) if lot_id else group
                            field = 'take_profit_price' if column == 9 else 'stop_loss_price'
                            original = target.get(field)
                            if not group and not lot_id and position.average_price > 0:
                                original = position.average_price * (Decimal('1.01') if column == 9 else Decimal('0.992'))
                        if original is not None and (column not in {9, 10} or item.text().startswith(('≥', '≤'))):
                            prefix = ('실제 매도 기준 ' + ('≥ ' if column == 9 else '≤ ') if column in {9, 10} else '원본 ')
                            item.setToolTip(item.toolTip() + f'\n{prefix}{_money(original, "USD", price=column in {4, 5, 9, 10})}\n{self._fx_note()}')
                    table.setItem(row, column, item)
                if key in selection:
                    table.selectionModel().select(table.model().index(row, 1),
                                                  QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            table.verticalScrollBar().setValue(vertical)
            table.horizontalScrollBar().setValue(horizontal)
        finally:
            table.setUpdatesEnabled(True)
