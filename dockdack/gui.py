"""DockDack demo trading desktop. Run with uv run --extra gui dockdack-gui."""

from __future__ import annotations

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack import Market, OrderOutcomeUnknown, OrderRequest, OrderSide, TradingMode
from dockdack.gui_service import Instrument, TradingService
from dockdack.environment_gui import EnvironmentSelector, confirm_environment, environment_name
from dockdack.environment_store import environment_base, selected_mode, store_for_service
from dockdack.window_controls import WindowControls
from dockdack.branding import APP_NAME, TAGLINE, app_icon, apply_branding, set_windows_app_id


STYLE = """
QWidget { color: #e7edf8; font-family: 'Malgun Gothic'; font-size: 13px; }
QMainWindow, QWidget#root { background: #0c111c; }
QFrame#sidebar { background: #101724; border-right: 1px solid #222e41; }
QFrame#card { background: #141d2c; border: 1px solid #253249; border-radius: 12px; }
QLabel { background: transparent; }
QLabel#muted { color: #95a4bb; }
QLabel#heading { font-size: 26px; font-weight: 700; }
QLabel#section { font-size: 17px; font-weight: 700; }
QLabel#price { font-size: 42px; font-weight: 700; }
QLabel#metric { font-size: 21px; font-weight: 700; }
QLabel#brand { font-size: 23px; font-weight: 700; }
QLabel#badge { color: #78e6c7; background: #173f3c; border: 1px solid #286357;
              border-radius: 8px; padding: 7px 12px; font-weight: 700; }
QLabel#error { color: #ffb586; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0d1523; border: 1px solid #33435d; border-radius: 7px;
    padding: 9px 11px; min-height: 20px; selection-background-color: #267a76;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #62d7bd; }
QComboBox QAbstractItemView { background: #182439; selection-background-color: #285567; }
QPushButton { background: #213048; border: 1px solid #344661; border-radius: 7px;
              padding: 10px 15px; font-weight: 600; }
QPushButton:hover { background: #2a3e59; border-color: #5e819e; }
QPushButton:pressed { background: #172335; }
QPushButton#primary { background: #66dcc0; color: #082820; border: none; }
QPushButton#primary:hover { background: #91ecd5; }
QPushButton#buy { background: #ec617b; color: #fff; border: none; min-height: 26px; }
QPushButton#sell { background: #527feb; color: #fff; border: none; min-height: 26px; }
QPushButton:disabled { color: #637087; background: #1d2839; border-color: #29374c; }
QPushButton#buy:disabled, QPushButton#sell:disabled, QPushButton#primary:disabled { color: #637087; background: #1d2839; }
QPushButton#watch { text-align: left; background: #172235; padding: 13px; }
QTabWidget::pane { border: 1px solid #253249; border-radius: 7px; background: #121b2a; }
QTabBar::tab { background: #101927; color: #93a4be; padding: 11px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #76e0c5; border-bottom: 2px solid #76e0c5; }
QTableWidget { background: #121b2a; alternate-background-color: #172132; border: none; gridline-color: #253249; }
QHeaderView::section { background: #1b293c; color: #9cadc7; padding: 9px;
                        border: none; border-bottom: 1px solid #314056; }
QTableWidget::item { padding: 8px; }
QTableWidget::item:selected { background: #25435a; }
QDialog { background: #141d2c; }
QCheckBox { spacing: 7px; color: #a5b6cc; }
QStatusBar { background: #101724; color: #9bacc6; }
QToolTip { background: #213048; color: #e7edf8; border: 1px solid #425977; }
"""


def label(text: str, name: str = "", *, wrap: bool = False) -> QLabel:
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setObjectName(name)
    result.setWordWrap(wrap)
    return result


def number(value, places: int = 2) -> str:
    return "—" if value is None else f"{value:,.{places}f}"


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 19, 22, 19)
    layout.setSpacing(12)
    return frame, layout


def table(headers: list[str]) -> QTableWidget:
    result = QTableWidget(0, len(headers))
    result.setHorizontalHeaderLabels(headers)
    result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    result.verticalHeader().hide()
    result.verticalHeader().setDefaultSectionSize(39)
    result.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    result.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    result.setAlternatingRowColors(True)
    result.setShowGrid(False)
    return result


class WorkerSignals(QObject):
    completed = Signal(object, object)
    progress = Signal(object)


class Worker(QRunnable):
    def __init__(self, operation: Callable):
        super().__init__()
        self.operation = operation
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.operation()
        except Exception as exc:
            self.signals.completed.emit(None, exc)
        else:
            self.signals.completed.emit(result, None)


class OrderDialog(QDialog):
    def __init__(self, request: OrderRequest, parent=None, *, mode=TradingMode.DEMO):
        super().__init__(parent)
        name = environment_name(mode)
        self.setWindowTitle(f"{name} 주문 최종 확인")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(17)
        side = "매수" if request.side is OrderSide.BUY else "매도"
        currency = "KRW" if request.market is Market.DOMESTIC else "USD"
        layout.addWidget(label(f"{name} 주문" + (" · 실제 자금 사용" if mode is TradingMode.REAL else ""), "badge"))
        layout.addWidget(label(f"{request.symbol}  {side} {request.quantity:,}주", "heading"))
        price = f"{request.price:,.4f} {currency}" if request.price is not None else "시장가 / 체결가격 미정"
        layout.addWidget(label(f"거래소  {request.exchange}\n주문 단가  {price}"))
        if request.estimated_notional is not None:
            layout.addWidget(label(f"주문금액  {request.estimated_notional:,.4f} {currency}\n수수료 제외", "section"))
        layout.addWidget(label("확인한 가격으로 한 번 전송합니다. 접수 후에도 체결 여부를 확인하세요.", "muted", wrap=True))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(f"{'실전' if mode is TradingMode.REAL else '모의'} {side} 전송")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setAutoDefault(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("돌아가기")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)


class TradingWindow(QMainWindow):
    def __init__(self, service: TradingService | None = None, *, store=None):
        super().__init__()
        self.service = service or TradingService()
        self.store = store if store is not None else store_for_service(self.service)
        if self.store.mode is not selected_mode(self.service):
            raise ValueError('거래 환경과 저장소가 다릅니다.')
        if self.store.mode is TradingMode.REAL and self.store.storage_scope != getattr(self.service, 'storage_scope', None):
            raise ValueError('실전 API 인증 범위와 저장소가 다릅니다.')
        self._environment_base = environment_base(self.store)
        self._environment_stores = {(self.store.mode, self.store.storage_scope): self.store}
        self.instrument: Instrument | None = None
        self.last_quote = None
        self._worker: Worker | None = None
        self._success: Callable | None = None
        self._confirming = False
        self._job_name = ""
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.setWindowTitle(f"{APP_NAME} | 모의투자 트레이딩 데스크")
        self.setWindowIcon(app_icon())
        self.window_controls = WindowControls(self)
        self.resize(1280, 880)
        self.setMinimumSize(1080, 790)
        self.setStyleSheet(STYLE)
        self._build()
        self._sync_environment()
        self.timer = QTimer(self)
        self.timer.setInterval(15000)
        self.timer.timeout.connect(self._auto_refresh)
        self.timer.start()

    def _sync_environment(self):
        mode = selected_mode(self.service)
        name = environment_name(mode)
        self.environment_selector.apply(mode)
        self.account_mode_badge.setText(f'{name} 계좌 · 선택')
        market_index = self.order_kind.findData('market')
        self.order_kind.setItemText(market_index, '시장가' if mode is TradingMode.REAL else '시장가 (국내)')
        self.environment_notice.setText(
            '실제 자금이 사용됩니다.\n실전 API 키와\nDOCKDACK_ALLOW_LIVE_ORDERS=true가 필요합니다.'
            if mode is TradingMode.REAL else '가상 자금으로 거래합니다.\n모의 API 키는 로컬 .env에서 읽습니다.')
        self.environment_notice.setStyleSheet('color: #ffad9d;' if mode is TradingMode.REAL else '')
        self.setWindowTitle(f'{APP_NAME} | {name} 트레이딩 데스크')
        self.statusBar().showMessage(f'{name} · 주문 접수와 체결은 다릅니다')

    def _clear_environment_context(self):
        self.instrument = self.last_quote = None
        self.auto_refresh.setChecked(False)
        self.stock_name.setText('거래 환경 전환 · 종목을 새로 조회하세요')
        self.price_label.setText('—')
        self.change_label.setText('이전 환경 시세를 사용하지 않습니다.')
        self.market_tag.setText('—')
        self.order_symbol.setText('선택된 종목 없음')
        self.quote_time.setText('아직 조회하지 않음')
        self.limit_price.setValue(0)
        for view in (self.positions, self.open_orders, self.executions, self.activity):
            view.setRowCount(0)
        for metric in (self.cash_label, self.available_label, self.profit_label):
            metric.setText('—')
        self._set_busy(False)

    def request_environment(self, mode):
        mode = TradingMode(mode)
        if self._worker or self._confirming or mode is selected_mode(self.service):
            return
        self._confirming = True
        self._set_busy(True)
        try:
            if not confirm_environment(self, mode):
                return
            service = TradingService(mode=mode)
            if mode is TradingMode.REAL:
                service.acknowledge_live_risk('REAL_TRADING_RISK_ACKNOWLEDGED')
            key = (mode, service.storage_scope)
            store = self._environment_stores.get(key)
            if store is None:
                store = store_for_service(service, base_folder=self._environment_base)
                self._environment_stores[key] = store
            revoke = getattr(self.service, 'revoke_live_risk', None)
            if revoke:
                revoke()
            self.service, self.store = service, store
            self._clear_environment_context()
            self._sync_environment()
            self.message.setText(f'{environment_name(mode)} 선택됨 · 종목과 계좌를 새로 조회한 뒤 주문하세요.')
        except Exception as exc:
            self.message.setText(f'거래 환경 전환 실패: {exc}')
        finally:
            self._confirming = False
            self._set_busy(False)

    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(192)
        left = QVBoxLayout(self.sidebar)
        left.setContentsMargins(18, 28, 18, 20)
        left.setSpacing(13)
        brand_mark = QLabel()
        brand_mark.setPixmap(app_icon().pixmap(48, 48))
        brand_mark.setToolTip("야구의 똑딱이 타자 · 짧고 정확한 타격을 담은 DOCKDACK 심볼")
        left.addWidget(brand_mark)
        left.addWidget(label(APP_NAME, "brand"))
        left.addWidget(label(TAGLINE, "muted"))
        left.addSpacing(30)
        self.watchlist_button = QPushButton("관심종목 · 자동매매")
        self.watchlist_button.setObjectName("primary")
        self.watchlist_button.clicked.connect(self.open_watchlist)
        left.addWidget(self.watchlist_button)
        left.addWidget(label("바로가기", "section"))
        self.watch_buttons = []
        for symbol, name in (("005930", "삼성전자"), ("000660", "SK하이닉스"), ("AAPL", "애플"), ("GOOGL", "알파벳 A")):
            button = QPushButton(f"{name}\n{symbol}")
            button.setObjectName("watch")
            button.clicked.connect(lambda checked=False, value=symbol: self.select_symbol(value))
            left.addWidget(button)
            self.watch_buttons.append(button)
        left.addStretch()
        self.account_mode_badge = label("모의 계좌 연결", "badge")
        self.environment_notice = label("", "muted", wrap=True)
        left.addWidget(self.account_mode_badge)
        left.addWidget(self.environment_notice)
        outer.addWidget(self.sidebar)
        main = QVBoxLayout()
        main.setContentsMargins(26, 23, 26, 18)
        main.setSpacing(17)
        outer.addLayout(main, 1)
        heading = QHBoxLayout()
        headings = QVBoxLayout()
        headings.addWidget(label("트레이딩 데스크", "heading"))
        headings.addWidget(label("국내 · 미국주식  /  현재가, 주문, 계좌를 한곳에서", "muted"))
        heading.addLayout(headings)
        heading.addStretch()
        self.environment_selector = EnvironmentSelector(selected_mode(self.service))
        self.environment_selector.requested.connect(self.request_environment)
        heading.addWidget(self.environment_selector)
        self.window_controls.add_to(heading)
        main.addLayout(heading)

        self.search_panel = QWidget()
        search = QHBoxLayout(self.search_panel)
        search.setContentsMargins(0, 0, 0, 0)
        self.symbol_input = QLineEdit("005930")
        self.symbol_input.setPlaceholderText("종목코드 또는 티커  예: 005930, AAPL")
        self.symbol_input.returnPressed.connect(self.search)
        self.exchange_input = QComboBox()
        for text, value in (("거래소 자동", ""), ("KRX", "KRX"), ("NASDAQ", "ND"), ("NYSE", "NY"), ("AMEX", "NA")):
            self.exchange_input.addItem(text, value)
        self.search_button = QPushButton("종목 조회")
        self.search_button.setObjectName("primary")
        self.search_button.clicked.connect(self.search)
        search.addWidget(self.symbol_input, 1)
        search.addWidget(self.exchange_input)
        search.addWidget(self.search_button)
        main.addWidget(self.search_panel)

        upper = QHBoxLayout()
        upper.setSpacing(17)
        market_column = QVBoxLayout()
        quote_card, quote_layout = card()
        quote_header = QHBoxLayout()
        self.stock_name = label("종목을 조회해 주세요", "section")
        quote_header.addWidget(self.stock_name)
        quote_header.addStretch()
        self.market_tag = label("KRX · KRW", "muted")
        quote_header.addWidget(self.market_tag)
        quote_layout.addLayout(quote_header)
        self.price_label = label("—", "price")
        self.change_label = label("조회 후 가격과 등락률이 표시됩니다.", "muted")
        quote_layout.addWidget(self.price_label)
        quote_layout.addWidget(self.change_label)
        quote_layout.addStretch()
        refresh_row = QHBoxLayout()
        self.quote_time = label("아직 조회하지 않음", "muted")
        self.auto_refresh = QCheckBox("15초 자동 조회")
        self.quote_refresh = QPushButton("새로고침")
        self.quote_refresh.clicked.connect(self.refresh_quote)
        refresh_row.addWidget(self.quote_time, 1)
        refresh_row.addWidget(self.auto_refresh)
        refresh_row.addWidget(self.quote_refresh)
        quote_layout.addLayout(refresh_row)
        market_column.addWidget(quote_card, 1)
        metrics = QHBoxLayout()
        self.cash_label, self.available_label, self.profit_label = label("—", "metric"), label("—", "metric"), label("—", "metric")
        for title, value in (("예수금", self.cash_label), ("주문 가능 금액", self.available_label), ("평가 손익", self.profit_label)):
            frame, layout = card()
            layout.setContentsMargins(15, 14, 15, 14)
            layout.addWidget(label(title, "muted"))
            layout.addWidget(value)
            metrics.addWidget(frame)
        market_column.addLayout(metrics)
        upper.addLayout(market_column, 3)

        self.ticket, order_layout = card()
        self.ticket.setMinimumWidth(310)
        order_layout.addWidget(label("주문하기", "section"))
        self.order_symbol = label("선택된 종목 없음", "muted")
        order_layout.addWidget(self.order_symbol)
        self.order_kind = QComboBox()
        for text, value in (("현재가 조회 → 지정가", "current"), ("지정가 직접 입력", "limit"), ("시장가 (국내)", "market")):
            self.order_kind.addItem(text, value)
        self.order_kind.currentIndexChanged.connect(self._order_type_changed)
        order_layout.addWidget(self.order_kind)
        fields = QGridLayout()
        fields.addWidget(label("수량", "muted"), 0, 0)
        fields.addWidget(label("단가", "muted"), 0, 1)
        self.quantity = QSpinBox()
        self.quantity.setRange(1, 999999999)
        self.quantity.setValue(1)
        self.quantity.setSuffix(" 주")
        self.limit_price = QDoubleSpinBox()
        self.limit_price.setDecimals(4)
        self.limit_price.setRange(0, 999999999)
        self.limit_price.setGroupSeparatorShown(True)
        self.limit_price.setEnabled(False)
        fields.addWidget(self.quantity, 1, 0)
        fields.addWidget(self.limit_price, 1, 1)
        order_layout.addLayout(fields)
        self.order_hint = label("최신 현재가로 지정가를 준비합니다. 미국 가격 단위는 매수 내림·매도 올림하며 수량은 그대로입니다. 최종 가격을 확인하세요.", "muted", wrap=True)
        order_layout.addWidget(self.order_hint)
        order_layout.addStretch()
        buttons = QHBoxLayout()
        self.buy_button, self.sell_button = QPushButton("매수"), QPushButton("매도")
        self.buy_button.setObjectName("buy")
        self.sell_button.setObjectName("sell")
        self.buy_button.clicked.connect(lambda: self.prepare_order("buy"))
        self.sell_button.clicked.connect(lambda: self.prepare_order("sell"))
        buttons.addWidget(self.buy_button)
        buttons.addWidget(self.sell_button)
        order_layout.addLayout(buttons)
        upper.addWidget(self.ticket, 2)
        main.addLayout(upper, 3)

        account_header = QHBoxLayout()
        self.account_title = label("계좌 및 주문 내역", "section")
        account_header.addWidget(self.account_title)
        account_header.addStretch()
        self.account_refresh = QPushButton("계좌 · 주문 내역 조회")
        self.account_refresh.clicked.connect(self.refresh_account)
        account_header.addWidget(self.account_refresh)
        main.addLayout(account_header)
        self.tabs = QTabWidget()
        self.positions = table(["종목", "보유", "매도 가능", "평균 단가", "현재가", "평가 손익"])
        self.open_orders = table(["주문번호", "종목", "구분", "주문량", "체결량", "잔량", "상태"])
        self.executions = table(["주문번호", "구분", "주문량", "체결량", "체결가", "상태", "주문시각"])
        self.activity = table(["시각", "내용"])
        self.activity.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for title, widget in (("보유 종목", self.positions), ("미체결", self.open_orders), ("당일 주문·체결", self.executions), ("활동 로그", self.activity)):
            self.tabs.addTab(widget, title)
        main.addWidget(self.tabs, 3)
        self.message = label("종목을 조회하면 주문할 수 있습니다. 계좌 데이터는 조회 버튼으로 불러옵니다.", "muted", wrap=True)
        main.addWidget(self.message)
        self.statusBar().showMessage("준비됨 · 모의투자 전용 · 주문 접수와 체결은 다릅니다")
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        self.environment_selector.setEnabled(not busy)
        for widget in (self.search_panel, self.ticket, self.sidebar):
            widget.setEnabled(not busy)
        enabled = not busy and self.instrument is not None
        for widget in (self.buy_button, self.sell_button, self.quote_refresh, self.account_refresh):
            widget.setEnabled(enabled)
        if not busy:
            self._order_type_changed()

    def _order_type_changed(self):
        kind = self.order_kind.currentData()
        self.limit_price.setEnabled(kind == "limit")
        hints = {"current": "최신 현재가로 지정가를 준비합니다. 미국 가격 단위는 매수 내림·매도 올림하며 수량은 그대로입니다. 최종 가격을 확인하세요.",
                 "limit": "입력한 단가로 주문합니다. 미국 $1 이상은 소수 2자리, 미만은 4자리까지입니다. 가격을 임의로 반올림하지 않습니다.",
                 "market": "체결가격이 정해지지 않은 주문입니다. 미국 모의투자는 지정가만 지원합니다."}
        self.order_hint.setText(hints[kind])

    def log(self, message: str):
        self.activity.insertRow(0)
        self.activity.setItem(0, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.activity.setItem(0, 1, QTableWidgetItem(message))
        if self.activity.rowCount() > 200:
            self.activity.removeRow(200)

    def _run(self, name: str, operation: Callable, success: Callable) -> bool:
        if self._worker is not None or self._confirming:
            return False
        self._job_name, self._success = name, success
        self._set_busy(True)
        self.message.setText(f"{name} 중…")
        self.statusBar().showMessage(f"{name} 중 · 응답을 기다리고 있습니다")
        self._worker = Worker(operation)
        self._worker.signals.completed.connect(self._completed)
        self.pool.start(self._worker)
        return True

    @Slot(object, object)
    def _completed(self, result, error):
        callback, name = self._success, self._job_name
        self._worker, self._success = None, None
        self._set_busy(False)
        if error is not None:
            if name == "종목 조회":
                self.stock_name.setText("종목 조회 실패")
                self.change_label.setText("입력한 종목과 API 키를 확인하세요.")
            text = f"{name} 실패: {error}"
            if isinstance(error, OrderOutcomeUnknown):
                text = f"주문 접수 여부 확인 필요: {error}"
            if name == "주문 전송":
                text += " / 자동 재주문하지 않습니다. 재시도 전 주문·체결 내역을 확인하세요."
            self.message.setText(text)
            self.message.setStyleSheet("color: #ffb586;")
            self.statusBar().showMessage("오류 · 활동 로그를 확인하세요")
            self.log(text)
            return
        self.message.setStyleSheet("")
        self.message.setText(f"{name} 완료")
        self.statusBar().showMessage(f"{name} 완료 · {datetime.now():%H:%M:%S}")
        if callback is not None:
            callback(result)

    def open_watchlist(self):
        if self._worker or self._confirming:
            return
        from dockdack.watch_gui import WatchlistDialog
        self._confirming = True
        self._set_busy(True)
        try:
            dialog = WatchlistDialog(self.service, self.store, parent=self)
            dialog.exec()
            if dialog.service is not self.service:
                self.service, self.store = dialog.service, dialog.store
                self._clear_environment_context()
                self._sync_environment()
        except Exception as exc:
            self.message.setText(f"관심종목 화면을 열 수 없습니다: {exc}")
        finally:
            self._confirming = False
            self._set_busy(False)

    def select_symbol(self, symbol: str):
        self.symbol_input.setText(symbol)
        self.exchange_input.setCurrentIndex(0)
        self.search()

    def search(self):
        if self._worker or self._confirming:
            return
        symbol, exchange = self.symbol_input.text(), self.exchange_input.currentData()
        self.instrument, self.last_quote = None, None
        self.stock_name.setText("종목 조회 중")
        self.price_label.setText("—")
        self.change_label.setText("새 시세를 확인하고 있습니다.")
        self.order_symbol.setText("선택된 종목 없음")
        self.quote_time.setText("아직 조회하지 않음")
        self.limit_price.setValue(0)
        for widget in (self.positions, self.open_orders, self.executions):
            widget.setRowCount(0)
        for widget in (self.cash_label, self.available_label, self.profit_label):
            widget.setText("—")
        def load():
            instrument = self.service.resolve(symbol, exchange)
            return instrument, self.service.quote(instrument)
        self._run("종목 조회", load, self._selected)

    def _selected(self, result):
        self.instrument, quote = result
        self.symbol_input.setText(self.instrument.symbol)
        self.order_symbol.setText(f"{quote.name} · {self.instrument.symbol} / {self.instrument.currency}")
        market_name = "국내주식" if self.instrument.market is Market.DOMESTIC else "미국주식"
        self.account_title.setText(f"{market_name} 계좌 · {self.instrument.symbol} 주문 내역")
        self.limit_price.setDecimals(0 if self.instrument.market is Market.DOMESTIC else 4)
        self._show_quote(quote)
        self.limit_price.setValue(float(quote.price))
        self._set_busy(False)
        self.log(f"{quote.name} ({quote.symbol}) 종목 선택")

    def _show_quote(self, quote):
        self.last_quote = quote
        self.stock_name.setText(f"{quote.name}  {quote.symbol}")
        self.market_tag.setText(f"{quote.exchange} · {quote.currency}")
        self.price_label.setText(number(quote.price, 0 if quote.market is Market.DOMESTIC else 2))
        change = "—" if quote.change is None else f"{quote.change:+,.2f}"
        rate = "—" if quote.change_rate is None else f"{quote.change_rate:+.2f}%"
        self.change_label.setText(f"전일 대비  {change}  ({rate})")
        color = "#ed7892" if (quote.change or 0) > 0 else "#7aa2ff" if (quote.change or 0) < 0 else "#95a4bb"
        self.change_label.setStyleSheet(f"color: {color};")
        self.quote_time.setText(f"조회 {datetime.now():%H:%M:%S} · API 제공 시세")

    def refresh_quote(self):
        if self.instrument is not None:
            instrument = self.instrument
            self._run("현재가 조회", lambda: self.service.quote(instrument), self._show_quote)

    def _auto_refresh(self):
        if self.auto_refresh.isChecked() and not self._worker and not self._confirming:
            self.refresh_quote()

    def prepare_order(self, side: str):
        if self.instrument is None:
            return
        instrument, quantity = self.instrument, self.quantity.value()
        kind = self.order_kind.currentData()
        price = Decimal(str(self.limit_price.value())) if kind == "limit" else None
        self._run("주문 준비", lambda: self.service.prepare(instrument, side, quantity, kind, price), self._confirm_prepared)

    def confirm_order(self, request: OrderRequest) -> bool:
        return OrderDialog(request, self, mode=selected_mode(self.service)).exec() == QDialog.DialogCode.Accepted

    def _confirm_prepared(self, request):
        self._confirming = True
        self._set_busy(True)
        try:
            accepted = self.confirm_order(request)
        finally:
            self._confirming = False
            self._set_busy(False)
        if not accepted:
            self.log("주문 확인 취소 · 전송하지 않음")
            self.message.setText("주문을 전송하지 않았습니다.")
            return
        from dockdack.manual_orders import submit_manual_order
        reference = self.last_quote.price if self.last_quote is not None else None
        service, store = self.service, self.store
        self._run("주문 전송", lambda: submit_manual_order(service, store, request, reference), self._order_submitted)

    def _order_submitted(self, result):
        side = "매수" if result.request.side is OrderSide.BUY else "매도"
        text = f"{'실전' if result.mode is TradingMode.REAL else '모의'} {side} 접수 · {result.request.symbol} {result.request.quantity}주 · 주문번호 {result.order_number or '확인 필요'}"
        self.log(text)
        self.message.setText(text + " / 체결 상태를 조회합니다.")
        self.tabs.setCurrentWidget(self.executions)
        self.refresh_account()

    def refresh_account(self):
        if self.instrument is None:
            return
        instrument = self.instrument
        def load():
            # Retain successful sections even when another account endpoint fails.
            result = {}
            execution_reader = getattr(self.service, 'safety_executions', self.service.executions)
            for key, operation in (("account", self.service.account), ("orders", self.service.orders), ("executions", execution_reader)):
                try:
                    result[key] = operation(instrument)
                    if key == 'executions':
                        from dockdack.manual_orders import reconcile_manual_executions
                        reconcile_manual_executions(self.store, instrument, result[key])
                except Exception as exc:
                    result[key] = exc
            return result
        self._run("계좌 및 주문 내역 조회", load, self._show_account)

    @staticmethod
    def _rows(widget: QTableWidget, rows):
        widget.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                widget.setItem(row_index, column, item)

    def _show_account(self, data):
        errors = []
        account = data["account"]
        if isinstance(account, Exception):
            self.positions.setRowCount(0)
            for widget in (self.cash_label, self.available_label, self.profit_label):
                widget.setText("조회 실패")
            errors.append(f"잔고: {account}")
        else:
            places = 0 if account.currency == "KRW" else 2
            self.cash_label.setText(number(account.cash, places))
            self.available_label.setText(number(account.available_to_order, places))
            self.profit_label.setText(number(account.total_profit_loss, places))
            self._rows(self.positions, [(f"{p.name}\n{p.symbol}", number(p.quantity, 0), number(p.sellable_quantity, 0),
                                         number(p.average_price, places), number(p.current_price, places), number(p.profit_loss, places)) for p in account.positions])
        for key, widget in (("orders", self.open_orders), ("executions", self.executions)):
            if isinstance(data[key], Exception):
                widget.setRowCount(0)
                errors.append(f"{key}: {data[key]}")
            elif key == "orders":
                self._rows(widget, [(o.order_number, o.symbol, o.side, number(o.order_quantity, 0), number(o.filled_quantity, 0), number(o.remaining_quantity, 0), o.status) for o in data[key]])
            else:
                self._rows(widget, [(o.order_number, o.side, number(o.order_quantity, 0), number(o.filled_quantity, 0), number(o.fill_price, 2) if o.filled_quantity else "—", o.status, o.order_time) for o in data[key]])
        if errors:
            self.message.setText("일부 조회 실패: " + " / ".join(errors))
            self.log(self.message.text())
        else:
            self.message.setText(f"계좌·주문 내역 조회 완료 {datetime.now():%H:%M:%S} · 미체결 없음만으로 체결을 판단하지 마세요.")
            self.log("계좌 및 당일 주문·체결 내역 갱신")

    def closeEvent(self, event):
        if self._worker or self._confirming:
            self.statusBar().showMessage("진행 중인 응답을 받은 후 닫을 수 있습니다. 전송된 주문은 자동 취소되지 않습니다.")
            event.ignore()
        else:
            self.timer.stop()
            event.accept()


def main() -> int:
    from dotenv import load_dotenv
    # The executable/shortcut can start outside the repository directory.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    set_windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_branding(app)
    app.setStyle("Fusion")
    app.setFont(QFont("Malgun Gothic", 10))
    window = TradingWindow()
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1280, available.width() - 20), min(880, available.height() - 40))
    window.move(available.x() + max(0, (available.width() - window.width()) // 2),
                available.y() + max(0, (available.height() - window.height() - 30) // 2))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
