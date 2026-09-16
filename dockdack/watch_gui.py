"""Persistent watchlist, daily chart, and explicitly armed demo automation dialog."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import sqlite3
from threading import Event
from time import monotonic

from PySide6.QtCore import QPointF, QRectF, QThreadPool, QTimer, Qt, QUrl, Slot
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QHeaderView, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from dockdack.autotrade import AutoTrader, _transient_poll_failure
from dockdack.gui import STYLE, Worker, label, number, table
from dockdack.gui_service import TradingService
from dockdack.models import Market, TradingMode
from dockdack.environment_gui import EnvironmentSelector, confirm_environment, environment_name
from dockdack.environment_store import environment_base, selected_mode, store_for_service
from dockdack.trade_journal_gui import DailyTradeJournalPanel
from dockdack.operations_gui import OperationsPanel, OrderHistoryPanel, local_time
from dockdack.portfolio import PortfolioCache
from dockdack.portfolio_gui import PortfolioPanel
from dockdack.market_schedule import RankingScheduler
from dockdack.test_strategy import RandomDemoSignals
from dockdack.window_controls import WindowControls
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader, export_charts
from dockdack.signal_connection_gui import SignalConnectionPanel
from dockdack.signal_status import inspect_signal_file
from dockdack.dashboard_theme import DASHBOARD_STYLE
from dockdack.branding import APP_NAME, TAGLINE, app_icon
from dockdack.fill_recovery import FillRecovery
from dockdack.v00_widgets import SourceList, OrderToast, ActivityProgressBar
from dockdack.market_status import market_statuses
from dockdack.activity_snapshot import LedgerCollector, collect_event_logs
from dockdack.watchlist import (
    STATUS_LABELS, TRIGGER_LABELS, MarketSnapshot, TriggerKind, TriggerRule, WatchItem, WatchStore, default_store, utc_now,
)


class DailyChart(QWidget):
    """OHLC candles with volume and hover details; dense histories use a close-price line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bars = ()
        self.currency = ""
        self.setMinimumHeight(220)
        self.setMouseTracking(True)

    def set_history(self, bars, currency):
        bars = tuple(bars)
        if bars == self.bars and currency == self.currency:
            return
        self.bars, self.currency = bars, currency
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101724"))
        painter.setPen(QColor("#95a4bb"))
        if not self.bars:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "종목을 추가하고 조회하면 일봉 차트가 표시됩니다.")
            return
        area = QRectF(14, 22, max(1, self.width() - 100), max(1, self.height() - 82))
        prices = [float(value) for b in self.bars for value in (b.open, b.high, b.low, b.close)]
        low, high = min(prices), max(prices)
        pad = max((high - low) * .07, high * .005, .001)
        low, high = low - pad, high + pad
        y = lambda value: area.bottom() - (float(value) - low) / (high - low) * area.height()
        step = area.width() / len(self.bars)
        x = lambda index: area.left() + (index + .5) * step
        for i in range(5):
            value = low + (high - low) * i / 4
            painter.setPen(QPen(QColor("#263246"), 1))
            painter.drawLine(QPointF(area.left(), y(value)), QPointF(area.right(), y(value)))
            painter.setPen(QColor("#95a4bb"))
            painter.drawText(QPointF(area.right() + 6, y(value) + 4), f"{value:,.0f}" if self.currency == "KRW" else f"{value:,.2f}")
        max_volume = max(float(b.volume) for b in self.bars) or 1
        for index, bar in enumerate(self.bars):
            color = QColor("#ed7892" if bar.close >= bar.open else "#7aa2ff")
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            if len(self.bars) <= 180:
                painter.drawLine(QPointF(x(index), y(bar.high)), QPointF(x(index), y(bar.low)))
                top, bottom = sorted((y(bar.open), y(bar.close)))
                painter.drawRect(QRectF(x(index) - step * .3, top, max(1, step * .6), max(1, bottom - top)))
            elif index:
                painter.setPen(QPen(QColor("#78e6c7"), 1.4))
                painter.drawLine(QPointF(x(index - 1), y(self.bars[index - 1].close)), QPointF(x(index), y(bar.close)))
            height = float(bar.volume) / max_volume * 27
            painter.fillRect(QRectF(x(index) - max(1, step * .6) / 2, area.bottom() + 35 - height, max(1, step * .6), height), color)
        painter.setPen(QColor("#95a4bb"))
        for index in sorted({0, len(self.bars) // 2, len(self.bars) - 1}):
            painter.drawText(QPointF(max(4, min(x(index) - 22, area.right() - 35)), self.height() - 6), self.bars[index].day.strftime("%m/%d"))
        painter.drawText(14, 14, f"{self.currency} · {'일봉' if len(self.bars) <= 180 else '종가선'} / 하단 거래량 · 마우스로 OHLCV 확인")

    def mouseMoveEvent(self, event):
        if self.bars:
            index = int((event.position().x() - 14) / max(1, self.width() - 100) * len(self.bars))
            bar = self.bars[min(max(index, 0), len(self.bars) - 1)]
            self.setToolTip(f"{bar.day} · {self.currency}\n시가 {bar.open:,}  고가 {bar.high:,}\n저가 {bar.low:,}  종가 {bar.close:,}\n거래량 {bar.volume:,}")


class WatchlistDialog(QDialog):
    def __init__(self, service: TradingService, store: WatchStore | None = None, parent=None):
        super().__init__(parent)
        self.service = service
        self.store = store if store is not None else store_for_service(service)
        if self.store.mode is not selected_mode(service):
            raise ValueError('선택한 거래 환경과 저장소가 다릅니다.')
        if self.store.mode is TradingMode.REAL and self.store.storage_scope != getattr(service, 'storage_scope', None):
            raise ValueError('실전 API 인증 범위와 저장소가 다릅니다.')
        self._environment_base = environment_base(self.store)
        self._environment_stores = {(self.store.mode, self.store.storage_scope): self.store}
        self._pending_environment = None
        self._confirming_environment = False
        self.engine = AutoTrader(service, self.store)
        self.scheduler = RankingScheduler(service, self.store, clock=lambda: self.engine.clock(), stopped=lambda: self.engine._stop.is_set())
        self.test_producer = None
        self.worker = None
        self.monitoring = False
        self.pending_auto_arm = False
        self._manual_arm_pending = False
        self._manual_external_error_baseline = 0
        self._confirming_orders = False
        self._order_request_revision = 0
        self._worker_kind = ""
        self._displayed_order_status = None
        self._rule_watch_id = None
        self.snapshots, self.errors, self.fresh_ids = {}, {}, set()
        self.portfolio = PortfolioCache(service)
        self._portfolio_payload = self.portfolio.snapshot()
        self._portfolio_requested = Event()
        self._account_stop = Event()
        self.fill_recovery = FillRecovery(service, self.store, clock=lambda: self.engine.clock())
        self._fill_refresh_requested = Event()
        self._fill_recovery_status = self.fill_recovery.status()
        self._last_progress_at = None
        self._last_activity = monotonic()
        self._last_completed_at = None
        self._worker_started = None
        self._progress_text = "아직 조회하지 않음"
        self._last_worker_error = ""
        self._last_log_reload = 0
        self._last_log_error = ""
        self._items_by_id, self._watch_rows = {}, {}
        self._chart_signature = None
        self._health_warning = None
        self._last_export_at = None
        self._last_update_at = None
        self._last_export_error = ""
        self._market_summary = "시장 개장 상태 미확인 · 첫 조회 시 확인"
        self._market_open = {}
        self._market_display_open = {}
        self._market_status_minute = None
        self._notification_cursor = 0
        self._notification_initialized = False
        self._activity_worker = None
        self._activity_pending = False
        self._ledger_collector = LedgerCollector(self.store)
        self._schedule_probe = None
        self._schedule_minute = None
        self._close_when_idle = False
        self.activity_pool = QThreadPool(self)
        self.activity_pool.setMaxThreadCount(1)
        self._inspection = None
        self._inspection_worker = None
        self._inspection_config = None
        self._applied_external_config = None
        self.inspection_pool = QThreadPool(self)
        self.inspection_pool.setMaxThreadCount(1)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh_all)
        self.schedule_timer = QTimer(self)
        self.schedule_timer.setInterval(1000)
        self.schedule_timer.timeout.connect(self._schedule_wakeup)
        self.setWindowTitle(f"{APP_NAME} | 관심종목 · {environment_name(selected_mode(service))}")
        self.setWindowIcon(app_icon())
        self.resize(1360, 900)
        self.setMinimumSize(1080, 780)
        self.setObjectName("watchDashboard")
        self.setStyleSheet(STYLE + DASHBOARD_STYLE)
        self.window_controls = WindowControls(self)
        self._build()
        self._apply_execution_preferences()
        self.order_toast = OrderToast(self)
        self._sync_environment()
        self.environment_timer = QTimer(self)
        self.environment_timer.setInterval(100)
        self.environment_timer.timeout.connect(self._finish_environment_switch)
        for item in self.store.items():
            try:
                cached = self.store.cached_snapshot(item)
                if cached:
                    self.snapshots[item.id] = cached
            except Exception as exc:
                self.errors[item.id] = f"저장 데이터 오류: {exc}"
        self.reload_tables()
        # Worker-side disarming must appear even while the next API request is
        # still pending. This timer only reads state; it does not poll the broker.
        self.order_status_timer = QTimer(self)
        self.order_status_timer.setInterval(250)
        self.order_status_timer.timeout.connect(self._sync_order_controls)
        self.order_status_timer.start()
        self.health_timer = QTimer(self)
        self.health_timer.setInterval(1000)
        self.health_timer.timeout.connect(self._update_health)
        self.health_timer.start()
        self._update_health()

    def _apply_execution_preferences(self):
        self.engine.session_only_poll = True
        self.engine.enable_holdings_exits = True
        self.engine.equity_buy_percent = Decimal(str(self.buy_percent.value())) if self.percent_sizing.isChecked() else None
        self.engine.isolated_symbol_errors = True
        self.engine.holding_caps = {Market.DOMESTIC: Decimal(str(self.external_krw.value())),
                                    Market.US: Decimal(str(self.external_usd.value()))}

    def _order_notifications_worker(self, progress):
        with self.store.connection() as db:
            if not self._notification_initialized:
                self._notification_cursor = db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
                self._notification_initialized = True
                return
            rows = db.execute("SELECT e.id,e.symbol,e.message FROM events e JOIN event_category_index c ON c.event_id=e.id WHERE e.id>? AND c.category='order' ORDER BY e.id LIMIT 50",
                              (self._notification_cursor,)).fetchall()
        if rows:
            self._notification_cursor = rows[-1]['id']
            progress(('notifications', tuple(f"{row['symbol'].split(':')[-1]} · {row['message']}" for row in rows)))

    def _sync_environment(self):
        mode = selected_mode(self.service)
        name = environment_name(mode)
        self.environment_selector.apply(mode, pending=self._pending_environment is not None)
        self.environment_caption.setText(f'{TAGLINE}   /   {name}')
        self.portfolio_panel.heading.setText(f'현재 보유종목 · {name} 계좌')
        self.order_history_panel.heading.setText(f'실제 주문·체결 내역 · {name} 계좌')
        self.setWindowTitle(f'{APP_NAME} | 관심종목 · {name}')
        self.environment_notice.setText(
            '실전 · 실제 자금 사용 / 내장 랜덤 모의 신호기 차단 / 실전 API 키와 DOCKDACK_ALLOW_LIVE_ORDERS=true 필요 / 전환만으로 주문 ON 안 됨'
            if mode is TradingMode.REAL else '모의 · 가상 자금 사용 / 실전과 잔고·규칙·매매일지 분리')
        self.environment_notice.setStyleSheet('color: #ffad9d;' if mode is TradingMode.REAL else '')
        self.random_demo.setEnabled(mode is TradingMode.DEMO)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.strategy_panel), mode is TradingMode.DEMO)
        self.arm_button.setToolTip(f'전체 조회 검증 후 {name} 자동주문을 켭니다. 실전은 별도 위험 확인과 API 설정이 필요합니다.')

    @Slot(object)
    def request_environment(self, mode):
        mode = TradingMode(mode)
        if mode is selected_mode(self.service) or self._pending_environment or self._confirming_environment:
            return
        # OFF is immediate even while the warning's nested Qt loop is open.
        self.stop_monitoring()
        self._confirming_environment = True
        self.update_controls()
        try:
            confirmed = confirm_environment(self, mode)
            if not confirmed:
                self.message.setText('거래 환경 전환 취소 · 기존 환경 유지 · 자동주문 OFF')
                return
            candidate = TradingService(mode=mode)
            if mode is TradingMode.REAL:
                candidate.acknowledge_live_risk('REAL_TRADING_RISK_ACKNOWLEDGED')
            key = (mode, candidate.storage_scope)
            candidate_store = self._environment_stores.get(key)
            if candidate_store is None:
                candidate_store = store_for_service(candidate, base_folder=self._environment_base)
                self._environment_stores[key] = candidate_store
            revoke = getattr(self.service, 'revoke_live_risk', None)
            if revoke:
                revoke()
            self._pending_environment = (candidate, candidate_store)
            self.environment_selector.apply(selected_mode(self.service), pending=True)
            self.message.setText('환경 전환 대기 · 진행 중인 조회 응답을 마무리합니다. 새 주문은 차단했습니다.')
            self.environment_timer.start()
        except Exception as exc:
            self.message.setText(f'환경 전환 실패 · 자동주문 OFF: {exc}')
        finally:
            self._confirming_environment = False
            self.update_controls()
        self._finish_environment_switch()

    @Slot()
    def _finish_environment_switch(self):
        if (not self._pending_environment or self.worker is not None or self._inspection_worker is not None
                or self._activity_worker is not None or self._schedule_probe is not None):
            return
        service, store = self._pending_environment
        self._pending_environment = None
        self.environment_timer.stop()
        self.service, self.store = service, store
        self.engine = AutoTrader(service, store)
        self._market_open = {}
        self._market_display_open = {}
        self._market_status_minute = None
        self._notification_initialized = False
        self._ledger_collector = LedgerCollector(store)
        self._schedule_minute = None
        self.scheduler = RankingScheduler(service, store, clock=lambda: self.engine.clock(), stopped=lambda: self.engine._stop.is_set())
        self.portfolio = PortfolioCache(service)
        self._portfolio_payload = self.portfolio.snapshot()
        self.fill_recovery = FillRecovery(service, store, clock=lambda: self.engine.clock())
        self._fill_recovery_status = self.fill_recovery.status()
        self._portfolio_requested.clear()
        self._fill_refresh_requested.clear()
        self._account_stop.clear()
        self.test_producer = None
        self.monitoring = self.pending_auto_arm = self._manual_arm_pending = False
        self._order_request_revision += 1
        self._manual_external_error_baseline = 0
        self._displayed_order_status = None
        self.snapshots, self.errors, self.fresh_ids = {}, {}, set()
        self._chart_signature = self._rule_watch_id = None
        self._last_worker_error = self._last_log_error = self._last_export_error = ''
        self._last_export_at = self._last_update_at = self._inspection = self._applied_external_config = None
        self._last_progress_at = self._last_completed_at = self._worker_started = None
        self._last_activity = monotonic()
        self._progress_text = '환경 전환 후 조회 대기'
        self._market_summary = '시장 개장 상태 미확인 · 첫 조회 시 확인'
        self.random_demo.setChecked(False)
        if hasattr(self, '_saved_external_config'):
            del self._saved_external_config
        self.external_mode.setChecked(False)
        self.external_source.setText('external-model')
        self.external_krw.setValue(0)
        self.external_usd.setValue(0)
        self.external_quantity.setValue(1)
        self.additional_sources.table.setRowCount(0)
        self.portfolio_panel._live_quotes.clear()
        self.portfolio_panel.set_exit_targets({})
        self.order_toast.hide()
        folder = store.path.parent / 'exchange'
        self.signal_path.setText(str(folder / 'signals.json'))
        self.chart_path.setText(str(folder / 'charts.json'))
        self.portfolio_panel.apply(self._portfolio_payload)
        self.order_history_panel._ledger = None
        self.order_history_panel.records = ()
        self.order_history_panel.table.setRowCount(0)
        self.order_history_panel.set_recovery_status(self._fill_recovery_status)
        self.trade_journal_panel.store = store
        # Do not leave DEMO logs visible beneath a REAL badge when the new
        # environment's asynchronous log query fails or has identical IDs.
        for view in (*self.operations_panel.logs.values(), self.order_history_panel.audit):
            view.apply_events((), head=None)
        self.operations_panel.flow.setText('최근 감시 기록 — · 최근 신호 수신 —')
        from dockdack.activity_snapshot import LedgerSnapshot
        from dockdack.trade_journal import daily_trade_journal
        empty = LedgerSnapshot(None, (), daily_trade_journal(()))
        self.order_history_panel.apply_snapshot(empty, force=True)
        self.trade_journal_panel.apply_snapshot(empty, force=True)
        self.sweep_progress.set_activity('관심종목 시세·차트 조회 · 환경 전환 후 대기')
        self._sync_environment()
        self.reload_tables()
        self._reload_activity(force=True)
        self._update_connection()
        self._update_health()
        self.message.setText(f'{environment_name(selected_mode(service))} 선택됨 · 감시/자동주문 OFF · API 설정 확인 후 새로 조회하세요.')

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 12)
        layout.setSpacing(8)
        title = QHBoxLayout()
        self.brand_mark = QLabel()
        self.brand_mark.setPixmap(app_icon().pixmap(46, 46))
        self.brand_mark.setFixedSize(48, 48)
        self.brand_mark.setAccessibleName("DOCKDACK 야구공과 배트 심볼")
        self.brand_mark.setToolTip("똑딱이 타자처럼, 짧고 정확하게. DOCKDACK")
        title.addWidget(self.brand_mark)
        brand = QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(label(APP_NAME + '  ver 0.0', "heading"))
        self.environment_caption = label(f"{TAGLINE}   /   모의투자", "eyebrow")
        brand.addWidget(self.environment_caption)
        title.addLayout(brand)
        title.addStretch()
        self.environment_selector = EnvironmentSelector(selected_mode(self.service))
        self.environment_selector.requested.connect(self.request_environment)
        title.addWidget(self.environment_selector)
        self.mode_label = label("자동주문 OFF · 주문 차단", "badge")
        title.addWidget(self.mode_label)
        self.window_controls.add_to(title)
        layout.addLayout(title)
        self.environment_notice = label('', 'muted', wrap=True)
        layout.addWidget(self.environment_notice)
        self.environment_notice.hide()
        status_line = QHBoxLayout()
        self.monitoring_label = label("감시 중지", "muted")
        self.order_status_detail = label("시세 감시와 자동주문이 중지되어 있습니다.", "muted", wrap=True)
        status_line.addWidget(self.monitoring_label)
        status_line.addWidget(self.order_status_detail, 1)
        layout.addLayout(status_line)
        market_line = QHBoxLayout()
        self.market_labels = {}
        for market, title in ((Market.DOMESTIC, '한국'), (Market.US, '미국')):
            badge = label(f'{title} · 장 시간 확인 중', 'connectionMode')
            self.market_labels[market] = badge
            market_line.addWidget(badge)
        market_line.addWidget(label('정규장 기준 · 거래 시간은 마우스를 올려 확인', 'muted'))
        market_line.addStretch()
        layout.addLayout(market_line)
        self.health_label = label("앱 응답 확인 중 · API 상태 미확인", "muted", wrap=True)
        layout.addWidget(self.health_label)
        self.health_label.hide()  # Full diagnostics live in the server/log page.

        control_card = QFrame()
        control_card.setObjectName("controlBar")
        controls = QHBoxLayout(control_card)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(8)
        self.interval = QSpinBox()
        self.interval.setRange(15, 3600)
        self.interval.setValue(30)
        self.interval.setSuffix(" 초")
        controls.addWidget(label("순회 후 대기", "muted"))
        controls.addWidget(self.interval)
        self.interval.setFixedWidth(100)
        self.interval.setToolTip("전체 관심종목을 순서대로 한 번 조회한 뒤 쉬는 시간입니다. 각 종목의 갱신 주기가 30초라는 뜻이 아닙니다.")
        self.refresh_button = QPushButton("전체 1회 조회")
        self.start_button = QPushButton("감시 시작 (조회만)")
        self.arm_button = QPushButton("자동주문 켜기 (ON)")
        self.disarm_button = QPushButton("자동주문 끄기 (OFF)")
        self.arm_button.setToolTip("확인 후 새 모의주문 전송을 허용합니다. 정규장·신호·잔고 조건도 충족해야 합니다.")
        self.disarm_button.setToolTip("새 자동주문과 예정된 자동 활성화를 차단합니다. 시세 감시는 계속되며 전송 중이거나 접수된 주문은 취소하지 않습니다.")
        self.stop_button = QPushButton("감시·주문 중지")
        self.start_button.setObjectName("primary")
        self.arm_button.setObjectName("armOrders")
        self.disarm_button.setObjectName("disarmOrders")
        self.refresh_button.clicked.connect(self.refresh_all)
        self.start_button.clicked.connect(self.start_monitoring)
        self.arm_button.clicked.connect(self.enable_auto_orders)
        self.disarm_button.clicked.connect(self.disable_auto_orders)
        self.stop_button.clicked.connect(self.stop_monitoring)
        for button in (self.refresh_button, self.start_button, self.arm_button, self.disarm_button, self.stop_button):
            button.setAutoDefault(False)
            controls.addWidget(button)
        layout.addWidget(control_card)

        flow = QHBoxLayout()
        self.connection_summary = label("신호 연결 미설정", "muted", wrap=True)
        flow.addWidget(self.connection_summary, 1)
        self.connection_shortcut = QPushButton("신호 연결 확인  →")
        self.connection_shortcut.setObjectName("linkButton")
        self.connection_shortcut.setAutoDefault(False)
        flow.addWidget(self.connection_shortcut)
        layout.addLayout(flow)
        self.sweep_progress = ActivityProgressBar()
        self.sweep_progress.setObjectName("sweepProgress")
        self.sweep_progress.setRange(0, 100)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setTextVisible(True)
        self.sweep_progress.setFormat('관심종목 시세·차트 조회 · 대기')
        self.sweep_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sweep_progress.setFixedHeight(24)
        self.sweep_progress.setAccessibleName('관심종목 시세·차트 조회 / 보유종목 매도 조건 확인 진행률 · 체결 표시 아님')
        layout.addWidget(self.sweep_progress)

        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setObjectName("workspaceTabs")
        self.portfolio_panel = PortfolioPanel()
        self.portfolio_panel.request_refresh.connect(self.refresh_portfolio)
        self.order_history_panel = OrderHistoryPanel()
        self.order_history_panel.request_refresh.connect(self.refresh_executions)
        self.trade_journal_panel = DailyTradeJournalPanel(self.store)
        self.operations_panel = OperationsPanel()
        self.watch_page = QWidget()
        watch_layout = QVBoxLayout(self.watch_page)
        self.workspace_tabs.addTab(self.portfolio_panel, "보유종목")
        self.workspace_tabs.addTab(self.order_history_panel, "실제 주문·체결")
        self.workspace_tabs.addTab(self.trade_journal_panel, "매매일지")
        self.workspace_tabs.addTab(self.operations_panel, "서버·감시 로그")
        self.workspace_tabs.addTab(self.watch_page, "관심종목·차트")
        layout.addWidget(self.workspace_tabs, 1)

        data_controls = QHBoxLayout()
        self.ranking_button = QPushButton("거래대금 TOP100 추가 · 한국 + 미국")
        self.ranking_button.setToolTip("시장별 일반 기업 보통주만 선정 · ETF/ETN/펀드/우선주/리츠/스팩 및 분류 불명 종목 제외")
        self.export_button = QPushButton("차트 JSON 내보내기")
        self.ranking_button.clicked.connect(self.add_top100)
        self.export_button.clicked.connect(self.export_json)
        data_controls.addWidget(self.ranking_button)
        data_controls.addWidget(self.export_button)
        self.hourly_ranking = QCheckBox("장중 개장·매 정시 TOP100 재선정")
        self.hourly_ranking.setChecked(True)
        self.hourly_ranking.setToolTip("감시 시작 후에만 작동 · 한국/미국 현지 정규장 · 휴장/조기폐장 반영 · 주문 활성화와 별도")
        data_controls.addWidget(self.hourly_ranking)
        watch_layout.addLayout(data_controls)

        self.edit_panel = QWidget()
        edit = QHBoxLayout(self.edit_panel)
        edit.setContentsMargins(0, 0, 0, 0)
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("종목코드 / 티커")
        self.exchange_input = QComboBox()
        for text, code in (("거래소 자동", ""), ("KRX", "KRX"), ("NASDAQ", "ND"), ("NYSE", "NY"), ("AMEX", "NA")):
            self.exchange_input.addItem(text, code)
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 1000)
        self.days_input.setValue(30)
        self.days_input.setSuffix(" 거래일")
        self.add_button = QPushButton("관심종목 추가")
        self.days_button = QPushButton("선택 종목 기간 적용")
        self.remove_button = QPushButton("제외")
        self.add_button.clicked.connect(self.add_item)
        self.days_button.clicked.connect(self.apply_days)
        self.remove_button.clicked.connect(self.remove_item)
        for widget in (self.symbol_input, self.exchange_input, self.days_input, self.add_button, self.days_button, self.remove_button):
            edit.addWidget(widget)
        watch_layout.addWidget(self.edit_panel)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setMinimumHeight(290)
        watch_box = QWidget()
        watch_box_layout = QVBoxLayout(watch_box)
        watch_box_layout.setContentsMargins(0, 0, 0, 0)
        self.watch_market_tabs = QTabWidget()
        self.watch_tables = {}
        for market, title in ((Market.DOMESTIC, "한국 · KRW"), (Market.US, "미국 · USD")):
            view = table(["종목", "현재가", "N일", "상태/조회시각"])
            view.setMinimumWidth(480)
            view.verticalHeader().setDefaultSectionSize(52)
            for column, width in ((0, 125), (1, 130), (2, 45)):
                view.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                view.setColumnWidth(column, width)
            view.itemSelectionChanged.connect(self.select_item)
            self.watch_tables[market] = view
            self.watch_market_tabs.addTab(view, title)
        self.watch_market_tabs.currentChanged.connect(self.select_item)
        self.watch_market_tabs.setToolTip("화면 표시만 시장별로 분리합니다. 감시와 정시 재선정은 양쪽 시장을 계속 처리합니다.")
        watch_box_layout.addWidget(self.watch_market_tabs, 1)
        self.watch_market_hint = label("", "muted", wrap=True)
        watch_box_layout.addWidget(self.watch_market_hint)
        split.addWidget(watch_box)
        chart_box = QWidget()
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(4, 0, 0, 0)
        self.chart_title = label("관심종목을 선택하세요", "section")
        chart_layout.addWidget(self.chart_title)
        chart_tabs = QTabWidget()
        self.chart = DailyChart()
        self.bar_table = table(["거래일", "시가", "고가", "저가", "종가", "거래량"])
        chart_tabs.addTab(self.chart, "일봉 차트")
        chart_tabs.addTab(self.bar_table, "일봉 데이터")
        chart_layout.addWidget(chart_tabs)
        chart_box.setMinimumHeight(290)
        split.addWidget(chart_box)
        split.setSizes([490, 650])
        watch_layout.addWidget(split, 1)

        self.rule_panel = QWidget()
        form = QGridLayout(self.rule_panel)
        form.setContentsMargins(0, 0, 0, 0)
        self.rule_label = label("선택 종목의 1회성 규칙 등록", "section")
        form.addWidget(self.rule_label, 0, 0, 1, 7)
        self.trigger = QComboBox()
        for kind, text in TRIGGER_LABELS.items():
            if kind is not TriggerKind.EXTERNAL:
                self.trigger.addItem(text, kind.value)
        self.trigger.currentIndexChanged.connect(self.trigger_changed)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 999999999)
        self.threshold.setDecimals(4)
        self.period = QSpinBox()
        self.period.setRange(2, 999)
        self.period.setValue(20)
        self.period.setSuffix(" 일")
        self.side = QComboBox()
        self.side.addItem("매수", "buy")
        self.side.addItem("매도", "sell")
        self.quantity = QSpinBox()
        self.quantity.setRange(1, 999999999)
        self.quantity.setSuffix(" 주")
        self.max_notional = QDoubleSpinBox()
        self.max_notional.setRange(0, 999999999999)
        self.max_notional.setDecimals(2)
        self.max_notional.setGroupSeparatorShown(True)
        self.rule_button = QPushButton("규칙 등록")
        self.rule_button.clicked.connect(self.add_rule)
        for column, (caption, widget) in enumerate((("조건", self.trigger), ("기준 가격", self.threshold),
                    ("이동평균 N", self.period), ("방향", self.side), ("수량", self.quantity),
                    ("주문금액 상한 (필수)", self.max_notional), ("", self.rule_button))):
            form.addWidget(label(caption, "muted"), 1, column)
            form.addWidget(widget, 2, column)
        self.trigger_changed()

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.tabs.setMinimumHeight(145)
        rules_box = QWidget()
        rules_layout = QVBoxLayout(rules_box)
        rules_layout.setContentsMargins(0, 3, 0, 0)
        rules_layout.addWidget(self.rule_panel)
        self.rules_table = table(["종목", "조건", "방향", "수량", "금액 상한", "상태"])
        for column in (0, 2, 3, 4, 5):
            self.rules_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        rules_layout.addWidget(self.rules_table)
        rule_actions = QHBoxLayout()
        self.pause_button = QPushButton("선택 규칙 비활성화")
        self.review_button = QPushButton("내역 확인 후 종목 차단 해제")
        self.pause_button.clicked.connect(self.pause_rule)
        self.review_button.clicked.connect(self.review_attempt)
        rule_actions.addWidget(self.pause_button)
        rule_actions.addWidget(self.review_button)
        rule_actions.addStretch()
        rules_layout.addLayout(rule_actions)
        self.rules_box = rules_box
        # Compatibility alias, now explicitly a server log rather than mixed trading records.
        self.log_table = self.operations_panel.logs["system"].table
        self.tabs.addTab(rules_box, "트리거 규칙")
        self._build_external_tab()
        self._build_strategy_tab()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.signal_connection_panel = SignalConnectionPanel()
        self.signal_connection_panel.setObjectName("signalConnectionPanel")
        self.signal_connection_page = QScrollArea()
        self.signal_connection_page.setWidgetResizable(True)
        self.signal_connection_page.setFrameShape(QFrame.Shape.NoFrame)
        self.signal_connection_page.setWidget(self.signal_connection_panel)
        self.signal_connection_panel.request_settings.connect(self.open_connection_settings)
        self.signal_connection_panel.request_inspect.connect(self.inspect_signals)
        self.signal_connection_panel.request_folder.connect(self.open_connection_folder)
        self.connection_shortcut.clicked.connect(lambda: self.workspace_tabs.setCurrentWidget(self.signal_connection_page))
        self.workspace_tabs.addTab(self.signal_connection_page, "외부 신호 연결")
        self.workspace_tabs.addTab(self.tabs, "매매 설정")
        self.message = label("감시는 꺼져 있습니다. 조건·수량·금액 상한을 직접 설정한 후 시작하세요.", "muted", wrap=True)
        layout.addWidget(self.message)
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        self.operations_panel.tabs.currentChanged.connect(lambda _: self._reload_activity(visible_force=True))
        self.order_history_panel.tabs.currentChanged.connect(lambda _: self._reload_activity(visible_force=True))
        self.update_controls()

    def update_controls(self):
        busy = self.worker is not None
        editing = not busy and not self.monitoring and not self.pending_auto_arm and not self._confirming_orders and not self._pending_environment and not self._confirming_environment
        for widget in (self.edit_panel, self.rule_panel, self.interval, self.pause_button, self.review_button,
                       self.ranking_button, self.export_button, self.external_panel, self.hourly_ranking, self.strategy_panel):
            widget.setEnabled(editing)
        self.refresh_button.setEnabled(editing)
        self.start_button.setEnabled(editing)
        self.random_demo.setEnabled(editing and selected_mode(self.service) is TradingMode.DEMO)
        self._sync_order_controls()

    def automation_status(self):
        """One snapshot of engine permission, shared by display and session JSON."""
        enabled = self.engine.orders_enabled
        pending = self.pending_auto_arm and not enabled
        state = "armed" if enabled else "warmup_orders_off" if pending else "orders_off" if self.monitoring else "stopped"
        return {"monitoring": self.monitoring, "orders_enabled": enabled, "pending_arm": pending, "state": state,
                "trading_mode": selected_mode(self.service).value}

    def set_pending_auto_arm(self, enabled: bool):
        self.pending_auto_arm = bool(enabled)
        if not enabled:
            self._manual_arm_pending = False
        self._sync_order_controls()

    @Slot()
    def _sync_order_controls(self, status=None):
        status = self.automation_status() if status is None else status
        armed, pending, monitoring = status["orders_enabled"], status["pending_arm"], status["monitoring"]
        busy = self.worker is not None
        # Raw pending matters before monitoring starts too: OFF must always be
        # capable of cancelling the launcher's one-time activation request.
        display_key = (armed, self.pending_auto_arm, monitoring, busy, self._confirming_orders,
                       selected_mode(self.service), self._pending_environment, self._confirming_environment)
        if display_key == self._displayed_order_status:
            return
        self._displayed_order_status = display_key
        self.arm_button.setEnabled(not armed and not self.pending_auto_arm and not self._confirming_orders
                                   and not self._pending_environment and not self._confirming_environment)
        self.disarm_button.setEnabled(armed or self.pending_auto_arm)
        self.stop_button.setEnabled(busy or monitoring or armed or self.pending_auto_arm)
        if armed:
            self.arm_button.setToolTip("현재 자동주문이 ON입니다. 끄려면 오른쪽 OFF 버튼을 누르세요.")
        elif self.pending_auto_arm:
            self.arm_button.setToolTip("전체 조회 성공 후 자동으로 ON이 됩니다. 다시 누를 필요가 없으며 OFF 버튼으로 예약을 취소할 수 있습니다.")
        elif not monitoring:
            self.arm_button.setToolTip("한 번 확인하면 감시·전체 조회부터 시작하고 성공한 뒤 자동으로 ON이 됩니다.")
        elif busy:
            self.arm_button.setToolTip("조회 중에도 예약할 수 있습니다. 확인 후 전체 조회가 성공하면 자동으로 ON이 됩니다.")
        else:
            self.arm_button.setToolTip(f"확인 후 전체 조회를 한 번 진행하고 성공하면 {environment_name(selected_mode(self.service))} 주문 전송을 허용합니다.")
        self.mode_label.setText("자동주문 ON · 주문 허용" if armed else "자동주문 OFF · 주문 차단")
        self.monitoring_label.setText("감시 중 · 시세/차트 갱신" if monitoring else "감시 중지")
        if armed:
            detail = f"정규장에 유효한 신호와 주문 조건을 충족하면 {environment_name(selected_mode(self.service))} 주문을 전송합니다."
            colors = ("#78e6c7", "#173f3c", "#286357")
            if selected_mode(self.service) is TradingMode.REAL:
                colors = ('#ffbfad', '#4b2625', '#aa685c')
        elif pending:
            detail = "전체 조회 완료 후 ON 예정 · 다시 누를 필요 없음 · OFF로 예약 취소 · 조회 실패 시 OFF 유지"
            colors = ("#ffda91", "#41341e", "#80683c")
        else:
            detail = "조회만 진행합니다. 새 자동주문은 차단하며, 이미 요청한 주문은 체결될 수 있습니다." if monitoring else "시세 감시와 자동주문이 중지되어 있습니다."
            colors = ("#c0c9d8", "#263140", "#46546a")
        self.order_status_detail.setText(detail)
        self.mode_label.setStyleSheet(f"color: {colors[0]}; background: {colors[1]}; border: 1px solid {colors[2]}; border-radius: 8px; padding: 7px 12px; font-weight: 700;")
        self.mode_label.setAccessibleName(self.mode_label.text())

    def _tab_changed(self, index):
        self.rule_panel.setVisible(self.tabs.widget(index) is self.rules_box)

    def _external_config_key(self):
        return (self.external_mode.isChecked(), self.random_demo.isChecked(), self.external_source.text().strip(),
                self.signal_path.text().strip(), self.chart_path.text().strip(), self.external_quantity.value(),
                self.external_krw.value(), self.external_usd.value(), self.random_us.currentData(),
                self.percent_sizing.isChecked(), self.buy_percent.value(), self.additional_sources.raw_sources())

    def connection_status(self):
        """Only observed in-memory worker state; no filesystem, DB or broker I/O."""
        reader = self.engine.external_reader
        applied = self._applied_external_config
        current = self._external_config_key()
        config = applied if applied and self.monitoring else current
        external, random, source, inbox, outbox, *_ = config
        active = bool(self.monitoring and self.engine.external_only and reader)
        blocked = self.engine.external_error or self._last_export_error or self._last_worker_error
        if not blocked and self.pending_auto_arm:
            blocked = "전체 관심종목·잔고 검증 대기"
        elif not blocked and not self.engine.orders_enabled:
            blocked = "사용자 주문 허용 OFF"
        return {"configured": external, "active": active, "producer": "random-demo" if random else "external-file" if external else "manual",
                "source_id": source, "input_path": inbox, "output_path": outbox,
                "update_path": str(Path(outbox).with_name(Path(outbox).stem + "_updates")) if outbox else "",
                "configuration_pending": applied is not None and current != applied,
                "last_export_at": self._last_export_at, "last_update_at": self._last_update_at,
                "market_summary": self._market_summary,
                "block_reason": blocked, "inspection": self._inspection if self._inspection_config == config else None,
                "inspection_busy": self._inspection_worker is not None,
                **self.automation_status(), **(reader.status() if reader else {})}

    def _update_connection(self):
        status = self.connection_status()
        self.signal_connection_panel.set_status(status)
        producer = "내장 모의 신호기" if status["producer"] == "random-demo" else "외부 JSON 신호" if status["configured"] else "수동 규칙"
        state = "수신 오류" if status.get("reader_error") else "입력 파일 대기" if status.get("reader_state") == "missing" else "신호 수신 중" if status["active"] else "수신 중지"
        text = f"{producer}  ·  {state}"
        if self.connection_summary.text() != text:
            self.connection_summary.setText(text)
        if hasattr(self, 'source_status'):
            states = getattr(self.engine, 'external_sources', {})
            errors = getattr(self.engine, 'external_source_errors', {})
            self.source_status.setText(' · '.join(f"{source}: {'오류·해당 출처 보류' if source in errors else reader.status().get('reader_state', '대기')}"
                                                  for source, (_, reader) in states.items()) or '추가 연결 없음')

    def open_connection_settings(self):
        self.workspace_tabs.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.external_panel)
        if self.monitoring or self.worker:
            self.message.setText("현재 적용 설정입니다. 연결을 변경하려면 ‘감시·주문 중지’ 후 작업 종료를 기다리세요. 설정 변경만으로 주문이 켜지지 않습니다.")

    def open_connection_folder(self):
        folder = Path(self.connection_status()["output_path"]).expanduser().resolve().parent
        if not folder.is_dir():
            self.message.setText(f"연결 폴더가 아직 없습니다: {folder} · 차트 저장 후 생성됩니다.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self.message.setText(f"폴더를 열지 못했습니다: {folder}")

    def inspect_signals(self):
        """Inspect local JSON in a separate file-only worker, even during quotes."""
        if self._inspection_worker:
            return
        status = self.connection_status()
        if not status["configured"]:
            self.message.setText("연결 설정에서 외부 신호 모드를 먼저 선택하세요.")
            return
        config = self._applied_external_config if self.monitoring and self._applied_external_config else self._external_config_key()
        _, random, source, inbox, _, quantity, krw, usd, _ = config[:9]
        try:
            policy = ExternalPolicy(source, quantity, Decimal(str(krw)), Decimal(str(usd)), allow_market=random)
        except Exception as exc:
            self.message.setText(f"연결 설정 검사 실패: {exc}")
            return
        worker = Worker(lambda: inspect_signal_file(inbox, policy, mode=selected_mode(self.service)))
        self._inspection_config = config
        self._inspection_worker = worker
        worker.signals.completed.connect(self._inspection_completed)
        self._update_connection()
        self.inspection_pool.start(worker)

    @Slot(object, object)
    def _inspection_completed(self, result, error):
        self._inspection_worker = None
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
            return
        config = self._applied_external_config if self.monitoring and self._applied_external_config else self._external_config_key()
        if self._inspection_config != config:
            self._inspection = None
            self._update_connection()
            self.message.setText("검사 중 연결 설정이 변경되어 이전 파일 검사 결과를 표시하지 않습니다. 새 경로로 다시 검사하세요.")
            return
        self._inspection = result if error is None else {"state": "error", "checked_at": utc_now(), "summary": str(error)}
        self._update_connection()
        self.message.setText(f"파일 검사 · {self._inspection['summary']} · 신호 접수/주문 없음")

    def _build_strategy_tab(self):
        self.strategy_panel = QWidget()
        grid = QGridLayout(self.strategy_panel)
        self.random_demo = QCheckBox("내장 모의 테스트 신호기 · 미보유 10% 매수 / +1% 익절 · -0.8% 손절")
        self.random_us = QComboBox()
        self.random_us.addItem("미국 주문 차단 (모의 시장가 미지원)", "blocked")
        self.random_us.addItem("미국 현재가 재조회 → 지정가", "limit")
        self.random_demo.toggled.connect(self._random_toggled)
        grid.addWidget(self.random_demo, 0, 0, 1, 2)
        grid.addWidget(label("미국 처리 방식", "muted"), 1, 0)
        grid.addWidget(self.random_us, 1, 1)
        grid.addWidget(label("국내 매수는 시장가 · 매수 수량은 설정한 평가자산 비중 또는 고정 수량 적용\n"
                             "보유분 매도는 상방·하방 가격을 별도로 확인하며 금액 상한과 매도 가능 수량 적용\n"
                             "같은 입력은 다시 추첨하지 않습니다. 이미 보유하면 추가매수하지 않습니다.\n"
                             "실제 평균 매입가 대비 +1% 이상 익절 / -0.8% 이하 손절 신호입니다.\n"
                             "호출 지연·가격 변동·미체결로 해당 수익률에서의 체결은 보장되지 않습니다 (수수료·세금 별도).\n"
                             "외부 신호 탭에서 KRW/USD 상한 설정 → 자동주문 켜기 확인 → 전체 조회 성공 후 ON.", "muted", wrap=True), 2, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        self.tabs.addTab(self.strategy_panel, "모의 테스트 신호기")

    def _random_toggled(self, enabled):
        if enabled and selected_mode(self.service) is TradingMode.REAL:
            self.random_demo.setChecked(False)
            self.message.setText('실전에서는 내장 랜덤 모의 신호기를 사용할 수 없습니다.')
            return
        if enabled:
            self._saved_external_config = (self.external_mode.isChecked(), self.external_source.text(), self.signal_path.text())
            self.external_mode.setChecked(True)
            self.external_source.setText("random-demo")
            self.signal_path.setText(str(self.store.path.parent / "exchange" / "random-signals.json"))
        elif hasattr(self, "_saved_external_config"):
            mode, source, path = self._saved_external_config
            self.external_mode.setChecked(mode)
            self.external_source.setText(source)
            self.signal_path.setText(path)

    def _build_external_tab(self):
        self.external_panel = QWidget()
        outer = QVBoxLayout(self.external_panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName('externalSignalSettingsContent')
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet('QWidget#externalSignalSettingsContent { background: #121b2a; }')
        scroll.setStyleSheet('QScrollArea { background: #121b2a; border: none; }')
        grid = self.external_grid = QGridLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.external_mode = QCheckBox("외부 신호 모드 (수동 트리거 실행 안 함)")
        self.external_source = QLineEdit("external-model")
        self.external_quantity = QSpinBox()
        self.external_quantity.setRange(1, 999999999)
        self.external_quantity.setValue(999999999)
        self.external_quantity.setSuffix(" 주 상한")
        self.external_quantity.setToolTip('별도 수량 상한입니다. 비중 매수는 10% 예산·주문당 금액·가용액을 우선 적용하므로 이 수량을 그대로 매수하지 않습니다.')
        self.external_krw, self.external_usd = QDoubleSpinBox(), QDoubleSpinBox()
        for field, currency in ((self.external_krw, "KRW"), (self.external_usd, "USD")):
            field.setRange(0, 999999999999)
            field.setDecimals(2)
            field.setSuffix(f" {currency} / 주문")
            field.setGroupSeparatorShown(True)
        folder = self.store.path.parent / "exchange"
        self.signal_path = QLineEdit(str(folder / "signals.json"))
        self.chart_path = QLineEdit(str(folder / "charts.json"))
        self.read_signals_button = QPushButton("신호 파일 1회 읽기 (주문 안 함)")
        self.read_signals_button.clicked.connect(self.read_signals)
        grid.addWidget(self.external_mode, 0, 0, 1, 2)
        grid.addWidget(label("source_id", "muted"), 0, 2)
        grid.addWidget(self.external_source, 0, 3)
        grid.addWidget(self.external_quantity, 0, 4)
        grid.addWidget(self.external_krw, 1, 3)
        grid.addWidget(self.external_usd, 1, 4)
        grid.addWidget(label("신호 JSON 입력", "muted"), 1, 0)
        grid.addWidget(self.signal_path, 1, 1, 1, 2)
        grid.addWidget(label("차트 JSON 출력", "muted"), 2, 0)
        grid.addWidget(self.chart_path, 2, 1, 1, 2)
        grid.addWidget(self.read_signals_button, 2, 3, 1, 2)
        grid.addWidget(label("외부 모드: 종목별 즉시 전송 + 순회 후 전체 파일 갱신 · 과거 일봉 DB 재사용 / 당일 봉 증분 조회\n"
                             "금액 0은 해당 시장 차단 · 파일 읽기는 주문 활성화가 아님 · 입력/상한은 이번 창에서만 유지", "muted", wrap=True), 3, 0, 1, 5)
        grid.setColumnStretch(1, 1)
        self.percent_sizing = QCheckBox('평가자산 비중으로 매수')
        self.percent_sizing.setChecked(True)
        self.buy_percent = QDoubleSpinBox()
        self.buy_percent.setRange(0.01, 100)
        self.buy_percent.setDecimals(2)
        self.buy_percent.setValue(10)
        self.buy_percent.setSuffix(' % / 1회 매수')
        self.order_popups = QCheckBox('매수·매도 주문 팝업 알림')
        self.order_popups.setChecked(True)
        grid.addWidget(self.percent_sizing, 4, 0, 1, 2)
        grid.addWidget(self.buy_percent, 4, 2)
        grid.addWidget(self.order_popups, 4, 3, 1, 2)
        grid.addWidget(label('시장별 현금 + 보유 평가금액 기준 · 국내 음수 예수금은 검증된 D+2 기준 · 가용액/상한 이내 정수 주\n'
                             '보유종목은 별도 매도 순회: 신호 목표가격 우선, 없으면 평균매입가 +1% / -0.8% · 일반 오류는 해당 종목만 보류', 'muted', wrap=True), 5, 0, 1, 5)
        self.additional_sources = SourceList()
        grid.addWidget(self.additional_sources, 6, 0, 1, 5)
        self.source_status = label('추가 연결 없음', 'muted', wrap=True)
        grid.addWidget(self.source_status, 8, 0, 1, 5)
        self.tabs.addTab(self.external_panel, "외부 신호 연결")

    def configure_external(self):
        self.engine.disarm()
        self._apply_execution_preferences()
        if selected_mode(self.service) is TradingMode.REAL and (
            self.random_demo.isChecked() or self.external_source.text().strip() == 'random-demo'
        ):
            raise ValueError('실전에서는 내장 랜덤 모의 신호기를 사용할 수 없습니다.')
        enabled = self.external_mode.isChecked()
        policy = ExternalPolicy(self.external_source.text().strip(), self.external_quantity.value(),
                                Decimal(str(self.external_krw.value())), Decimal(str(self.external_usd.value())),
                                allow_market=self.random_demo.isChecked())
        if self.random_demo.isChecked() and (not enabled or policy.source_id != "random-demo"):
            raise ValueError("테스트 신호기는 외부 모드와 source_id=random-demo를 사용하세요.")
        inbox, outbox = Path(self.signal_path.text().strip()), Path(self.chart_path.text().strip())
        if not self.signal_path.text().strip() or not self.chart_path.text().strip() or inbox.resolve() == outbox.resolve():
            raise ValueError("입력/출력 JSON 파일은 서로 다른 경로여야 합니다.")
        if inbox.suffix.lower() != ".json" or outbox.suffix.lower() != ".json":
            raise ValueError("입력/출력은 .json 파일을 사용하세요.")
        self.engine.external_policy = policy
        self.engine.external_only = enabled
        self.engine.external_reader = SignalFileReader(self.store, inbox, policy, self.engine.clock) if enabled else None
        self.test_producer = RandomDemoSignals(self.service, self.store, policy, inbox, clock=lambda: self.engine.clock(),
                                              quantity=1 if self.percent_sizing.isChecked() else self.external_quantity.value(), us_order_type=self.random_us.currentData()) if self.random_demo.isChecked() else None
        sources = [(policy, self.engine.external_reader)] if enabled else []
        seen_sources, seen_paths = {policy.source_id}, {inbox.resolve(), outbox.resolve()}
        for source, path in self.additional_sources.sources():
            path = Path(path).expanduser().resolve()
            if source in seen_sources or path in seen_paths or path.suffix.lower() != '.json' or path == self.store.path:
                raise ValueError('신호기 출처/입력은 중복할 수 없고, 출력·DB와 다른 .json 경로여야 합니다.')
            extra = ExternalPolicy(source, policy.max_quantity, policy.max_krw, policy.max_usd)
            seen_sources.add(source)
            seen_paths.add(path)
            if enabled:
                sources.append((extra, SignalFileReader(self.store, path, extra, self.engine.clock)))
        if hasattr(self.engine, 'configure_external_sources'):
            self.engine.configure_external_sources(sources)
        self._applied_external_config = self._external_config_key()
        self._last_export_at, self._last_update_at, self._last_export_error, self._inspection = None, None, "", None
        return inbox, outbox, policy

    def read_signals(self):
        if self.worker or self.monitoring:
            return
        try:
            inbox, _, policy = self.configure_external()
            reader = self.engine.external_reader or SignalFileReader(self.store, inbox, policy, self.engine.clock)
            def read():
                reader()
                return {}
            self._run(read, done="신호 파일 읽기 완료 · 자동주문 OFF · 접수 결과는 외부 신호 연결 탭을 확인하세요.")
        except Exception as exc:
            self.message.setText(f"신호 읽기 실패: {exc}")

    def add_top100(self):
        if self.worker or self.monitoring:
            return
        days = self.days_input.value()
        def collect():
            from dockdack.market_schedule import ranking_allowed
            for market in Market:
                if ranking_allowed(market, self.engine.clock()):
                    ranks = self.service.top_volume(market, 100)
                    if not ranking_allowed(market, self.engine.clock()):
                        raise InterruptedError("장이 종료되어 조회한 순위를 적용하지 않습니다.")
                    protected = self.service.protected_symbols(market)
                    if not ranking_allowed(market, self.engine.clock()):
                        raise InterruptedError("장이 종료되어 조회한 순위를 적용하지 않습니다.")
                    self.store.replace_ranked(market, ranks, protected, days=days, separate_holdings=True)
            return {}
        self._run(collect, done="선정 가능한 시장만 거래량 TOP100 갱신 · 장외/휴장 시장은 조회하지 않았습니다.")

    def export_json(self):
        if self.worker or self.monitoring:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "차트 JSON 내보내기", self.chart_path.text(), "JSON (*.json)")
        if not filename:
            return
        if Path(filename).resolve() == Path(self.signal_path.text()).resolve():
            self.message.setText("신호 입력 파일에 차트를 덮어쓸 수 없습니다.")
            return
        self.chart_path.setText(filename)
        def save():
            export_charts(self.store, filename, now=self.engine.clock(), errors=dict(self.errors))
            self._last_export_at = self.engine.clock()
            self._last_export_error = ""
            return {}
        self._run(save, done=f"차트 내보내기 완료: {filename} · JSON의 status/complete/quote_stale을 확인하세요.")

    def trigger_changed(self):
        uses_price = self.trigger.currentData() in {"price_ge", "price_le"}
        self.threshold.setEnabled(uses_price)
        self.period.setEnabled(not uses_price)

    @property
    def watch_market(self):
        return (Market.DOMESTIC, Market.US)[self.watch_market_tabs.currentIndex()]

    @property
    def watch_table(self):
        """Compatibility accessor for the selected market's table, not a mixed list."""
        return self.watch_tables[self.watch_market]

    def selected_item(self):
        row = self.watch_table.currentRow()
        cell = self.watch_table.item(row, 0) if row >= 0 else None
        key = cell.data(Qt.ItemDataRole.UserRole) if cell else None
        item = self._items_by_id.get(key)
        return item if item and item.instrument.market is self.watch_market else None

    def focus_watch_item(self, watch_id):
        item = self._items_by_id.get(watch_id)
        if item is None:
            return
        view = self.watch_tables[item.instrument.market]
        self.watch_market_tabs.setCurrentWidget(view)
        for row in range(view.rowCount()):
            if view.item(row, 0).data(Qt.ItemDataRole.UserRole) == watch_id:
                view.selectRow(row)
                view.scrollToItem(view.item(row, 0))
                break
        self.select_item()

    def selected_rule(self):
        row = self.rules_table.currentRow()
        cell = self.rules_table.item(row, 0) if row >= 0 else None
        key = cell.data(Qt.ItemDataRole.UserRole) if cell else None
        return next((rule for rule in self.store.rules() if rule.id == key), None)

    @staticmethod
    def set_rows(widget, rows, ids=None):
        if widget.rowCount() != len(rows):
            widget.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                cell = widget.item(row, column)
                if cell is None:
                    cell = QTableWidgetItem()
                    widget.setItem(row, column, cell)
                text = str(value)
                if cell.text() != text:
                    cell.setText(text)
                    cell.setToolTip(text)
                if column == 0 and ids and cell.data(Qt.ItemDataRole.UserRole) != ids[row]:
                    cell.setData(Qt.ItemDataRole.UserRole, ids[row])

    def _watch_values(self, item):
        market = item.instrument.market
        snapshot = self.snapshots.get(item.id)
        status = self.errors.get(item.id) or (
            ("조회 " if item.id in self.fresh_ids else "저장값 ") + snapshot.fetched_at.astimezone().strftime("%m/%d %H:%M:%S")
            if snapshot else "미조회")
        price = f"{number(snapshot.quote.price, 0 if market is Market.DOMESTIC else 4)} {item.instrument.currency}" if snapshot else "—"
        return (f"{item.name or item.instrument.symbol}\n{item.instrument.symbol}", price, item.days, status)

    def _update_watch_row(self, key):
        """One quote changes one row, never hundreds of rows or hidden logs."""
        item = self._items_by_id.get(key)
        row = self._watch_rows.get(key)
        if item is None or row is None:
            # A ranking worker may have just added this symbol. Topology is
            # refreshed at the sweep boundary; state is already kept above.
            return
        view = self.watch_tables[item.instrument.market]
        for column, value in enumerate(self._watch_values(item)):
            cell = view.item(row, column)
            text = str(value)
            if cell.text() != text:
                cell.setText(text)
                cell.setToolTip(text)
        if self.selected_item() is item:
            self.select_item()

    def reload_tables(self):
        items = self.store.items()
        self._items_by_id = {item.id: item for item in items}
        self._watch_rows = {}
        for index, (market, view) in enumerate(self.watch_tables.items()):
            selected = view.item(view.currentRow(), 0)
            selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else None
            scrollbar = view.verticalScrollBar()
            scroll = scrollbar.value()
            market_items = [item for item in items if item.instrument.market is market]
            rows = [self._watch_values(item) for item in market_items]
            self._watch_rows.update({item.id: row for row, item in enumerate(market_items)})
            view.blockSignals(True)
            try:
                self.set_rows(view, rows, [item.id for item in market_items])
                if market_items:
                    selected_index = next((i for i, item in enumerate(market_items) if item.id == selected_id), 0)
                    view.selectRow(selected_index)
                scrollbar.setValue(scroll)
            finally:
                view.blockSignals(False)
            self.watch_market_tabs.setTabText(index, f"{'한국 · KRW' if market is Market.DOMESTIC else '미국 · USD'} ({len(market_items)})")
        rules = self.store.rules(limit=500)
        self.rules_table.setToolTip('최근 규칙 최대 500개 표시 · 전체 주문 기록은 실제 주문·체결 및 매매일지에서 확인하세요.')
        item_map = {i.id: i for i in items}
        self.set_rows(self.rules_table, [(item_map[r.watch_id].instrument.symbol, r.description,
                      "매수" if r.side.value == "buy" else "매도", r.quantity,
                      f"{r.max_notional:,} {item_map[r.watch_id].instrument.currency}", STATUS_LABELS[r.status]) for r in rules],
                      [r.id for r in rules])
        self._reload_activity()
        self.select_item()
        self.update_controls()

    def _workspace_changed(self, *_):
        self._reload_activity(visible_force=True)
        self._update_connection()
        if self.workspace_tabs.currentWidget() is self.watch_page:
            self.select_item()

    def _reload_activity(self, *, force=False, visible_force=False):
        page = self.workspace_tabs.currentWidget()
        if not force and page not in (self.operations_panel, self.order_history_panel, self.trade_journal_panel):
            return
        if self._close_when_idle:
            return
        if self._activity_worker is not None:
            self._activity_pending |= force or visible_force
            return
        now = monotonic()
        if not (force or visible_force) and now - self._last_log_reload < 2:
            return
        self._last_log_reload = now
        store, collector = self.store, self._ledger_collector
        categories = tuple(self.operations_panel.logs) if force else (
            (self.operations_panel.tabs.currentWidget().category,) if page is self.operations_panel else ())
        if force or page is self.order_history_panel:
            categories += ('order',)
        heads = {key: view._head for key, view in self.operations_panel.logs.items()}
        heads['order'] = self.order_history_panel.audit._head
        ledger_needed = force or page in (self.order_history_panel, self.trade_journal_panel)
        def collect():
            logs = collect_event_logs(store, categories, previous_heads=heads)
            snapshot = collector.collect() if ledger_needed else None
            return store, logs, snapshot
        worker = Worker(collect)
        self._activity_worker = worker
        worker.signals.completed.connect(self._activity_completed)
        self.activity_pool.start(worker)

    @Slot(object, object)
    def _activity_completed(self, result, error):
        self._activity_worker = None
        if error is not None:
            self._last_log_error = f'기록 DB 조회 실패 · 이전 화면 유지: {error}'
        elif result[0] is self.store:
            self._last_log_error = ''
            _, logs, snapshot = result
            self.operations_panel.apply_logs(logs)
            self.order_history_panel.apply_logs(logs)
            if snapshot is not None:
                self.order_history_panel.apply_snapshot(snapshot)
                self.trade_journal_panel.apply_snapshot(snapshot)
        pending, self._activity_pending = self._activity_pending, False
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
        elif self._pending_environment:
            self._finish_environment_switch()
        elif pending:
            self._reload_activity(force=True)

    def api_rate_status(self):
        """Read only existing clients; opening the dashboard never authenticates."""
        states = {}
        for broker in getattr(self.service, "brokers", {}).copy().values():
            for attribute in ("_domestic_http", "_us_http"):
                client = getattr(broker, attribute, None)
                if client is not None and callable(getattr(client, "rate_status", None)):
                    state = client.rate_status()
                    states[state["scope_id"]] = state
        return list(states.values())

    def _update_health(self):
        self._schedule_wakeup()  # Calendar-only badges also refresh while orders/monitoring are OFF.
        now = utc_now()
        age = monotonic() - self._last_activity
        if self.worker:
            work = f"조회 응답 지연 확인 필요 · {int(age)}초간 진행 없음" if age > 120 else f"작업 중 · {self._progress_text}"
            if self._worker_started:
                seconds = max(0, int((now - self._worker_started).total_seconds()))
                work += f" · 경과 {seconds // 60}분 {seconds % 60:02d}초"
        elif self.monitoring:
            work = f"다음 순회 대기 · {max(0, self.timer.remainingTime() // 1000)}초"
        else:
            work = "감시 중지"
        failures = len(self.errors)
        account_errors = sum(bool(s.error) for s in self._portfolio_payload.values())
        text = (f"앱 응답 {now.astimezone():%H:%M:%S}  ·  {work}  ·  "
                f"시세 오류 {failures} / 잔고 오류 {account_errors}건")
        rates = self.api_rate_status()
        if rates:
            remaining = max(s["wait_remaining_seconds"] for s in rates)
            cooldown = max(s["cooldown_remaining_seconds"] for s in rates)
            interval = max(s["effective_interval_seconds"] for s in rates)
            retries = sum(s["retry_count"] for s in rates)
            pacing = f"제한 감지 후 {cooldown:.1f}초 대기" if cooldown > 0 else f"다음 호출까지 {remaining:.1f}초 대기" if remaining > 0 else "요청 처리 중" if any(s["in_flight"] for s in rates) else "호출 대기열 비어 있음"
            text += f"\nAPI 속도 조절 · {pacing} · 안전 간격 {interval:.2f}초 + 응답시간 · 조회 재시도 {retries}회"
        if self.worker:
            suffix = (f'호출 제한 대기 {cooldown:.1f}초' if cooldown > 0 else
                      f'호출 간격 대기 {remaining:.1f}초' if remaining > 0 else '') if rates else ''
            self.sweep_progress.refresh_wait(suffix)
        elif self.monitoring and self._all_markets_closed():
            self.sweep_progress.set_activity('장외 대기 · 한국·미국 정규장 전에는 시세·매수·매도 감시를 쉬고 있습니다')
        if self._last_log_error:
            text += f" · {self._last_log_error}"
        self.health_label.setText(text)
        warning = failures or account_errors or self._last_worker_error or self._last_log_error or (self.worker and age > 120) or any(s["cooldown_remaining_seconds"] > 0 for s in rates)
        if bool(warning) != self._health_warning:
            self._health_warning = bool(warning)
            self.health_label.setStyleSheet("color: #ffda91;" if warning else "color: #95a4bb;")
        self.operations_panel.runtime.setText(
            f"{text}\n마지막 작업 응답 {local_time(self._last_progress_at)}  ·  "
            f"마지막 작업 종료 {local_time(self._last_completed_at)}\n"
            f"자동주문 {'ON · 주문 허용' if self.engine.orders_enabled else 'OFF · 주문 차단'}"
            + (f" · 작업 오류: {self._last_worker_error}" if self._last_worker_error else "")
        )
        self.portfolio_panel.apply(self._portfolio_payload, now=now)
        self._reload_activity()
        self._update_connection()

    def _refresh_portfolio_worker(self, progress, *, force=False):
        """Only called by the existing single worker, never a parallel API thread."""
        if self.engine._stop.is_set():
            return
        requested = self._portfolio_requested.is_set()
        self._portfolio_requested.clear()
        previous = self.portfolio.snapshot()
        payload = self.portfolio.refresh_due(force=force or requested, stopped=self.engine._stop.is_set)
        if (payload != previous or requested or force) and hasattr(self.engine, 'holding_exit_targets'):
            progress(('exit_targets', self._portfolio_exit_targets_worker(payload)))
        if payload == previous and not requested and not force:
            return
        for market, state in payload.items():
            if state.last_attempt != previous[market].last_attempt:
                market_name = "한국" if market is Market.DOMESTIC else "미국"
                text = f"{market_name} 잔고 조회 실패 · {state.error}" if state.error else f"{market_name} 잔고 조회 완료 · 보유 {len(state.positions)}종목"
                self.store.event("SYSTEM", text, category="system")
        progress(("portfolio", payload))

    def _portfolio_exit_targets_worker(self, payload):
        """An unsupported holding must not interrupt other markets' monitoring.

        Keep the row visible, but never fabricate a tradable venue or a target
        for it. Storage/integrity failures still escape to the fail-closed path.
        """
        targets = {}
        for state in payload.values():
            for position in state.positions:
                key = f'{position.market.value}:{position.exchange}:{position.symbol}'
                try:
                    targets[key] = self.engine.holding_exit_targets(position)
                except ValueError as exc:
                    targets[key] = {'take_profit_price': None, 'stop_loss_price': None,
                                    'source': '자동매도 보류', 'error': str(exc), 'watch_id': key}
                    self.engine._message('holding-target-error:' + key, key,
                        f'해당 보유종목 자동매도 보류 · 종목/거래소 확인 필요 ({position.exchange}): {exc}'
                        ' · 다른 종목 감시는 계속', category='monitor')
        return targets

    @Slot()
    def refresh_portfolio(self):
        self._portfolio_requested.set()
        if self.worker:
            self.message.setText("잔고 조회 예약 · 현재 API 요청 후 순서대로 처리합니다. 시장별 최소 60초 간격이며 자동주문 상태는 바뀌지 않습니다.")
            return
        # Read-only account refresh does not resume monitoring or change the order gate.
        def refresh(progress):
            previous = self.portfolio.snapshot()
            self._portfolio_requested.clear()
            payload = self.portfolio.refresh_due(force=True, stopped=self._account_stop.is_set)
            progress(('exit_targets', self._portfolio_exit_targets_worker(payload)))
            for market, state in payload.items():
                if state.last_attempt != previous[market].last_attempt:
                    self.store.event("SYSTEM", f"{'한국' if market is Market.DOMESTIC else '미국'} 잔고 "
                                     + (f"조회 실패 · {state.error}" if state.error else f"조회 완료 · 보유 {len(state.positions)}종목"), category="system")
            progress(("portfolio", payload))
            return {}
        self._run(refresh, streaming=True, done="잔고 조회 처리 완료 · 결과/오류와 기준 시각은 보유종목 탭에서 확인하세요. 재조회는 시장별 최소 60초 간격입니다.")

    def select_item(self, *_):
        item = self.selected_item()
        market_name = "한국" if self.watch_market is Market.DOMESTIC else "미국"
        count = self.watch_table.rowCount()
        self.watch_market_hint.setText(
            f"{market_name} 관심종목 {count}개 · 탭 전환과 무관하게 양쪽 시장 감시" if count else
            f"{market_name} 관심종목이 없습니다. 종목을 추가하거나 TOP100 재선정을 이용하세요.")
        if (item.id if item else None) != self._rule_watch_id:
            self._rule_watch_id = item.id if item else None
            # Prices and caps must never carry over silently between instruments/currencies.
            self.threshold.setValue(0)
            self.max_notional.setValue(0)
            if item:
                self.days_input.setValue(item.days)
        self.rule_label.setText(f"{item.instrument.symbol} · {item.instrument.currency} · 1회성 규칙 등록" if item else "관심종목을 먼저 선택하세요")
        if item:
            self.max_notional.setSuffix(f" {item.instrument.currency}")
            self.threshold.setSuffix(f" {item.instrument.currency}")
        else:
            self.max_notional.setSuffix("")
            self.threshold.setSuffix("")
        snapshot = self.snapshots.get(item.id) if item else None
        signature = (item.id if item else None, item.days if item else None, id(snapshot),
                     item.id in self.fresh_ids if item else False, item.id in self.errors if item else False,
                     self.watch_market)
        if signature == self._chart_signature:
            return
        self._chart_signature = signature
        if not snapshot:
            self.chart.set_history((), "")
            self.bar_table.setRowCount(0)
            self.chart_title.setText("종목을 조회해 주세요" if item else f"{market_name} 관심종목 없음")
            return
        bars = snapshot.history.bars[-item.days:]
        warning = " · 이전 조회값" if item.id not in self.fresh_ids or item.id in self.errors else ""
        self.chart_title.setText(f"{item.instrument.symbol} · {len(bars)}/{item.days} 거래일 · 수정주가{warning}")
        self.chart.set_history(bars, item.instrument.currency)
        self.set_rows(self.bar_table, [(b.day, b.open, b.high, b.low, b.close, b.volume) for b in reversed(bars)])

    def _refresh_executions_worker(self, progress, *, force=False):
        """Shares the existing broker worker and HTTP pacing; no parallel API calls."""
        requested = self._fill_refresh_requested.is_set()
        self._fill_refresh_requested.clear()
        previous = self.fill_recovery.status()
        if requested or force:
            progress(("executions", {**previous, "state": "running", "message": "증권사 체결가 보완 조회 중 · 주문하지 않습니다."}))
        status = self.fill_recovery.refresh_due(force=force or requested, stopped=self._account_stop.is_set)
        if status != previous or requested or force:
            progress(("executions", status))

    def refresh_executions(self):
        self._fill_refresh_requested.set()
        if self.worker:
            self.message.setText("체결가 조회 예약 · 현재 API 요청 후 순서대로 확인합니다. 주문 상태는 바뀌지 않습니다. 재조회 최소 60초.")
            return
        def refresh(progress):
            self._refresh_executions_worker(progress, force=True)
            return {}
        self._run(refresh, streaming=True, done="체결가 보완 조회 완료 · 실제 주문·체결 탭의 가격/실현손익과 미확인 사유를 확인하세요.")

    def _run(self, operation, *, streaming=False, done=None, focus_result=False, job_kind="task"):
        if self.worker:
            return False
        self._done_message = done
        self._focus_result = focus_result
        self._account_stop.clear()
        if streaming:
            worker = Worker(lambda: operation(worker.signals.progress.emit))
        else:
            worker = Worker(operation)
        self.worker = worker
        self._worker_kind = job_kind
        self._worker_started = utc_now()
        self._last_activity = monotonic()
        self._progress_text = "API 요청 대기"
        self.sweep_progress.set_activity('계좌·체결 조회 준비 · 이후 관심종목 → 보유종목 순회'
            if job_kind == 'quotes' else '계좌·자료 조회 처리 중 · 체결 표시 아님', waiting=True)
        self._last_worker_error = ""
        self.worker.signals.progress.connect(self._progress)
        self.worker.signals.completed.connect(self._completed)
        self.update_controls()
        self.message.setText("API 조회 중… 호출 간격을 지키며 순서대로 처리합니다. 중지 버튼은 사용할 수 있습니다.")
        self.pool.start(self.worker)
        return True

    @Slot(object)
    def _progress(self, data):
        self._last_progress_at = utc_now()
        self._last_activity = monotonic()
        if len(data) == 2 and data[0] == 'market_status':
            self._apply_market_status(data[1])
            return
        if len(data) == 2 and data[0] == 'watch_progress':
            value = data[1]
            name = '한국' if value['market'] is Market.DOMESTIC else '미국'
            self._progress_text = f"{name} 관심종목 시세·차트 조회 {value['completed']}/{value['total']}종목 · {value['symbol']}"
            self.sweep_progress.set_activity(self._progress_text, waiting=True)
            return
        if len(data) == 2 and data[0] == 'holdings_progress':
            self._holding_progress(data[1])
            return
        if len(data) == 2 and data[0] == 'notifications':
            if self.order_popups.isChecked():
                self.order_toast.notify(data[1])
            return
        if len(data) == 2 and data[0] == 'exit_targets':
            self.portfolio_panel.set_exit_targets(data[1])
            return
        if len(data) == 2 and data[0] == 'phase':
            self._progress_text = str(data[1])
            self.message.setText(str(data[1]))
            self.sweep_progress.set_activity(str(data[1]) + ' · 현재가로 상방/하방 확인', waiting=True)
            return
        if len(data) == 2 and data[0] == 'holding_quote':
            self.portfolio_panel.apply_holding_quote(data[1])
            self._progress_text = f"보유종목 매도 감시 · {data[1]['instrument'].symbol}"
            return
        if len(data) == 2 and data[0] == "executions":
            self._fill_recovery_status = data[1]
            self.order_history_panel.set_recovery_status(data[1])
            self._reload_activity(visible_force=True)
            return
        if len(data) == 2 and data[0] == "watchlist":
            # Structural refresh is rare (ranking changed), not per quote.
            self.reload_tables()
            return
        if len(data) == 2 and data[0] == "portfolio":
            self._portfolio_payload = data[1]
            self.portfolio_panel.apply(self._portfolio_payload)
            self._progress_text = "계좌 잔고 조회 처리"
            self._update_health()
            return
        key, value, count, total = data
        if isinstance(value, Exception):
            self.errors[key] = str(value)
            self.fresh_ids.discard(key)
        else:
            self.snapshots[key] = value
            self.errors.pop(key, None)
            self.fresh_ids.add(key)
        self.message.setText(f"조회 {count}/{total} · {key.split(':')[-1]} · 외부 모드에서는 새 신호를 우선 확인합니다.")
        self._progress_text = f"시세·차트 {count}/{total} · {key.split(':')[-1]}"
        self.sweep_progress.set_activity('관심종목 시세·차트 조회 %v/%m종목 · %p%', completed=count, total=total)
        self.sweep_progress.setToolTip(f"이번 순회 {count}/{total} · 조회 시도 진행률이며 매수·매도·체결 또는 오류 없음의 표시가 아닙니다.")
        self._update_watch_row(key)

    def _all_markets_closed(self):
        return all(self._market_display_open.get(market) is False for market in Market)

    def _apply_market_status(self, statuses):
        # Badges can complete independently of a broker sweep. They must never
        # replace the sweep's activation-validation market snapshot.
        self._market_display_open = {market: value['is_open'] for market, value in statuses.items()}
        self._market_summary = ' · '.join(value['text'] for value in statuses.values())
        for market, value in statuses.items():
            badge = self.market_labels[market]
            badge.setText(value['text'])
            badge.setToolTip(value['detail'] + f"\n확인 시각 {value['checked_at'].astimezone():%H:%M:%S}")
            tone = 'active' if value['is_open'] else 'error' if value['is_open'] is None else ''
            if badge.property('tone') != tone:
                badge.setProperty('tone', tone)
                badge.style().unpolish(badge)
                badge.style().polish(badge)

    def _holding_progress(self, value):
        phase = value['phase']
        market = value.get('market')
        title = '한국' if market is Market.DOMESTIC else '미국' if market is Market.US else ''
        completed, total = value.get('completed', 0), value.get('total', 0)
        if phase == 'account':
            text, waiting = f'{title} 보유종목 목록 조회 중', True
        elif phase in {'checking', 'checked'}:
            text = f"{title} 보유종목 매도 조건 확인 {completed}/{total}종목 · {value.get('symbol', '')}"
            if value.get('error'):
                text += ' · 해당 종목 확인 실패'
            waiting = phase == 'checking'
        elif phase == 'market_closed':
            text, waiting = f'{title} 장외 · 보유종목 매도 감시 대기', False
        elif phase == 'market_error':
            text, waiting = f'{title} 보유종목 조회 실패 · 다음 순회 재확인', False
        elif phase == 'market_complete':
            text = f'{title} 보유종목 매도 조건 점검 순회 완료 {completed}/{total}종목' if total else f'{title} 보유종목 없음'
            waiting = False
        else:
            text = ('장외 대기 · 한국·미국 정규장 전에는 주문하지 않습니다' if self._all_markets_closed() else
                    f'보유종목 매도 조건 점검 순회 완료 · {completed}/{total}종목 · 체결 표시 아님')
            waiting = False
        self._progress_text = text
        self.sweep_progress.set_activity(text, completed=completed, total=total, waiting=waiting)
        self.sweep_progress.setToolTip(value.get('error') or '보유종목의 현재가와 상방·하방 목표가를 점검하는 진행 상태입니다. 주문·체결 성공을 의미하지 않습니다.')

    @Slot(object, object)
    def _completed(self, results, error):
        job_kind = self._worker_kind
        self._worker_kind = ""
        self.worker = None
        self._last_completed_at = utc_now()
        self._last_activity = monotonic()
        if job_kind == 'quotes':
            state = '중단' if error is not None or self.engine._stop.is_set() else '순회 완료'
            summary = ('장외 대기 · 한국·미국 정규장 전에는 시세·매수·매도 감시를 쉬고 있습니다'
                       if state == '순회 완료' and self._all_markets_closed() else
                       f'관심종목·보유종목 {state} · 체결 표시 아님')
            self.sweep_progress.set_activity(summary, completed=1 if state == '순회 완료' else 0, total=1)
        else:
            self.sweep_progress.set_activity('계좌·자료 조회 오류' if error is not None else '계좌·자료 조회 완료 · 체결 표시 아님', completed=int(error is None), total=1)
        if error is not None:
            # Failures before/after engine.poll need the same classification as
            # checkpoint failures inside it. Never retain ON for a broken ledger.
            if not self.engine.isolated_symbol_errors or not _transient_poll_failure(error):
                self.engine.disarm()
            self._last_worker_error = str(error)
            state = '해당 작업 보류 · 다음 순회 재시도' if self.engine.orders_enabled else '자동주문 OFF'
            self.store.event("SYSTEM", f"작업 실패 · {state}: {error}", category="system")
            self.message.setText(f"처리 실패 · {state}: {error}")
        else:
            for key, value in (results or {}).items():
                if isinstance(value, Exception):
                    self.errors[key] = str(value)
                    self.fresh_ids.discard(key)
                else:
                    self.snapshots[key] = value
                    self.errors.pop(key, None)
                    self.fresh_ids.add(key)
            self.message.setText(self._done_message or "조회 완료 · 규칙 상태와 기록을 확인하세요. 당일 일봉은 장중에 변하며, 시세는 API 제공값입니다.")
            if not self._done_message:
                success = sum(not isinstance(v, Exception) for v in (results or {}).values())
                failed = sum(isinstance(v, Exception) for v in (results or {}).values())
                self.store.event("SYSTEM", f"감시 순회 {'중단' if self.engine._stop.is_set() else '종료'} · 성공 {success} / 오류 {failed}종목", category="system")
        self.reload_tables()
        if getattr(self, "_focus_result", False) and error is None:
            for key, value in (results or {}).items():
                if not isinstance(value, Exception):
                    self.focus_watch_item(key)
                    break
        self._focus_result = False
        self._reload_activity(force=True)
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
            return
        if self._pending_environment is not None:
            self._finish_environment_switch()
            return
        if self._advance_manual_activation(job_kind, results, error):
            self._update_health()
            return
        if self.monitoring:
            self.timer.start(self.interval.value() * 1000)
        self._update_health()

    def refresh_all(self):
        if not self.worker:
            self.engine.resume_monitoring()
            outbox = Path(self.chart_path.text())
            def publish(item, snapshot):
                if not self.engine.external_only:
                    return
                updates = outbox.with_name(outbox.stem + "_updates")
                path = updates / f"{item.instrument.market.value}_{item.instrument.exchange}_{item.instrument.symbol}.json"
                try:
                    payload = export_charts(self.store, path, now=self.engine.clock(), watch_ids={item.id})
                except Exception as exc:
                    self._last_export_error = f"차트 전달 실패: {exc}"
                    raise
                self._last_update_at, self._last_export_error = self.engine.clock(), ""
                if self.test_producer is not None:
                    self.test_producer.publish(payload)
            def refresh(progress):
                # Calendar construction and session checks stay off the GUI thread.
                checked = None
                def market_status():
                    nonlocal checked
                    now = self.engine.clock()
                    if checked and (now - checked).total_seconds() < 60:
                        return
                    statuses = market_statuses(now)
                    self._market_open = {market: value['is_open'] for market, value in statuses.items()}
                    self._market_summary = ' · '.join(value['text'] for value in statuses.values())
                    progress(('market_status', statuses))
                    checked = now
                market_status()
                self.store.event("SYSTEM", "감시 순회 시작 · 시세/차트 조회와 신호 전달 (주문 상태는 별도)", category="system")
                def checkpoint():
                    market_status()
                    self._refresh_executions_worker(progress)
                    self._refresh_portfolio_worker(progress)
                    self._order_notifications_worker(progress)
                    changed = self.scheduler.tick() if self.monitoring else False
                    if changed:
                        progress(("watchlist", None))
                    return changed
                self._refresh_executions_worker(progress)
                self._refresh_portfolio_worker(progress)
                self._order_notifications_worker(progress)
                results = self.engine.poll(progress=progress, checkpoint=checkpoint, on_snapshot=publish)
                self._order_notifications_worker(progress)
                if self.engine.external_only and not self.engine._stop.is_set():
                    try:
                        export_charts(self.store, outbox, now=self.engine.clock(),
                                      errors={k: str(v) for k, v in results.items() if isinstance(v, Exception)})
                    except Exception as exc:
                        self._last_export_error = f"전체 차트 전달 실패: {exc}"
                        raise
                    self._last_export_at, self._last_export_error = self.engine.clock(), ""
                return results
            self._run(refresh, streaming=True, job_kind="quotes")

    def start_monitoring(self):
        if self._pending_environment or self._confirming_environment:
            return
        if self.worker or (not self.store.items() and not self.hourly_ranking.isChecked()
                           and not self.engine.enable_holdings_exits):
            return
        self.engine.disarm()
        try:
            self.configure_external()
        except Exception as exc:
            self.message.setText(str(exc))
            return
        self.monitoring = True
        self.store.event("SYSTEM", "감시 시작 · 자동주문 OFF · 시세/차트 및 신호 수신만 진행", category="system")
        if self.hourly_ranking.isChecked():
            self.scheduler.start()
            self.schedule_timer.start()
        self.refresh_all()

    def _schedule_wakeup(self):
        if self._close_when_idle or self._pending_environment or self._schedule_probe is not None:
            return
        now = self.engine.clock()
        minute = int(now.timestamp()) // 60
        check_schedule = self.monitoring and not self.worker and minute != self._schedule_minute
        if not check_schedule and minute == self._market_status_minute:
            return
        if check_schedule:
            self._schedule_minute = minute
        self._market_status_minute = minute
        # Calendar construction and SQLite must never block the UI timer.
        scheduler = self.scheduler
        worker = Worker(lambda: (scheduler, scheduler.due() if check_schedule else False, market_statuses(now)))
        self._schedule_probe = worker
        worker.signals.completed.connect(self._schedule_checked)
        self.activity_pool.start(worker)

    @Slot(object, object)
    def _schedule_checked(self, result, error):
        self._schedule_probe = None
        if error is None and result[0] is self.scheduler:
            self._apply_market_status(result[2])
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
        elif self._pending_environment:
            self._finish_environment_switch()
        elif error is None and result[0] is self.scheduler and result[1] and self.monitoring and not self.worker:
            self.timer.stop()
            self.refresh_all()
        elif self.monitoring and not self.worker:
            # Monitoring may have started while the initial calendar-only
            # probe was running. Do not lose that first scheduling check.
            self._schedule_wakeup()

    def stop_monitoring(self):
        was_running = self.monitoring or self.pending_auto_arm or self.engine.orders_enabled
        self._order_request_revision += 1
        self._manual_arm_pending = False
        self.pending_auto_arm = False
        self.monitoring = False
        self.timer.stop()
        self.schedule_timer.stop()
        self.scheduler.stop()
        self.engine.stop()
        self._account_stop.set()
        if was_running:
            self.store.event("SYSTEM", "감시·주문 중지 요청 · 기존 접수 주문 취소 아님", category="system")
        self.update_controls()
        self.message.setText("중지 요청됨 · 다음 전송을 차단합니다. 이미 전송된 주문은 취소되지 않으므로 내역을 확인하세요.")

    def confirm_automation(self):
        mode = selected_mode(self.service)
        name = environment_name(mode)
        if mode is TradingMode.REAL:
            if self.random_demo.isChecked() or (self.external_mode.isChecked() and self.external_source.text().strip() == 'random-demo'):
                raise ValueError('실전에서는 내장 랜덤 모의 신호기를 사용할 수 없습니다.')
            for item in self.store.items():
                self.service.ensure_order_permission(item.instrument)
        # Preview the locked GUI settings without mutating a running worker's reader/policy.
        if self.external_mode.isChecked():
            policy = ExternalPolicy(self.external_source.text().strip(), self.external_quantity.value(),
                                    Decimal(str(self.external_krw.value())), Decimal(str(self.external_usd.value())),
                                    allow_market=self.random_demo.isChecked())
            return QMessageBox.question(self, f"외부 신호 {name} 자동주문 확인",
                "확인하면 전체 조회 완료 후 자동주문이 ON 됩니다. 다시 켜기를 누를 필요가 없습니다.\n"
                "조회 실패 시 OFF를 유지하며, 기다리는 동안 OFF로 예약을 취소할 수 있습니다.\n\n"
                f"{policy.source_id}의 새 buy/sell 신호를 개별 확인 없이 {name} 주문할까요?\n"
                f"추가 연결 신호기: {len(self.additional_sources.sources())}개\n"
                f"입력: {self.signal_path.text()}\n"
                + (f"1회 매수: 시장별 계좌 평가금액(예수금 + 보유 평가액)의 {self.buy_percent.value():g}% · 정수 주식 수 내림\n국내 음수 예수금은 검증된 D+2 추정예수금을 사용하며 해당 현금·주문가능금액 이내로 제한합니다.\n" if self.percent_sizing.isChecked()
                   else f"주문당 최대 {policy.max_quantity}주\n") +
                f"주문당 금액 상한: {policy.max_krw:,} KRW / {policy.max_usd:,} USD\n"
                f"국내 시장가 허용: {'예 (금액 상한은 현재가 추정치)' if policy.allow_market else '아니오'}\n"
                f"미국: {self.random_us.currentText() if self.random_demo.isChecked() else '현재가 지정가만 허용'}\n"
                + ("내장 모의 신호기: 매수 확률 10% · 평균 매입가 대비 +1% 익절 / -0.8% 손절\n" if self.random_demo.isChecked() else "") +
                "0인 시장은 차단됩니다. 수동 트리거는 실행하지 않습니다.\n"
                "보유분은 목표가 또는 평균매입가 +1% / −0.8% 조건을 별도로 점검합니다.\n"
                "미국은 증권사 거절이 확정된 경우만 조건을 재확인해 최대 총 3회 시도합니다. 접수·미체결·불명확 주문이나 로컬 차단은 재전송하지 않습니다.\n"
                "상한은 주문당 제한이며 하루 누적 한도는 아닙니다."
                + ("\n실제 자금으로 반복 주문되며 손실이 발생할 수 있습니다." if mode is TradingMode.REAL else ""),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
        rules = [r for r in self.store.rules() if r.status == "ready" and r.kind is not TriggerKind.EXTERNAL]
        if not rules:
            self.message.setText("대기 중인 규칙을 먼저 등록하세요.")
            return False
        items = {item.id: item for item in self.store.items()}
        summary = "\n".join(f"{r.watch_id.split(':')[-1]} · {r.description} · {'매수' if r.side.value == 'buy' else '매도'} {r.quantity}주 · 상한 {r.max_notional} {items[r.watch_id].instrument.currency}"
                            for r in rules)
        return QMessageBox.question(self, f"{name} 자동주문 활성화 확인",
            "확인하면 전체 조회 성공 후 자동주문이 ON 됩니다. 기다리는 동안 OFF로 예약을 취소할 수 있습니다.\n"
            f"조건이 맞으면 개별 주문 확인창 없이 {name} 지정가 주문이 전송됩니다.\n"
            "미국 증권사 거절이 확정된 경우 조건 재확인 후 최대 총 3회 시도합니다. 접수·미체결·불명확 주문이나 로컬 차단은 재전송하지 않습니다.\n"
            "거래일 캘린더의 정규장만 허용하며 캘린더 오류 시 차단합니다.\n\n" + summary
            + ("\n실제 자금으로 주문되며 원금 손실이 발생할 수 있습니다." if mode is TradingMode.REAL else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def toggle_orders(self):
        if self.engine.orders_enabled or self.pending_auto_arm:
            self.disable_auto_orders()
        else:
            self.enable_auto_orders()

    def enable_auto_orders(self):
        if self.engine.orders_enabled or self.pending_auto_arm or self._confirming_orders or self._pending_environment or self._confirming_environment:
            return
        if not self.store.items() and not self.hourly_ranking.isChecked() and not self.engine.enable_holdings_exits:
            self.message.setText("관심종목을 먼저 추가하세요. 자동주문은 OFF입니다.")
            return
        revision = self._order_request_revision
        self._confirming_orders = True
        self.update_controls()
        try:
            confirmed = self.confirm_automation()
        except Exception as exc:
            confirmed = False
            self.message.setText(f"자동주문 설정 확인 실패 · OFF 유지: {exc}")
        finally:
            self._confirming_orders = False
            self.update_controls()
        # A stop can arrive while the modal confirmation is running its nested event loop.
        if not confirmed or revision != self._order_request_revision:
            return
        self._manual_arm_pending = True
        self.pending_auto_arm = True
        self._manual_external_error_baseline = self.engine.external_error_count
        try:
            self.store.event("SYSTEM", "사용자 확인 · 전체 조회 성공 후 자동주문 ON 예약 · OFF로 취소 가능", category="system")
        except (OSError, sqlite3.Error) as exc:
            self._cancel_manual_activation(f"예약 기록 저장 실패: {exc}")
            return
        self.message.setText("전체 조회 완료 후 ON 예정 · 다시 켜기를 누를 필요가 없습니다. 오류가 있으면 OFF를 유지합니다.")
        self.update_controls()
        if not self.worker:
            self._begin_manual_warmup()

    def _cancel_manual_activation(self, reason):
        self._order_request_revision += 1
        self._manual_arm_pending = False
        self.pending_auto_arm = False
        self.engine.disarm()
        self.message.setText(f"자동주문 OFF · ON 예약 취소: {reason}")
        try:
            self.store.event("SYSTEM", f"자동주문 ON 예약 취소 · {reason}", category="system")
        except (OSError, sqlite3.Error) as exc:
            self.message.setText(f"자동주문 OFF · ON 예약 취소: {reason} · 기록 저장 실패: {exc}")
        self.update_controls()

    def _begin_manual_warmup(self):
        if not self._manual_arm_pending or not self.pending_auto_arm or self.worker:
            return
        try:
            self.timer.stop()
            if not self.monitoring:
                self.start_monitoring()
            else:
                self.refresh_all()
            if self.worker is None:
                self._cancel_manual_activation("전체 조회를 시작하지 못했습니다. 관심종목과 설정을 확인하세요.")
        except Exception as exc:
            self._cancel_manual_activation(str(exc))

    def activation_failure(self, results, error, *, external_error_baseline):
        """Shared fail-closed check for manual and explicitly authorized launcher warmups."""
        if error is not None:
            return str(error) or type(error).__name__
        active_items = self.store.items()
        if self._market_open:
            active_items = tuple(item for item in active_items if self._market_open.get(item.instrument.market, False))
            if not active_items and self.monitoring and not self.engine._stop.is_set():
                return ''  # Closed markets never require out-of-session quotes to arm a gated engine.
        active = {item.id for item in active_items}
        results = results or {}
        failures = {key: str(value) for key, value in results.items() if key in active and isinstance(value, Exception)}
        missing = active - {key for key, value in results.items() if isinstance(value, MarketSnapshot)}
        markets = {item.instrument.market for item in active_items}
        account_errors = {market.value: self._portfolio_payload[market].error or self._portfolio_payload[market].status()
                          for market in markets if self._portfolio_payload[market].status() not in {"ok", "empty"}}
        external_failure = (self.engine.external_error or
                            ("조회 중 외부 신호 읽기 오류가 발생했습니다." if self.engine.external_error_count != external_error_baseline else "")) if self.engine.external_only else ""
        reason = ("감시가 중지되었습니다." if not self.monitoring or self.engine._stop.is_set() else
                  failures or ("전체 종목 조회가 완료되지 않았습니다." if missing or not active else "") or
                  account_errors or external_failure or (self.scheduler.errors if self.hourly_ranking.isChecked() else {}))
        return str(reason) if reason else ""

    def _advance_manual_activation(self, job_kind, results, error):
        """Consume a user's one-shot request only after a complete, successful sweep."""
        if not self._manual_arm_pending or not self.pending_auto_arm:
            return False
        if error is not None:
            self._cancel_manual_activation(str(error))
            return False
        if not self.monitoring or job_kind != "quotes":
            self._begin_manual_warmup()
            return self.worker is not None
        reason = self.activation_failure(results, error, external_error_baseline=self._manual_external_error_baseline)
        if reason:
            self._cancel_manual_activation(str(reason))
            return False
        # Consume before arming; an error or a late completion never retries this approval.
        self._manual_arm_pending = False
        self.pending_auto_arm = False
        try:
            self.engine.enable_orders("REAL_AUTOTRADE" if selected_mode(self.service) is TradingMode.REAL else "DEMO_AUTOTRADE")
        except Exception as exc:
            self._cancel_manual_activation(str(exc))
            return False
        self.message.setText(f"전체 조회 성공 · 자동주문 ON · 정규장에 유효한 신호와 주문 조건을 충족하면 {environment_name(selected_mode(self.service))} 주문합니다.")
        self.update_controls()
        self.timer.stop()
        self.refresh_all()
        return True

    def disable_auto_orders(self):
        was_enabled = self.engine.orders_enabled or self.pending_auto_arm
        self._order_request_revision += 1
        self._manual_arm_pending = False
        self.pending_auto_arm = False
        self.engine.disarm()
        if was_enabled:
            self.store.event("SYSTEM", "자동주문 OFF · 새 주문 및 ON 예약 차단 · 감시/신호 수신은 별도", category="system")
        self.message.setText("자동주문 OFF · 새 주문과 자동 활성화 예약을 차단했습니다. 전송 중이거나 접수된 주문은 취소되지 않습니다.")
        self.update_controls()

    def add_item(self):
        if self.monitoring or self.worker:
            return
        symbol, exchange, days = self.symbol_input.text(), self.exchange_input.currentData(), self.days_input.value()
        def add():
            instrument = self.service.resolve(symbol, exchange)
            item = WatchItem(instrument, "", days)
            snapshot = self.engine.snapshot(item)
            self.store.save_item(replace(item, name=snapshot.quote.name))
            return {item.id: snapshot}
        self.engine.resume_monitoring()
        self._run(add, focus_result=True)

    def apply_days(self):
        item = self.selected_item()
        if item and not self.monitoring and not self.worker:
            self.store.save_item(replace(item, days=self.days_input.value()))
            self.fresh_ids.discard(item.id)
            self.reload_tables()
            self.message.setText("기간 저장됨 · 전체 조회를 눌러 데이터를 갱신하세요.")

    def remove_item(self):
        item = self.selected_item()
        if item and not self.monitoring and not self.worker:
            if QMessageBox.question(self, "관심종목 제외", f"{item.instrument.symbol}을 제외하고 대기 규칙을 비활성화할까요?\n주문 이력은 보존하며 접수된 주문은 취소되지 않습니다.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                try:
                    self.store.remove_item(item.id)
                    self.reload_tables()
                except Exception as exc:
                    self.message.setText(str(exc))

    def add_rule(self):
        item = self.selected_item()
        if not item or self.monitoring or self.worker:
            return
        try:
            rule = TriggerRule.create(item, self.trigger.currentData(), self.side.currentData(), self.quantity.value(),
                                      Decimal(str(self.max_notional.value())), Decimal(str(self.threshold.value())), self.period.value())
            self.store.add_rule(rule)
            self.reload_tables()
            self.message.setText("규칙 저장됨 · 자동주문은 별도로 켜야 합니다. 시도 후 재실행하려면 내역 확인 후 새 규칙을 등록하세요.")
        except Exception as exc:
            self.message.setText(f"규칙 등록 실패: {exc}")

    def pause_rule(self):
        rule = self.selected_rule()
        if rule and not self.monitoring and not self.worker:
            self.store.pause_rule(rule.id)
            self.reload_tables()

    def review_attempt(self):
        rule = self.selected_rule()
        if not rule or self.monitoring or self.worker or rule.status not in {"unknown", "submitting", "accepted"}:
            self.message.setText("확인할 미확정/미체결 규칙을 선택하고 감시를 중지하세요.")
            return
        if QMessageBox.question(self, "주문 내역 직접 확인 필요",
            "영웅문에서 해당 주문의 접수·체결·취소 여부를 확인하고 남은 주문을 정리했습니까?\n"
            "확인 후 해제하면 이 종목의 다른 규칙이 주문할 수 있습니다. 기존 규칙은 재실행되지 않습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.store.mark_reviewed(rule.id, "CHECKED_ORDER_HISTORY")
            self.reload_tables()

    def reject(self):
        self.close()

    def closeEvent(self, event):
        self._close_when_idle = True
        self._pending_environment = None
        self.environment_timer.stop()
        self.stop_monitoring()
        if self.worker or self._inspection_worker or self._activity_worker or self._schedule_probe:
            event.ignore()
        else:
            self.health_timer.stop()
            self.order_status_timer.stop()
            event.accept()
