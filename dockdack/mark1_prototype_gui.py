"""Latest saved research models in a permanently observation-only dashboard."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from dockdack.gui_service import TradingService
from dockdack.history import market_time
from dockdack.lstm30_adapter import atomic_json, previous_trading_day, read_json
from dockdack.lstm30_gui import LSTM30WatchlistDialog
from dockdack.mark1_adapter import TARGET_BASIS
from dockdack.mark1_gui import Mark1WatchlistDialog
from dockdack.mark1_prototype_adapter import (
    ORDER_BLOCK_REASON, RISK_NOTICE, SOURCE_ID, STRATEGY_ID, STRATEGY_NOTICE,
    PrototypeAutoTrader, PrototypeReadOnlyService, PrototypeSignalProducer,
)
from dockdack.market_schedule import calendar_for
from dockdack.models import Market, TradingMode
from dockdack.watchlist import utc_now

DEFAULT_RUNTIME_DIR = Path(".dockdack/mark1-prototype")
ACTION_LABELS = {"buy": "매수", "sell": "매도", "hold": "대기", "none": "미전달"}
REASON_LABELS = {
    "PREDICTED_DAILY_BARRIER_SUCCESS": "모델 추정 확률 50% 초과 · 연구 매수 신호",
    "BELOW_OR_EQUAL_BUY_THRESHOLD": "모델 추정 확률 50% 이하 · 대기",
    "TAKE_PROFIT_1PCT": "평균매입가 대비 +1% 익절 기준 도달",
    "STOP_LOSS_0_9PCT": "평균매입가 대비 -0.9% 손절 기준 도달",
    "POSITION_INSIDE_EXIT_BOUNDS": "보유 중 · 익절/손절 기준 미도달",
    "NO_SELLABLE_POSITION": "매도 가능한 보유수량 없음",
    "PREDICTION_UNAVAILABLE": "모델 입력 또는 추론 확인 필요 · 대기",
    "QUOTE_OR_POSITION_UNAVAILABLE": "현재가 또는 보유정보 확인 필요 · 대기",
    "QUOTE_EXPIRED_DURING_INFERENCE": "추론 중 시세 유효시간 초과 · 대기",
    "POSITION_EXPIRED_DURING_INFERENCE": "보유정보 유효시간 초과 · 대기",
    "USER_QUANTITY_OR_NOTIONAL_CAP": "연구 신호의 수량/금액 한도 초과 · 대기",
    "UNREGISTERED_INVALID_CHART": "차트 조회 미완료 · 신호 미전달",
}


def _marker_for(predictors):
    fingerprints = {}
    expected = {"domestic": "cat_joint6", "us": "cat_binary8"}
    for market, predictor in predictors.items():
        metadata = getattr(predictor, "metadata", {})
        fingerprint = metadata.get("bundle_manifest_sha256")
        if (market not in expected or metadata.get("market") != market
                or metadata.get("architecture", metadata.get("model_name")) != expected[market]
                or metadata.get("research_only") is not True or metadata.get("deployment_allowed") is not False
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)):
            raise ValueError("검증된 mark1 prototype 모델·시장·연구 전용 표시·번들 지문이 필요합니다.")
        fingerprints[market] = fingerprint
    if not fingerprints:
        raise ValueError("mark1 prototype 학습 모델이 필요합니다.")
    return {"strategy": STRATEGY_ID, "source_id": SOURCE_ID,
            "bundle_manifest_sha256": fingerprints, "observation_only": True, "target_basis": TARGET_BASIS}


class Mark1PrototypeWatchlistDialog(Mark1WatchlistDialog):
    """Reuse the existing reader/table behavior without old strategy markers."""
    source_id = SOURCE_ID
    strategy_id = STRATEGY_ID
    producer_class = PrototypeSignalProducer
    engine_class = PrototypeAutoTrader

    def __init__(self, service=None, *, runtime_dir=DEFAULT_RUNTIME_DIR, predictors, **kwargs):
        root = Path(runtime_dir).resolve()
        if root.name in {"mark1-demo", "lstm30-demo"}:
            raise ValueError("기존 자동매매 폴더는 사용할 수 없습니다. prototype 전용 폴더가 필요합니다.")
        if kwargs.get("close_all_before_minutes") is not None or kwargs.get("close_all_confirmation") is not None:
            raise ValueError(ORDER_BLOCK_REASON + " 마감 청산도 사용할 수 없습니다.")
        marker_payload = _marker_for(predictors)
        marker = root / "strategy.json"
        if marker.exists():
            if read_json(marker) != marker_payload:
                raise ValueError("다른 전략/모델 번들의 실행 폴더입니다. prototype 전용 폴더를 사용하세요.")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("기존 장부를 재사용할 수 없습니다. 비어 있는 prototype 전용 폴더를 사용하세요.")
        decision_path = root / "exchange/decisions.json"
        if decision_path.exists():
            decisions = read_json(decision_path)
            if not isinstance(decisions, dict) or any(
                not isinstance(row, dict) or not isinstance(row.get("payload"), dict)
                or row["payload"].get("source_id") != SOURCE_ID for row in decisions.values()
            ):
                raise ValueError("다른 전략의 결정 상태입니다. prototype 전용 폴더를 사용하세요.")
        now = kwargs.get("clock", utc_now)()
        for market in Market:
            calendar_for(market, market_time(market, now).year)
            previous_trading_day(market.value, now)
        service = service if service is not None else TradingService(mode=TradingMode.DEMO)
        service = service if isinstance(service, PrototypeReadOnlyService) else PrototypeReadOnlyService(service)
        # Mark1WatchlistDialog.__init__ hardcodes its own SOURCE_ID marker.
        # Skip only that initializer; inherit its table rendering/shutdown hooks.
        LSTM30WatchlistDialog.__init__(self, service, runtime_dir=root, predictors=predictors, **kwargs)
        self.engine.predictors = dict(predictors)
        atomic_json(marker, marker_payload)  # Shared initializer now owns SessionLock.
        self.close_timer.stop()             # No account-wide closing-order worker.
        self.mark1_strategy = QLabel(STRATEGY_NOTICE)
        self.mark1_strategy.setWordWrap(True)
        self.mark1_model_summary = QLabel()
        names = [f"{'국내' if market == 'domestic' else '미국'}: {predictor.metadata.get('model_name', predictor.metadata.get('architecture', '모델'))} · 3개 시드 앙상블"
                 for market, predictor in sorted(predictors.items())]
        self.mark1_model_summary.setText("연결된 저장 모델 · " + " / ".join(names))
        self.mark1_model_summary.setWordWrap(True)
        self.mark1_limitations = QLabel(
            RISK_NOTICE + "\n" + ORDER_BLOCK_REASON + "\n"
            "완료된 과거 30일 OHLCV와 현재가로 184개 표 특성을 계산합니다. "
            "당일 최종 고가·저가·종가·거래량은 입력하지 않습니다.\n"
            "표시 확률은 하루 전체 구간의 보수적 장벽 사건 추정이며, 장중 진입 이후의 성공확률이나 체결을 보장하지 않습니다. "
            "양쪽 경계를 모두 건드리면 손절 우선으로 실패 처리합니다.\n"
            "감시는 시작 시 OFF입니다. 직접 감시를 시작해도 조회와 연구 신호 기록만 수행하며 주문은 항상 차단됩니다."
        )
        self.mark1_limitations.setWordWrap(True)
        self.mark1_limitations.setStyleSheet("color: #ffda91; font-weight: 600; padding: 8px;")
        self.mark1_model_table = QTableWidget(0, 5)
        self.mark1_model_table.setHorizontalHeaderLabels(["종목", "모델", "모델 추정 확률 (미검증)", "연구 신호 (주문 아님)", "판단 사유"])
        self.mark1_model_table.horizontalHeader().setStretchLastSection(True)
        self.mark1_model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mark1_model_table.setStyleSheet(
            "QTableWidget { background: #141e2d; color: #e6edf7; gridline-color: #2c3b50; }"
            "QHeaderView { background: #1d2a3d; color: #dce7f5; }"
            "QHeaderView::section { background: #1d2a3d; color: #dce7f5; border: 1px solid #31425b; padding: 7px; }"
            "QTableCornerButton::section { background: #1d2a3d; border: 1px solid #31425b; }"
        )
        page = QWidget()
        layout = QVBoxLayout(page)
        for widget in (self.mark1_strategy, self.mark1_model_summary, self.mark1_limitations, self.mark1_model_table):
            layout.addWidget(widget)
        self.workspace_tabs.addTab(page, "mark1 prototype · 연구 신호")
        self.mark1_refresh_timer = QTimer(self)
        self.mark1_refresh_timer.setInterval(500)
        self.mark1_refresh_timer.timeout.connect(lambda: self._update_mark1_table(visible_only=True))
        self.workspace_tabs.currentChanged.connect(lambda _: self._update_mark1_table(visible_only=True))
        self.mark1_refresh_timer.start()
        self._update_mark1_table()
        self._sync_environment()
        self._sync_order_controls()
        self.message.setText("mark1 prototype 연결됨 · 감시 OFF · 주문 항상 차단 · 감시는 직접 시작하세요")

    def _apply_execution_preferences(self):
        # Main v0.0 defaults to 10% sizing, independent holding exits and extra
        # feeds. This prototype keeps one isolated fixed-quantity signal source.
        self.percent_sizing.setChecked(False)
        self.percent_sizing.setEnabled(False)
        self.buy_percent.setEnabled(False)
        self.additional_sources.setEnabled(False)
        self.engine.session_only_poll = True
        self.engine.enable_holdings_exits = False
        self.engine.equity_buy_percent = None
        self.engine.isolated_symbol_errors = True
        self.engine.us_retry_attempts = 1
        self.engine.holding_caps = {Market.DOMESTIC: Decimal(str(self.external_krw.value())),
                                    Market.US: Decimal(str(self.external_usd.value()))}
        self.source_status.setText("prototype 단일 연구 신호원 · 추가 연결/비중 주문/자동청산 사용 안 함")

    def _validate_fixed_configuration(self):
        super()._validate_fixed_configuration()
        if self.percent_sizing.isChecked() or self.additional_sources.raw_sources():
            self.engine.disarm()
            raise ValueError("prototype은 고정 수량·단일 신호원만 사용합니다. 추가 연결/비중 주문은 금지됩니다.")

    def _update_mark1_table(self, *, visible_only=False):
        # Translate presentation only. Stored raw actions/reasons remain intact.
        if visible_only and not self.mark1_model_table.isVisible():
            return
        diagnostics = dict(self.lstm_bridge.diagnostics)
        rows = []
        for item in self.lstm_items:
            row = diagnostics.get(item.id, {})
            probability = row.get("prediction", {}).get("probability_success")
            display = "조회 전 / 모델 판단 없음" if probability is None else f"{float(probability):.2%}"
            action, reason = row.get("action", "none"), row.get("reason", "")
            values = (item.instrument.symbol, row.get("model_name", "—"), display,
                      ACTION_LABELS.get(action, "상태 확인 필요"),
                      REASON_LABELS.get(reason, "감시를 시작하지 않았습니다" if not reason else "진단 정보 확인 필요"))
            rows.append((values, action, reason))
        signature = tuple(rows)
        if signature == getattr(self, '_mark1_table_signature', None):
            return
        self._mark1_table_signature = signature
        self.mark1_model_table.setRowCount(len(rows))
        for index, (values, action, reason) in enumerate(rows):
            for column, value in enumerate(values):
                cell = self.mark1_model_table.item(index, column)
                if cell is None:
                    cell = QTableWidgetItem(str(value))
                    self.mark1_model_table.setItem(index, column, cell)
                elif cell.text() != str(value):
                    cell.setText(str(value))
                if column in (3, 4):
                    raw = action if column == 3 else reason
                    cell.setData(Qt.ItemDataRole.UserRole, raw)
                    cell.setToolTip(raw)
        self.mark1_model_table.resizeColumnsToContents()

    def _sync_environment(self):
        LSTM30WatchlistDialog._sync_environment(self)
        self.setWindowTitle("DOCKDACK | mark1 prototype · 관찰 전용")
        self.environment_caption.setText("mark1 prototype · 최신 저장 모델 · 모의환경 조회 / 주문 불가")
        self.environment_notice.setText(RISK_NOTICE + "\n" + STRATEGY_NOTICE)
        self.environment_notice.show()  # Main v0.0 hides this banner; research warnings stay visible here.
        self.portfolio_panel.heading.setText("현재 보유종목 · 모의 계좌 읽기 전용 / +1%·-0.9% 연구 기준")
        # Hide the unrelated mark0 random-test tab and its -.8% descriptions.
        index = self.tabs.indexOf(self.strategy_panel)
        if index >= 0:
            self.tabs.setTabVisible(index, False)
        for label in self.external_panel.findChildren(QLabel):
            if "-0.8%" in label.text() or "−0.8%" in label.text():
                label.setText("prototype 연구 신호만 표시 · 고정 수량 / 주문 전송 없음\n"
                              "보유종목 표시는 평균매입가 +1% / -0.9% · 자동청산·추가 신호원·비중 주문 차단")

    def _update_connection(self):
        LSTM30WatchlistDialog._update_connection(self)
        if not getattr(self, "_lstm_configured", False):
            return
        self.signal_connection_panel.source_label.setText(f"mark1 prototype · 관찰 전용 · source_id: {SOURCE_ID}")
        state = "조회 감시 중" if self.monitoring else "감시 중지"
        self.connection_summary.setText(f"mark1 prototype · >50% / +1% / -0.9% · 주문 항상 차단 · {state} | {self._market_summary}")

    def _sync_order_controls(self, status=None):
        # Parent health/report timers also call this directly, not just update_controls.
        self.pending_auto_arm = False
        self._manual_arm_pending = False
        super()._sync_order_controls()
        self.arm_button.setEnabled(False)
        self.arm_button.setText("자동주문 사용 불가 (관찰 전용)")
        self.arm_button.setToolTip(ORDER_BLOCK_REASON)
        self.disarm_button.setEnabled(False)
        self.mode_label.setText("주문 항상 차단 · 연구 신호 전용")
        self.order_status_detail.setText(ORDER_BLOCK_REASON)

    def set_pending_auto_arm(self, enabled):
        self.pending_auto_arm = self._manual_arm_pending = False
        self.engine.disarm()
        self._sync_order_controls()
        if enabled:
            raise ValueError(ORDER_BLOCK_REASON)

    def enable_auto_orders(self):
        self.engine.disarm()
        self.pending_auto_arm = self._manual_arm_pending = False
        self.message.setText(ORDER_BLOCK_REASON)
        self._sync_order_controls()
        return False

    def confirm_automation(self):
        return False

    def activation_failure(self, *args, **kwargs):
        return ORDER_BLOCK_REASON

    def start_session(self, confirmation=None):
        if confirmation is not None:
            raise ValueError(ORDER_BLOCK_REASON)
        return super().start_session(None)

    def automation_status(self):
        status = super().automation_status()
        status.update(orders_enabled=False, pending_arm=False, research_only=True,
                      deployment_allowed=False, order_block_reason=ORDER_BLOCK_REASON)
        return status
