"""Controller for an explicitly requested demo session and its current status.

Starting this controller requires a caller's user-authorized configuration.
The normal GUI does not construct it or automatically arm orders.
"""

import json
import os
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Slot

from dockdack.signal_bridge import atomic_json
from dockdack.watchlist import utc_now
from dockdack.environment_store import selected_mode
from dockdack.models import TradingMode


class SessionController(QObject):
    def __init__(self, window, folder, *, auto_arm=True):
        super().__init__(window)
        self.window, self.folder = window, folder
        self.session_id = uuid4().hex
        self.started_at = utc_now().isoformat()
        # This launch controller is explicitly DEMO-only. A warning/mode
        # selection can never inherit the previous demo startup approval.
        self.pending_arm = auto_arm and selected_mode(window.service) is TradingMode.DEMO
        self.phase = "initializing"
        self.reason = ""
        self._external_error_baseline = window.engine.external_error_count
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.report)

    @property
    def pending_arm(self):
        return self.window.pending_auto_arm

    @pending_arm.setter
    def pending_arm(self, enabled):
        self.window.set_pending_auto_arm(enabled)

    @Slot()
    def cancel_arm(self):
        self.pending_arm = False

    @Slot()
    def start(self):
        self._external_error_baseline = self.window.engine.external_error_count
        self.window.start_monitoring()
        if not self.window.monitoring or self.window.worker is None:
            self.pending_arm = False
            self.phase = "start_blocked"
            self.reason = self.window.message.text()
        elif self.pending_arm:
            self.phase = "warmup_orders_off"
            self.window.worker.signals.completed.connect(self.warmup_finished)
        else:
            self.phase = "monitoring_orders_off"
        self.timer.start()
        self.report()

    @Slot(object, object)
    def warmup_finished(self, results, error):
        window = self.window
        if selected_mode(window.service) is not TradingMode.DEMO:
            self.cancel_arm()
            self.phase = 'demo_startup_permission_not_applicable'
            self.report()
            return
        if window._manual_arm_pending:
            # The GUI's newer explicit request owns its own full-sweep completion.
            return
        should_arm = self.pending_arm
        self.pending_arm = False  # Consume once, including cancellation/error.
        if not should_arm or not window.monitoring:
            self.phase = "auto_arm_cancelled"
            self.report()
            return
        failure = window.activation_failure(results, error, external_error_baseline=self._external_error_baseline)
        if failure:
            self.phase = "warmup_failed_orders_off"
            self.reason = failure
            window.engine.disarm()
            window.message.setText("초기 조회 실패 · 자동주문 OFF 유지 · ON 예약 취소. 활동 기록을 확인하세요.")
            window.update_controls()
            self.report()
            return
        try:
            window.timer.stop()
            window.engine.enable_orders("DEMO_AUTOTRADE")
            self.phase = "auto_arm_completed"
            self.reason = ""
            window.update_controls()
            self.report()
            if window.monitoring and window.engine.orders_enabled:
                window.refresh_all()
        except Exception as exc:
            window.engine.disarm()
            self.phase, self.reason = "arm_failed_orders_off", str(exc)
            window.message.setText(f"자동주문 OFF · 켜기 실패: {exc}")
            window.update_controls()
            self.report()

    @Slot()
    def report(self):
        window = self.window
        stop_path = self.folder / "demo-session-stop.json"
        if stop_path.exists():
            try:
                request = json.loads(stop_path.read_text(encoding="utf-8"))
                if request == {"session_id": self.session_id, "action": "stop"}:
                    self.cancel_arm()
                    window.stop_monitoring()
            except (OSError, ValueError):
                pass
        if not window.monitoring and not window._manual_arm_pending:
            self.cancel_arm()
        # Never infer permission from a previous lifecycle phase. In particular,
        # manual ON after a startup error must immediately be reported as ON.
        status = window.automation_status()
        window._sync_order_controls(status)
        with window.store.connection() as db:
            orders = dict(db.execute("SELECT status,COUNT(*) FROM attempts WHERE started_at>=? GROUP BY status", (self.started_at,)))
        atomic_json(self.folder / "demo-session-status.json", {
            "session_id": self.session_id, "pid": os.getpid(), "started_at": self.started_at,
            "reported_at": utc_now().isoformat(), **status,
            "startup_phase": self.phase,
            "reason": "" if status["orders_enabled"] else self.reason,
            "worker_active": window.worker is not None,
            "quantity": window.external_quantity.value(), "us_order_type": window.random_us.currentData(),
            "max_krw": str(window.external_krw.value()), "max_usd": str(window.external_usd.value()),
            "watch_count": len(window._items_by_id), "fresh_count": len(window.fresh_ids),
            "errors": window.errors, "session_order_status_counts": orders,
            "orders_badge": window.mode_label.text(),
            "orders_detail": window.order_status_detail.text(),
            "on_button_enabled": window.arm_button.isEnabled(),
            "off_button_enabled": window.disarm_button.isEnabled(),
            "message": window.message.text(),
            "dashboard": {
                "window_visible": window.isVisible(),
                "window_minimized": window.isMinimized(),
                "page": window.workspace_tabs.tabText(window.workspace_tabs.currentIndex()),
                "watch_market": window.watch_market.value,
                "watch_market_counts": {market.value: view.rowCount() for market, view in window.watch_tables.items()},
                "holdings_market": window.portfolio_panel.current_market.value,
                "holdings_market_tabs": window.portfolio_panel.market_tabs.count(),
                "health": window.health_label.text(),
                "holdings": {
                    market.value: {"count": len(state.positions) if state.snapshot is not None else None,
                                   "fetched_at": state.fetched_at.isoformat() if state.fetched_at else None,
                                   "status": state.status(), "error": state.error}
                    for market, state in window._portfolio_payload.items()
                },
                "order_rows": len(window.order_history_panel.records),
                "api_rates": window.api_rate_status(),
                "fill_recovery": {key: (value.isoformat() if hasattr(value, "isoformat") else value)
                                  for key, value in window._fill_recovery_status.items()},
                "connection": {key: value for key, value in window.connection_status().items()
                               if key in {"configured", "active", "producer", "source_id", "reader_state", "reader_error",
                                          "received_counts", "block_reason", "market_summary", "configuration_pending"}},
            },
        })
