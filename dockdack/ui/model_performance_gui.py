"""Read-only model execution performance. Rendering never reads a DB or broker."""
from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QHeaderView, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from dockdack.gui import label, table
from dockdack.models import TradingMode
from dockdack.ui.operations_gui import populate


MODELS = (("mark1-prototype", "mark1 prototype"),
          ("mark1-1-prototype", "mark1.1 prototype"),
          ("mark1-2-prototype", "mark1.2 prototype"))
MARKETS = (("domestic", "국내 · KRW", "KRW"), ("us", "미국 · USD", "USD"))
RETURN_DESCRIPTION = "누적 실현 수익률 = 확인된 매도 실현손익 ÷ 해당 매도분의 매입원가 × 100 · 수수료·세금 제외"
WARNING_TEXT = {
    "model_environment_mismatch": "모의 전용 모델의 기록이 실전 장부에 있어 모델 성과로 합산하지 않았습니다.",
    "unassigned_model_origin": "매수 모델 출처를 확인할 수 없는 매도가 미분류 항목에 있습니다.",
    "invalid_model_origin": "저장된 매수 모델 정보가 올바르지 않아 출처를 확정하지 못했습니다.",
    "invalid_time": "주문시각이 올바르지 않아 거래 순서를 확정하지 못했습니다.",
    "invalid_quantity": "체결 수량이 없거나 주문 수량과 일치하지 않습니다.",
    "invalid_instrument": "시장·거래소·종목·통화 정보가 불완전합니다.",
    "invalid_side": "매수·매도 구분을 확인하지 못했습니다.",
    "invalid_lot_allocation": "모델별 매도와 원래 매수의 연결을 확인하지 못했습니다.",
}


def _mode(value):
    value = getattr(value, "value", value)
    return TradingMode.REAL if value == "live" else TradingMode(value)


def _money(value, currency, *, signed=False):
    if value is None:
        return "미확인"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:,.{0 if currency == 'KRW' else 2}f} {currency}"


def _warning_text(item):
    if not isinstance(item, Mapping):
        return str(item)
    if item.get("reason") or item.get("message"):
        return str(item.get("reason") or item["message"])
    codes = item.get("reason_codes") or (item.get("code"),)
    if isinstance(codes, str):
        codes = (codes,)
    return " · ".join(dict.fromkeys(WARNING_TEXT.get(code, "장부 정보 확인이 필요합니다.") for code in codes))


class ModelPerformancePanel(QWidget):
    """Display a precomputed, full-history report from the selected account.

    The caller owns collection, refresh cadence and stale worker rejection.
    ``store`` is metadata only: no method on it is ever called by this widget.
    Environment changes clear the previous report before showing a new badge.
    """

    request_refresh = Signal()

    def __init__(self, store=None, parent=None, *, mode=None):
        super().__init__(parent)
        self.store = None
        self.mode = TradingMode.DEMO
        self._context_key = None
        self._applied_report = None
        self.report = None
        self.tables, self.summaries, self.scroll_areas = {}, {}, {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(label("모델별 누적 실현 성과", "section"), 1)
        self.mode_badge = label("", "badge")
        heading.addWidget(self.mode_badge)
        self.refresh_button = QPushButton("성과 새로고침")
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.setToolTip("현재 계정의 저장된 전체 체결 장부만 다시 집계합니다. API 조회나 주문 활성화는 하지 않습니다.")
        self.refresh_button.clicked.connect(self.request_refresh.emit)
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
        self.explanation = label(RETURN_DESCRIPTION + "\n계좌 전체 수익률·평가손익·복리 수익률이 아닙니다. KRW와 USD는 합산하지 않습니다.", "muted", wrap=True)
        basis = label('확인된 매도 체결 기준 · 매도분 매입원가 대비 · 수수료·세금 및 미매도 평가손익 제외', 'muted', wrap=True)
        basis.setToolTip(RETURN_DESCRIPTION)
        layout.addWidget(basis)
        self.coverage_label = label("현재 계정의 저장된 장부 집계 대기", "muted", wrap=True)
        self.refresh_error = label("", "error", wrap=True)
        self.refresh_error.hide()
        layout.addWidget(self.refresh_error)
        self.market_tabs = QTabWidget()
        for market, title, currency in MARKETS:
            page = QWidget()
            page.setObjectName("modelPerformanceContent")
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            page.setStyleSheet("QWidget#modelPerformanceContent { background: #121b2a; }")
            content = QVBoxLayout(page)
            content.setContentsMargins(8, 8, 8, 8)
            content.setSpacing(6)
            summary = label("집계 대기 · 거래 없음과 구분합니다.", "muted", wrap=True)
            self.summaries[market] = summary
            view = table(["매수 모델", "누적 실현 수익률 (세전)", "매도분 매입원가", "실현손익 (세전)", "매도 체결", "집계 상태"])
            view.setAccessibleName(f"{title} 모델별 누적 실현 성과")
            view.setWordWrap(True)
            view.setMinimumHeight(140)
            view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            view.horizontalHeader().setStretchLastSection(True)
            for column, width in enumerate((180, 205, 180, 180, 110, 200)):
                view.setColumnWidth(column, width)
            view.verticalHeader().setDefaultSectionSize(44)
            self.tables[market] = view
            content.addWidget(view, 1)
            content.addWidget(summary)
            content.addWidget(label(f"{currency} 체결분만 표시 · 모델 이름은 당시 매수 기록 기준 · 현재 모델 선택과 무관", "muted", wrap=True))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self.scroll_areas[market] = scroll
            self.market_tabs.addTab(scroll, title)
        layout.addWidget(self.market_tabs, 1)
        layout.addWidget(self.coverage_label)
        layout.addWidget(self.explanation)
        self.warning = label("", "error", wrap=True)
        layout.addWidget(self.warning)
        self.read_only_notice = label("기록 조회 전용 · 실전 기록을 보더라도 주문은 켜지지 않습니다. 모의·실전과 계정 장부는 분리됩니다.\n앱 재시작 후에도 저장된 체결 장부에서 전체 기간을 다시 집계합니다. 과거 기록이 없거나 출처·원가가 불명확하면 추정하지 않습니다.", "muted", wrap=True)
        layout.addWidget(self.read_only_notice)
        self.set_context(store, mode=mode)

    def set_context(self, store=None, *, mode=None):
        """Read metadata only and discard a report when the account changes."""
        target_store = store if store is not None else self.store
        target_mode = _mode(mode if mode is not None else getattr(target_store, "mode", self.mode))
        key = (target_mode, str(getattr(target_store, "path", "")), getattr(target_store, "storage_scope", None))
        self.store, self.mode = target_store, target_mode
        self.mode_badge.setText("실전 기록 · REAL" if target_mode is TradingMode.REAL else "모의투자 기록 · DEMO")
        self.mode_badge.setStyleSheet("color: #ffd2d8; background: #4b2432; border: 1px solid #a65066;" if target_mode is TradingMode.REAL else "")
        self.mode_badge.setToolTip("현재 선택한 환경의 계정 장부만 표시합니다. 기록 조회는 주문을 활성화하지 않습니다.\n" + key[1])
        if key == self._context_key:
            return False
        self._context_key = key
        self._applied_report = self.report = None
        self.coverage_label.setText("현재 계정의 저장된 장부 집계 대기")
        self.set_refresh_error("")
        self.warning.clear()
        self.warning.hide()
        for market, _, _ in MARKETS:
            populate(self.tables[market], [(title, "—", "—", "—", "—", "집계 대기") for _, title in MODELS], [model for model, _ in MODELS])
            self.summaries[market].setText("집계 대기 · 거래 없음과 구분합니다.")
        return True

    def set_mode(self, mode):
        return self.set_context(mode=mode)

    def set_refresh_error(self, message):
        """Keep old evidence visible, explicitly marked stale after a read error."""
        text = ("갱신 실패 · 이전 수치 표시 중" if self.report is not None else "갱신 실패 · 성과 미확인") if message else ""
        if message:
            text += "\n" + str(message)
        self.refresh_error.setText(text)
        self.refresh_error.setVisible(bool(message))

    def apply_snapshot(self, snapshot, *, force=False):
        report = snapshot if isinstance(snapshot, Mapping) else getattr(snapshot, "model_performance", None)
        return self.apply_report(report, force=force) if report is not None else False

    def apply_report(self, report, *, force=False):
        """Paint an already computed report. Never infer missing fills or costs."""
        if _mode(report["mode"]) is not self.mode:
            return False  # A late old-environment result must not cross badges.
        self.set_refresh_error("")
        if report is self._applied_report and not force:
            return False
        self._applied_report = self.report = report
        coverage = report.get("coverage", {})
        first, last = coverage.get("first_order_at"), coverage.get("last_order_at")
        period = f"{first} ~ {last}" if first and last else "기록 기간 미확인" if coverage.get("ledger_order_count") else "저장된 주문 없음"
        self.coverage_label.setText(f"현재 계정 · 저장된 전체 주문 {coverage.get('ledger_order_count', 0):,}건 · {period}")
        self.coverage_label.setToolTip("전체 장부를 집계하며 최근 화면 표시 건수 제한과 무관합니다. 기간은 저장된 주문시각 기준입니다.")
        rows = tuple(report.get("rows", ()))
        for market, _, currency in MARKETS:
            selected = tuple(row for row in rows if row.get("market") == market and row.get("currency") == currency)
            cells = []
            for row in selected:
                known = row.get("known_sell_count", 0)
                unknown = row.get("unknown_sell_count", 0)
                no_sales = row.get("status") == "no_sales"
                incomplete = not row.get("complete", True) or bool(unknown)
                rate = row.get("known_return_pct")
                rate_text = f"{rate:+.2f}%" if rate is not None else "미확인"
                cost = _money(row.get("known_cost_basis"), currency)
                profit = _money(row.get("known_realized_profit"), currency, signed=True)
                state = f"손익 확인 {known}건 / 미확인 {unknown}건"
                if no_sales:
                    rate_text = cost = profit = "—"
                    state = "매도 체결 없음"
                elif incomplete:
                    rate_text = "전체 미확인" + (f"\n확인분 {rate_text}" if rate is not None else "")
                    cost = f"확인분 {cost}" if known else "미확인"
                    profit = f"확인분 {profit}" if known else "미확인"
                if row.get("strategy_id") == "unassigned" or row.get("attribution_complete") is False:
                    state = "모델 출처 미확인\n" + state
                count = row.get("executed_sell_count", known + unknown)
                cells.append((row["model_title"], rate_text, cost, profit, f"{count}건", state))
            view = self.tables[market]
            populate(view, cells, [row["strategy_id"] for row in selected])
            for index, row in enumerate(selected):
                needs_detail = not row.get('complete', True) or row.get('attribution_complete') is False
                view.setRowHeight(index, 66 if needs_detail else 44)
                tooltip = (RETURN_DESCRIPTION + "\n확인된 체결가격과 당시 매수 원가만 사용하며 현재가·주문가로 대체하지 않습니다.\n"
                           f"원가 확인 매도 수량: {row.get('known_quantity', 0)}주 / 미확인: {row.get('unknown_quantity', 0)}주\n"
                           "미확인 손익은 0원이 아닙니다. 미분류 거래는 특정 모델 성과에 넣지 않습니다.\n"
                           "하나의 매도가 여러 모델 보유분을 정리할 수 있으므로 모델별 매도 건수는 합산하지 않습니다.")
                for column in range(view.columnCount()):
                    view.item(index, column).setToolTip(tooltip)
            total = sum(row.get("known_sell_count", 0) + row.get("unknown_sell_count", 0) for row in selected)
            self.summaries[market].setText(f"{currency} · 전체 저장 이력 기준 · 모델별 매도 건수는 합산하지 않습니다." if total else f"{currency} · 매도 체결 기록 없음 · 수익률을 0%로 표시하지 않습니다.")
        warnings = tuple(report.get("warnings", ()))
        details = tuple(dict.fromkeys(_warning_text(item) for item in warnings))
        messages = details[:3]
        incomplete = coverage.get("incomplete_sell_count", 0)
        unassigned = coverage.get("unassigned_sell_count", 0)
        warning = f"확인 필요 · 모델 미분류 매도 {unassigned}건 / 손익 미확인 {incomplete}건" if unassigned or incomplete or warnings or not report.get("complete", True) else ""
        if messages:
            warning += "\n" + " · ".join(messages)
        self.warning.setText(warning)
        self.warning.setToolTip("\n".join(details))
        self.warning.setVisible(bool(warning))
        return True
