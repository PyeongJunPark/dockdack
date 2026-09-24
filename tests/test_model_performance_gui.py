"""Offline, read-only rendering checks for model execution performance."""
from __future__ import annotations

import importlib.util
import os
from decimal import Decimal as D
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.ui.model_performance_gui import ModelPerformancePanel


def performance_report(*, mode="demo", unknown=0, no_sales=False, unassigned=False):
    rows = []
    for market, currency in (("domestic", "KRW"), ("us", "USD")):
        for strategy, title in (("mark1-prototype", "mark1 prototype"), ("mark1-1-prototype", "mark1.1 prototype")):
            rows.append({"strategy_id": strategy, "model_title": title, "market": market, "currency": currency,
                         "known_sell_count": 0 if no_sales else 1, "unknown_sell_count": unknown,
                         "known_quantity": D(0 if no_sales else 2), "unknown_quantity": D(unknown),
                         "known_cost_basis": None if no_sales else D(100), "known_realized_profit": None if no_sales else D(1),
                         "known_return_pct": None if no_sales else D(1), "complete": not unknown,
                         "status": "no_sales" if no_sales else "incomplete" if unknown else "known"})
    if unassigned:
        rows.append({**rows[0], "strategy_id": "unassigned", "model_title": "미확인 / 수동·외부", "attribution_complete": False})
    return {"mode": mode, "rows": tuple(rows), "complete": not (unknown or unassigned), "warnings": (), "gross": True,
            "coverage": {"ledger_order_count": 0 if no_sales else 8, "incomplete_sell_count": unknown * 4,
                         "unassigned_sell_count": int(unassigned)}}


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class ModelPerformanceGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.store = SimpleNamespace(mode="demo", path="offline-demo.sqlite3", storage_scope="test-account",
                                     order_history=Mock(side_effect=AssertionError("UI must not read ledger")))
        self.panel = ModelPerformancePanel(self.store)

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def test_starts_with_unknown_collection_not_zero_percent(self):
        self.assertIn("DEMO", self.panel.mode_badge.text())
        for market in ("domestic", "us"):
            self.assertEqual(self.panel.tables[market].rowCount(), 2)
            self.assertEqual(self.panel.tables[market].item(0, 1).text(), "—")
            self.assertEqual(self.panel.tables[market].item(0, 5).text(), "집계 대기")
        self.store.order_history.assert_not_called()

    def test_two_models_in_separate_currencies_and_execution_return(self):
        self.panel.apply_report(performance_report())
        self.assertEqual(self.panel.market_tabs.count(), 2)
        for market, currency in (("domestic", "KRW"), ("us", "USD")):
            view = self.panel.tables[market]
            self.assertEqual(view.rowCount(), 2)
            self.assertEqual(view.item(0, 1).text(), "+1.00%")
            self.assertIn(currency, view.item(0, 2).text())
            self.assertIn(currency, view.item(0, 3).text())
            self.assertEqual(view.item(0, 4).text(), "1건")
            self.assertIn("수수료·세금 제외", view.item(0, 1).toolTip())
        self.assertIn("KRW와 USD는 합산하지 않습니다", self.panel.explanation.text())
        self.store.order_history.assert_not_called()

    def test_real_backend_report_schema_and_whole_history_projection(self):
        from dockdack.model_performance import model_realized_performance
        from test_model_performance import order
        records = (order(1, "buy", model="mark1-prototype"), order(2, "sell", price="102"))
        self.panel.apply_report(model_realized_performance(records, mode="demo"))
        self.assertEqual(self.panel.tables["domestic"].item(0, 1).text(), "+2.00%")
        self.assertEqual(self.panel.tables["domestic"].item(0, 3).text(), "+2 KRW")
        self.assertEqual(self.panel.tables["us"].item(0, 5).text(), "매도 체결 없음")
        self.assertIn("전체 주문 2건", self.panel.coverage_label.text())
        self.store.order_history.assert_not_called()

    def test_missing_sales_are_not_zero_performance(self):
        self.panel.apply_report(performance_report(no_sales=True))
        self.assertEqual(self.panel.tables["domestic"].item(0, 1).text(), "—")
        self.assertEqual(self.panel.tables["domestic"].item(0, 5).text(), "매도 체결 없음")
        self.assertIn("저장된 주문 없음", self.panel.coverage_label.text())

    def test_known_and_unknown_parts_of_one_sale_do_not_double_count(self):
        report = performance_report(unknown=1)
        report["rows"][0]["executed_sell_count"] = 1
        self.panel.apply_report(report)
        self.assertEqual(self.panel.tables["domestic"].item(0, 4).text(), "1건")
        self.assertIn("합산하지 않습니다", self.panel.summaries["domestic"].text())

    def test_incomplete_and_unassigned_are_explicit_not_folded_into_model(self):
        self.panel.apply_report(performance_report(unknown=1, unassigned=True))
        view = self.panel.tables["domestic"]
        self.assertEqual(view.rowCount(), 3)
        self.assertIn("전체 미확인", view.item(0, 1).text())
        self.assertIn("확인분 +1.00%", view.item(0, 1).text())
        self.assertEqual(view.item(2, 0).text(), "미확인 / 수동·외부")
        self.assertIn("모델 출처 미확인", view.item(2, 5).text())
        self.assertIn("모델 미분류 매도 1건", self.panel.warning.text())

    def test_provenance_warning_codes_have_distinct_human_explanations(self):
        report = performance_report(unassigned=True)
        report["warnings"] = tuple({"reason_codes": (code,)} for code in
                                   ("model_environment_mismatch", "unassigned_model_origin", "invalid_model_origin"))
        self.panel.apply_report(report)
        self.assertIn("모의 전용 모델의 기록이 실전 장부", self.panel.warning.text())
        self.assertIn("출처를 확인할 수 없는 매도", self.panel.warning.text())
        self.assertIn("매수 모델 정보가 올바르지 않아", self.panel.warning.text())

    def test_refresh_only_emits_and_real_mode_clears_old_data(self):
        self.panel.apply_report(performance_report())
        requested = Mock()
        self.panel.request_refresh.connect(requested)
        self.panel.refresh_button.click()
        requested.assert_called_once_with()
        self.store.order_history.assert_not_called()
        self.panel.set_mode("real")
        self.assertIn("REAL", self.panel.mode_badge.text())
        self.assertEqual(self.panel.tables["domestic"].item(0, 5).text(), "집계 대기")
        self.assertFalse(self.panel.apply_report(performance_report(mode="demo")))
        self.assertTrue(self.panel.apply_report(performance_report(mode="real")))
        self.assertIn("주문은 켜지지 않습니다", self.panel.read_only_notice.text())

    def test_same_mode_account_switch_also_clears_old_report(self):
        self.panel.apply_report(performance_report())
        replacement = SimpleNamespace(mode="demo", path="other.sqlite3", storage_scope="other")
        self.panel.set_context(replacement)
        self.assertIsNone(self.panel.report)
        self.assertEqual(self.panel.tables["us"].item(0, 5).text(), "집계 대기")

    def test_snapshot_apply_and_identical_report_skip_repaint(self):
        report = performance_report()
        snapshot = SimpleNamespace(model_performance=report)
        self.assertTrue(self.panel.apply_snapshot(snapshot))
        cell = self.panel.tables["domestic"].item(0, 0)
        self.assertFalse(self.panel.apply_snapshot(snapshot))
        self.assertIs(self.panel.tables["domestic"].item(0, 0), cell)
        self.assertFalse(self.panel.apply_snapshot(SimpleNamespace(model_performance=None)))

    def test_refresh_failure_keeps_values_but_explicitly_marks_stale(self):
        report = performance_report()
        self.panel.apply_report(report)
        self.panel.set_refresh_error("offline read failed")
        self.assertIn("이전 수치", self.panel.refresh_error.text())
        self.assertEqual(self.panel.tables["domestic"].item(0, 1).text(), "+1.00%")
        self.panel.apply_report(report)
        self.assertEqual(self.panel.refresh_error.text(), "")

    def test_narrow_panel_preserves_table_columns_with_scroll(self):
        self.panel.resize(640, 420)
        self.panel.show()
        self.panel.apply_report(performance_report(unknown=1))
        self.app.processEvents()
        view = self.panel.tables["domestic"]
        self.assertGreater(view.horizontalScrollBar().maximum(), 0)
        self.assertGreaterEqual(view.height(), 140)
        self.assertGreater(self.panel.scroll_areas["domestic"].verticalScrollBar().maximum(), 0)
        self.assertGreaterEqual(view.rowHeight(0), 60)
        self.store.order_history.assert_not_called()


if __name__ == "__main__":
    unittest.main()
