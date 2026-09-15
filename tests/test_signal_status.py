from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dockdack.signal_bridge import ExternalPolicy
from dockdack.signal_status import inspect_signal_file

NOW = datetime(2026, 9, 15, 6, 43, tzinfo=timezone.utc)
HAS_QT = importlib.util.find_spec("PySide6") is not None
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if HAS_QT:
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication
    from dockdack.signal_connection_gui import SignalConnectionPanel, connection_view


class SignalFileInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "signals.json"
        self.policy = ExternalPolicy("external-model", 1, Decimal("10000000"), Decimal("10000"))

    def tearDown(self):
        self.temp.cleanup()

    def signal(self, action="buy"):
        value = {"signal_id": "decision-1", "export_id": "export-1", "market": "domestic", "symbol": "005930",
                 "exchange": "KRX", "action": action, "generated_at": NOW.isoformat(),
                 "expires_at": (NOW + timedelta(minutes=2)).isoformat()}
        if action != "hold":
            value.update(quantity=1, max_notional="1000000", order_type="limit")
        return value

    def write(self, signals, **overrides):
        payload = {"schema_version": 1, "source_id": "external-model", "signals": signals, **overrides}
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def inspect(self):
        return inspect_signal_file(self.path, self.policy, now=NOW)

    def test_buy_inspection_never_ingests_or_creates_a_store(self):
        self.write([self.signal()])
        with patch("dockdack.signal_bridge.ingest_signals", side_effect=AssertionError("inspection must not ingest")), \
                patch("dockdack.watchlist.WatchStore", side_effect=AssertionError("inspection must not create store")):
            result = self.inspect()
        self.assertEqual(result["state"], "format_ok")
        self.assertEqual(result["counts"]["buy"], 1)
        self.assertIn("아직 접수/주문하지 않았습니다", result["summary"])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.path])

    def test_hold_is_not_a_buy_or_order(self):
        self.write([self.signal("hold")])
        result = self.inspect()
        self.assertEqual(result["state"], "format_ok")
        self.assertEqual(result["counts"], {"buy": 0, "sell": 0, "hold": 1, "expired": 0})

    def test_missing_and_malformed_files_have_no_success_state(self):
        self.assertEqual(self.inspect()["state"], "missing")
        self.path.write_text('{"schema_version":1', encoding="utf-8")
        self.assertEqual(self.inspect()["state"], "error")
        self.path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        self.assertEqual(self.inspect()["state"], "error")

    def test_untrusted_source_unknown_keys_and_wrong_schema_are_rejected(self):
        for overrides in ({"source_id": "attacker"}, {"schema_version": True}, {"code": "print('do not run')"}):
            with self.subTest(overrides=overrides):
                self.write([self.signal()], **overrides)
                self.assertEqual(self.inspect()["state"], "error")

    def test_quantity_and_amount_limits_are_checked_without_ordering(self):
        for overrides in ({"quantity": 2}, {"quantity": True}, {"max_notional": "10000001"},
                          {"max_notional": "NaN"}, {"order_type": "market"}):
            with self.subTest(overrides=overrides):
                self.write([{**self.signal(), **overrides}])
                self.assertEqual(self.inspect()["state"], "error")

    def test_duplicate_symbols_and_hold_order_fields_rejected(self):
        self.write([self.signal(), {**self.signal(), "signal_id": "decision-2"}])
        self.assertEqual(self.inspect()["state"], "error")
        self.write([{**self.signal("hold"), "quantity": 1}])
        self.assertEqual(self.inspect()["state"], "error")

    def test_expired_signals_are_reported_as_expired_not_executable(self):
        value = self.signal()
        value.update(generated_at=(NOW - timedelta(minutes=7)).isoformat(),
                     expires_at=(NOW - timedelta(minutes=1)).isoformat())
        self.write([value])
        result = self.inspect()
        self.assertEqual(result["state"], "format_ok")
        self.assertEqual(result["counts"]["expired"], 1)
        value.update(generated_at=(NOW + timedelta(minutes=1)).isoformat(),
                     expires_at=(NOW + timedelta(minutes=2)).isoformat())
        self.write([value])
        self.assertEqual(self.inspect()["state"], "error")

    def test_stop_loss_inspection_accepts_explicit_sell_metadata_only(self):
        self.write([{**self.signal("sell"), "cost_loss_pct": "0.8"}])
        result = self.inspect()
        self.assertEqual(result["state"], "format_ok")
        self.assertEqual(result["counts"]["sell"], 1)
        for overrides in ({"action": "buy"}, {"action": "hold"}, {"cost_profit_pct": "1"},
                          {"min_sell_price": "100"}, {"cost_loss_pct": "0"},
                          {"cost_loss_pct": "100"}, {"cost_loss_pct": "NaN"}, {"cost_loss_pct": 0.8}):
            with self.subTest(overrides=overrides):
                self.write([{**self.signal("sell"), "cost_loss_pct": "0.8", **overrides}])
                self.assertEqual(self.inspect()["state"], "error")

    def test_export_membership_is_explicitly_not_claimed_verified(self):
        value = self.signal()
        value["export_id"] = "unknown-but-valid-format"
        self.write([value])
        result = self.inspect()
        self.assertEqual(result["state"], "format_ok")
        self.assertIn("파일 형식", result["scope"])
        self.assertNotIn("연결 성공", result["summary"])


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class SignalConnectionPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def base_status(self):
        return {"producer": "external-file", "configured": True, "active": True, "monitoring": True,
                "orders_enabled": False, "source_id": "external-model", "input_path": "C:/signals.json",
                "output_path": "C:/charts.json", "now": NOW}

    def test_configured_file_alone_never_claims_successful_connection(self):
        view = connection_view(self.base_status())
        self.assertIn("연결 성공은 아직 미확인", view["incoming"])
        self.assertIn("접수 확인 없음", view["accepted"])
        self.assertIn("저장 성공 기록 없음", view["export"])

    def test_demo_and_external_sources_and_off_gate_are_distinct(self):
        status = {**self.base_status(), "producer": "random-demo", "source_id": "random-demo",
                  "last_read_at": NOW, "last_accepted_at": NOW, "received_counts": {"queued": 1, "hold": 1}}
        view = connection_view(status)
        self.assertIn("외부 프로그램 연결 아님", view["source"])
        self.assertIn("OFF", view["gate"])
        self.assertIn("주문·체결이 아닙니다", view["accepted"])
        self.assertIn("HOLD 1", view["accepted"])
        self.assertIn("ON 예약됨", connection_view({**status, "pending_arm": True})["gate"])
        self.assertIn("ON · 유효 신호", connection_view({**status, "orders_enabled": True})["gate"])

    def test_missing_error_and_stale_have_explicit_observed_states(self):
        status = self.base_status()
        self.assertIn("입력 파일 없음", connection_view({**status, "reader_state": "missing"})["incoming"])
        failed = connection_view({**status, "reader_state": "error", "reader_error": "source_id mismatch"})
        self.assertIn("source_id mismatch", failed["incoming"])
        self.assertEqual(failed["tone"], "error")
        stale = connection_view({**status, "last_accepted_at": NOW - timedelta(minutes=6)})
        self.assertIn("5분 경과", stale["accepted"])
        self.assertNotIn("연결 끊김", stale["accepted"])

    def test_reader_object_does_not_mean_active_when_monitoring_has_stopped(self):
        view = connection_view({**self.base_status(), "monitoring": False, "last_read_at": NOW,
                                "last_accepted_at": NOW, "reader_state": "ok"})
        self.assertIn("수신기 비활성", view["mode"])
        self.assertIn("수신 중지", view["incoming"])

    def test_symbol_update_success_never_claims_aggregate_file_was_saved(self):
        view = connection_view({**self.base_status(), "update_path": "C:/charts_updates", "last_update_at": NOW})
        lines = view["export"].splitlines()
        self.assertIn("종목별 즉시 파일 · 저장 성공", lines[0])
        self.assertEqual(lines[1], "전체 차트 파일 (순회 완료 후) · 저장 성공 기록 없음")
        self.assertEqual(view["update_path"], "C:/charts_updates")
        later = connection_view({**self.base_status(), "last_update_at": NOW,
                                 "last_export_at": NOW + timedelta(minutes=10)})
        self.assertNotIn("저장 성공 기록 없음", later["export"])
        self.assertNotEqual(later["export"].splitlines()[0].split("저장 성공")[1],
                            later["export"].splitlines()[1].split("저장 성공")[1])

    def test_readonly_buttons_only_emit_requests_and_updates_are_cached(self):
        widget = SignalConnectionPanel()
        status = self.base_status()
        self.assertTrue(widget.set_status(status))
        self.assertFalse(widget.set_status(status))
        self.assertTrue(widget.input_path.isReadOnly())
        self.assertTrue(widget.output_path.isReadOnly())
        self.assertTrue(widget.update_path.isReadOnly())
        self.assertEqual(widget.input_path.text(), status["input_path"])
        widget.set_status({**status, "update_path": "C:/charts_updates"})
        self.assertEqual(widget.update_path.text(), "C:/charts_updates")
        for button, signal in ((widget.settings_button, widget.request_settings),
                               (widget.inspect_button, widget.request_inspect),
                               (widget.folder_button, widget.request_folder)):
            spy = QSignalSpy(signal)
            button.click()
            self.assertEqual(spy.count(), 1)
        widget.set_status({**status, "inspection_busy": True})
        self.assertFalse(widget.inspect_button.isEnabled())
        widget.close()


if __name__ == "__main__":
    unittest.main()
