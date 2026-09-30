"""The hedge model connection is visible but never an executable order path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dockdack.minute_hedge_research import Variant
from dockdack.models import TradingMode
from dockdack.research.hedge_connection import load_hedge_connections


def report():
    return {
        "schema": "dockdack.minute_hedge_research.v1",
        "market": "domestic",
        "stock_symbol": "005930",
        "benchmark_index_code": "201",
        "hedge_leg": "long_inverse_etf",
        "inverse_etf_symbol": "114800",
        "actual_broker_trades": False,
        "inputs": {
            key: {"source": "kiwoom_rest_demo_minute_chart", "symbol": symbol, "sha256": "a" * 64}
            for key, symbol in (("stock", "005930"), ("index", "201"), ("etf", "114800"))
        },
        "candidates": [
            {"variant": variant.value, "data_gate": "abstain", "deployment_allowed": False,
             "pass_means_profitable": False, "out_of_sample_trades": 0,
             "out_of_sample_net_pnl": None, "reason": "no_qualified_threshold"}
            for variant in Variant
        ],
    }


class HedgeConnectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "hedge.json"

    def save(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return self.path

    def assert_all_abstain(self, statuses):
        self.assertEqual(len(statuses), len(Variant))
        self.assertEqual({status.variant for status in statuses}, set(Variant))
        self.assertTrue(all(status.state == "abstain" and not status.order_eligible
                            and status.order_intents == () for status in statuses))

    def test_valid_demo_report_exposes_all_seven_but_no_order_signal(self):
        statuses = load_hedge_connections(self.save(report()), mode=TradingMode.DEMO)
        self.assert_all_abstain(statuses)
        self.assertTrue(all(status.research_gate == "abstain" for status in statuses))
        self.assertEqual(statuses[0].stock_symbol, "005930")
        self.assertEqual(statuses[0].inverse_etf_symbol, "114800")
        self.assertTrue(all(status.reason == "historical_research_only_no_qualified_threshold"
                            for status in statuses))
        with self.assertRaises((AttributeError, FrozenInstanceError)):
            statuses[0].order_eligible = True
        with self.assertRaises((AttributeError, FrozenInstanceError)):
            statuses[0].order_intents = ({"side": "buy"},)

    def test_paper_trade_data_gate_pass_is_not_deployment_permission(self):
        payload = report()
        row = payload["candidates"][-1]
        row["data_gate"] = "pass"
        row["out_of_sample_trades"] = 1
        row["out_of_sample_net_pnl"] = -5020.42
        statuses = load_hedge_connections(self.save(payload), mode=TradingMode.DEMO)
        self.assert_all_abstain(statuses)
        self.assertEqual(statuses[-1].research_gate, "pass")
        self.assertFalse(statuses[-1].order_eligible)

    def test_real_mode_never_reads_report(self):
        statuses = load_hedge_connections(self.path, mode=TradingMode.REAL)
        self.assert_all_abstain(statuses)
        self.assertTrue(all(status.reason == "demo_only" for status in statuses))

    def test_missing_report_fails_closed(self):
        statuses = load_hedge_connections(None, mode=TradingMode.DEMO)
        self.assert_all_abstain(statuses)
        self.assertTrue(all(status.reason == "report_missing" for status in statuses))
        self.assert_all_abstain(load_hedge_connections(self.path, mode=TradingMode.DEMO))

    def test_synthetic_short_and_wrong_etf_are_not_executable_connections(self):
        for mutation in ({"hedge_leg": "synthetic_index_short"}, {"inverse_etf_symbol": "252670"}):
            with self.subTest(mutation=mutation):
                payload = report() | mutation
                statuses = load_hedge_connections(self.save(payload), mode=TradingMode.DEMO)
                self.assert_all_abstain(statuses)
                self.assertTrue(all(status.reason == "report_unverified" for status in statuses))

    def test_missing_or_real_source_rejects_whole_report(self):
        payload = report()
        payload["inputs"]["index"]["source"] = "kiwoom_rest_real_minute_chart"
        statuses = load_hedge_connections(self.save(payload), mode=TradingMode.DEMO)
        self.assert_all_abstain(statuses)
        self.assertTrue(all(status.reason == "report_unverified" for status in statuses))

    def test_deployment_flag_missing_variant_or_duplicate_variant_rejected(self):
        for edit in ("deployment", "missing", "duplicate"):
            with self.subTest(edit=edit):
                payload = deepcopy(report())
                if edit == "deployment":
                    payload["candidates"][0]["deployment_allowed"] = True
                elif edit == "missing":
                    payload["candidates"].pop()
                else:
                    payload["candidates"][0]["variant"] = payload["candidates"][1]["variant"]
                statuses = load_hedge_connections(self.save(payload), mode=TradingMode.DEMO)
                self.assert_all_abstain(statuses)
                self.assertTrue(all(status.reason == "report_unverified" for status in statuses))

    def test_duplicate_json_field_and_nonfinite_number_rejected(self):
        raw = json.dumps(report())
        self.path.write_text(raw.replace('"market": "domestic",', '"market": "domestic", "market": "domestic",'),
                             encoding="utf-8")
        self.assertTrue(all(s.reason == "report_unverified" for s in
                            load_hedge_connections(self.path, mode=TradingMode.DEMO)))
        self.path.write_text(raw.replace('"out_of_sample_trades": 0', '"out_of_sample_trades": NaN'),
                             encoding="utf-8")
        self.assertTrue(all(s.reason == "report_unverified" for s in
                            load_hedge_connections(self.path, mode=TradingMode.DEMO)))

    def test_mode_is_explicit(self):
        with self.assertRaises(TypeError):
            load_hedge_connections(None, mode="demo")


if __name__ == "__main__":
    unittest.main()
