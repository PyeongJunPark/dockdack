from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from contextlib import redirect_stderr
from decimal import Decimal
from unittest.mock import patch

from dockdack import BrokerAPIError, KiwoomBroker, Market, OrderOutcomeUnknown, TradingMode
from dockdack.cli import Terminal, create_broker, identify_symbol, main, parser
from test_kiwoom import FakeResponse, QueueTransport, config, token_response


class TerminalTests(unittest.TestCase):
    def terminal(self, *responses, mode=TradingMode.DEMO, enabled=False, inputs=()):
        transport = QueueTransport(*responses)
        broker = KiwoomBroker(config(mode, allow_live_orders=enabled), transport=transport)
        output = []
        reader = iter(inputs)

        def read(_prompt):
            try:
                return next(reader)
            except StopIteration:
                raise EOFError from None

        terminal = Terminal(mode.value, broker_factory=lambda *_: broker, read=read, write=output.append)
        return terminal, transport, output

    def test_all_eight_order_payloads(self):
        for market in ("domestic", "us"):
            for side in ("buy", "sell"):
                for kind in ("limit", "market"):
                    with self.subTest(market=market, side=side, kind=kind):
                        terminal, transport, output = self.terminal(
                            token_response(), FakeResponse({"return_code": 0, "ord_no": "123"}), inputs=["y"]
                        )
                        symbol = "005930" if market == "domestic" else "AAPL"
                        exchange = "KRX" if market == "domestic" else "NASDAQ"
                        price = "70000" if market == "domestic" else "200.25"
                        argv = [side, symbol, "2", "--type", kind, "--exchange", exchange]
                        if kind == "limit":
                            argv += ["--price", price]
                        self.assertEqual(terminal.execute(parser().parse_args(argv)), 0)
                        call = transport.calls[-1]
                        prefix = "kt1000" if market == "domestic" else "ust2000"
                        self.assertEqual(call["headers"]["api-id"], prefix + ("0" if side == "buy" else "1"))
                        expected = {
                            "stk_cd": symbol, "ord_qty": "2", "ord_uv": price if kind == "limit" else "",
                            "trde_tp": ("0" if kind == "limit" else "3") if market == "domestic"
                                       else ("00" if kind == "limit" else "03"),
                        }
                        if market == "domestic":
                            expected.update(dmst_stex_tp="KRX", cond_uv="")
                        else:
                            expected["stex_tp"] = "ND"
                            if side == "sell":
                                expected["stop_pric"] = ""
                        self.assertEqual(call["json"], expected)
                        self.assertEqual(len(transport.calls), 2)
                        self.assertIn("접수는 체결 완료가 아닙니다", output[-1])

    def test_domestic_quote_preserves_leading_zeros(self):
        terminal, transport, output = self.terminal(token_response(), FakeResponse({
            "return_code": 0, "stk_cd": "005930", "stk_nm": "삼성전자", "cur_prc": "-70000",
        }))
        terminal.execute(parser().parse_args(["quote", "005930"]))
        self.assertEqual(transport.calls[-1]["json"], {"stk_cd": "005930"})
        self.assertIn("현재가: 70,000 KRW", output)

    def test_current_price_cli_orders_use_displayed_limit(self):
        for side in ("buy", "sell"):
            terminal, transport, output = self.terminal(
                token_response(), FakeResponse({"cur_prc": "329.4900"}),
                FakeResponse({"return_code": 0, "ord_no": "123"}), inputs=["y"],
            )
            terminal.execute(parser().parse_args([side, "AAPL", "1", "--type", "current", "--exchange", "NASDAQ"]))
            self.assertEqual(transport.calls[-1]["json"]["ord_uv"], "329.49")
            self.assertEqual(transport.calls[-1]["json"]["trde_tp"], "00")
            self.assertTrue(any("지정가 329.49 USD" in line for line in output))

    def test_current_dry_run_queries_but_does_not_order(self):
        terminal, transport, _ = self.terminal(token_response(), FakeResponse({"cur_prc": "251250"}))
        terminal.execute(parser().parse_args(["buy", "005930", "1", "--type", "current", "--dry-run"]))
        self.assertEqual(len(transport.calls), 2)

    def test_current_shell_buy_and_sell_without_price_input(self):
        for action in ("9", "10"):
            terminal, transport, _ = self.terminal(
                token_response(), FakeResponse({"cur_prc": "251250"}),
                FakeResponse({"return_code": 0, "ord_no": "123"}),
                inputs=["005930", action, "1", "y", "q"],
            )
            self.assertEqual(terminal.shell(), 0)
            self.assertEqual(transport.calls[-1]["json"]["ord_uv"], "251250")

    def test_current_with_explicit_price_is_rejected(self):
        terminal, transport, _ = self.terminal()
        with self.assertRaises(ValueError):
            terminal.execute(parser().parse_args(["buy", "AAPL", "1", "--type", "current", "--price", "100"]))
        self.assertEqual(transport.calls, [])

    def test_us_ticker_resolves_exchange_once_per_session(self):
        terminal, transport, output = self.terminal(
            token_response(), FakeResponse({"return_code": 0, "list": [
                {"stk_cd": "AAPL", "stex_tp": "ND"}, {"stk_cd": "AAP", "stex_tp": "NY"},
            ]}),
            FakeResponse({"return_code": 0, "cur_prc": "+200.2500"}),
            FakeResponse({"return_code": 0, "cur_prc": "+201.2500"}),
        )
        for _ in range(2):
            terminal.execute(parser().parse_args(["quote", "aapl"]))
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "usa10098")
        self.assertEqual(transport.calls[1]["json"], {"stk_cd": "AAPL"})
        self.assertEqual(transport.calls[-1]["json"], {"stk_cd": "AAPL", "stex_tp": "ND"})
        self.assertEqual(len(transport.calls), 4)
        self.assertIn("현재가: 200.2500 USD", output)

    def test_unknown_or_ambiguous_ticker_does_not_place_order(self):
        for rows in ([], [{"stk_cd": "AAP", "stex_tp": "ND"}], [
            {"stk_cd": "AAPL", "stex_tp": "ND"}, {"stk_cd": "AAPL", "stex_tp": "NY"},
        ]):
            with self.subTest(rows=rows):
                terminal, transport, _ = self.terminal(token_response(), FakeResponse({"list": rows}))
                with self.assertRaisesRegex(ValueError, "거래소"):
                    terminal.execute(parser().parse_args(["buy", "AAPL", "1", "--type", "market"]))
                self.assertEqual(len(transport.calls), 2)

    def test_bad_orders_fail_before_network_or_confirmation(self):
        cases = [
            ["buy", "005930", "0", "--type", "market"],
            ["buy", "005930", "-1", "--type", "market"],
            ["buy", "005930", "1", "--type", "market", "--price", "70000"],
            ["buy", "005930", "1", "--type", "limit"],
            ["buy", "005930", "1", "--type", "limit", "--price", "0.5"],
            ["buy", "005930", "1", "--type", "market", "--exchange", "NXT"],
            ["buy", "005930", "1", "--type", "market", "--exchange", "NASDAQ"],
        ]
        for price in ("NaN", "sNaN", "Infinity", "-1", "0", "oops", "1e1000000", "1e-1000000"):
            cases.append(["buy", "AAPL", "1", "--type", "limit", "--price", price])
        for argv in cases:
            with self.subTest(argv=argv):
                terminal, transport, _ = self.terminal()
                with self.assertRaises(ValueError):
                    terminal.execute(parser().parse_args(argv))
                self.assertEqual(transport.calls, [])

    def test_dry_run_and_declined_confirmation_do_not_submit(self):
        for extra, inputs in ((["--dry-run"], []), ([], [""]), ([], ["no"])):
            terminal, transport, _ = self.terminal(inputs=inputs)
            terminal.execute(parser().parse_args(["buy", "005930", "1", "--type", "market"] + extra))
            self.assertEqual(transport.calls, [])

    def test_live_orders_require_configuration_and_exact_confirmation(self):
        argv = parser().parse_args(["buy", "005930", "1", "--type", "market"])
        terminal, transport, _ = self.terminal(mode=TradingMode.REAL, inputs=["LIVE_ORDER"])
        with self.assertRaisesRegex(ValueError, "DOCKDACK_ALLOW_LIVE_ORDERS"):
            terminal.execute(argv)
        self.assertEqual(transport.calls, [])
        terminal, transport, _ = self.terminal(mode=TradingMode.REAL, enabled=True, inputs=["y"])
        terminal.execute(argv)
        self.assertEqual(transport.calls, [])
        terminal, transport, _ = self.terminal(
            token_response(), FakeResponse({"return_code": 0, "ord_no": "123"}),
            mode=TradingMode.REAL, enabled=True, inputs=["LIVE_ORDER"],
        )
        terminal.execute(argv)
        self.assertEqual(transport.calls[-1]["url"], "https://api.kiwoom.com/api/dostk/ordr")

    def test_incomplete_acknowledgement_does_not_print_acceptance_or_retry(self):
        terminal, transport, output = self.terminal(token_response(), FakeResponse({}), inputs=["y"])
        with self.assertRaises(OrderOutcomeUnknown):
            terminal.execute(parser().parse_args(["buy", "005930", "1", "--type", "market"]))
        self.assertEqual(len(transport.calls), 2)
        self.assertFalse(any("주문 접수:" in line for line in output))
        self.assertIn("재주문 전에", output[-1])

    def test_order_auth_error_is_not_retried(self):
        terminal, transport, output = self.terminal(
            token_response(), FakeResponse({"return_code": 1, "return_msg": "expired"}, status_code=401),
            inputs=["y"],
        )
        with self.assertRaises(BrokerAPIError):
            terminal.execute(parser().parse_args(["sell", "005930", "1", "--type", "market"]))
        self.assertEqual(len(transport.calls), 2)
        self.assertIn("재주문 전에", output[-1])

    def test_order_timeout_is_not_retried(self):
        terminal, transport, output = self.terminal(token_response(), inputs=["y"])
        transport_request = transport.request

        def request(method, url, **kwargs):
            if url.endswith("/ordr"):
                transport.calls.append({"url": url})
                raise TimeoutError("timeout")
            return transport_request(method, url, **kwargs)

        transport.request = request
        with self.assertRaises(BrokerAPIError):
            terminal.execute(parser().parse_args(["buy", "005930", "1", "--type", "market"]))
        self.assertEqual(len(transport.calls), 2)
        self.assertIn("재주문 전에", output[-1])

    def test_shell_recovers_from_bad_input_and_changes_symbol(self):
        terminal, transport, output = self.terminal(inputs=[
            "?", "005930", "2", "oops", "70000", "4", "1", "", "8", "AAPL", "q",
        ])
        self.assertEqual(terminal.shell(), 0)
        self.assertEqual(transport.calls, [])
        self.assertTrue(any("오류:" in line for line in output))
        self.assertTrue(any("전송하지 않았습니다" in line for line in output))
        self.assertTrue(any("AAPL |" in line for line in output))

    def test_shell_eof_exits_without_submitting(self):
        terminal, transport, _ = self.terminal(inputs=["005930", "4", "1"])
        self.assertEqual(terminal.shell(), 0)
        self.assertEqual(transport.calls, [])

    def test_empty_orders_does_not_claim_filled(self):
        terminal, _, output = self.terminal(token_response(), FakeResponse({"return_code": 0, "oso": []}))
        terminal.execute(parser().parse_args(["orders", "005930"]))
        self.assertIn("체결내역에서 확인", output[-1])

    def test_balance_uses_selected_market(self):
        terminal, transport, output = self.terminal(
            token_response(), FakeResponse({"return_code": 0, "acnt_evlt_remn_indv_tot": []}),
            FakeResponse({"return_code": 0, "entr": "100000", "ord_alow_amt": "90000"}),
        )
        terminal.execute(parser().parse_args(["balance", "005930"]))
        self.assertEqual(transport.calls[1]["headers"]["api-id"], "kt00018")
        self.assertIn("예수금: 100000 / 주문가능금액: 90000", output)

    def test_symbol_validation(self):
        for value, expected in (("005930", (Market.DOMESTIC, "005930")),
                                ("A005930", (Market.DOMESTIC, "005930")),
                                ("00593K", (Market.DOMESTIC, "00593K")),
                                (" aapl ", (Market.US, "AAPL")), ("brk.b", (Market.US, "BRK.B"))):
            self.assertEqual(identify_symbol(value), expected)
        for value in ("5930", "", "삼성전자", "12 345", "../../../"):
            with self.assertRaises(ValueError):
                identify_symbol(value)

    def test_cli_defaults_demo_and_reports_invalid_args(self):
        self.assertEqual(parser().parse_args(["quote", "005930"]).mode, "demo")
        with redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main(["buy", "005930", "1.5", "--type", "market"]), 1)
        self.assertIn("오류:", output.getvalue())

    def test_credentials_are_loaded_only_for_selected_market(self):
        with patch("dockdack.cli.KiwoomConfig.from_env", return_value=config()) as factory:
            create_broker(TradingMode.DEMO, Market.US)
        factory.assert_called_once_with(TradingMode.DEMO, market=Market.US)

    @unittest.skipUnless(sys.platform == "win32", "Windows output encoding")
    def test_windows_shell_handles_redirected_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "dockdack"], input=b"q\n", capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp949"}, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("모의", result.stdout.decode("utf-8"))


class BrokerValidationTests(unittest.TestCase):
    def test_direct_order_request_is_validated_before_http(self):
        transport = QueueTransport()
        broker = KiwoomBroker(config(), transport=transport)
        request = broker.build_order(market="domestic", side="buy", symbol="005930",
                                     exchange="KRX", quantity=1)
        with self.assertRaises(ValueError):
            broker.place_order(replace(request, quantity=-1))
        self.assertEqual(transport.calls, [])

    def test_invalid_price_never_falls_back_to_market_order(self):
        broker = KiwoomBroker(config(), transport=QueueTransport())
        for value in ("", "oops", "NaN", "sNaN", "Infinity", "-1", "1e99999"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                broker.build_order(market="domestic", side="buy", symbol="005930",
                                   exchange="KRX", quantity=1, price=value)

    def test_noninteger_quantity_is_rejected(self):
        broker = KiwoomBroker(config(), transport=QueueTransport())
        for value in (True, 1.5, Decimal("1.5"), 0, -1, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                broker.build_order(market="us", side="buy", symbol="AAPL", exchange="NASDAQ", quantity=value)


if __name__ == "__main__":
    unittest.main()
