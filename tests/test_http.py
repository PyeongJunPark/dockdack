from __future__ import annotations

import gc
import threading
import unittest
import weakref
from contextvars import Context
from dataclasses import dataclass, field
from typing import Any

from dockdack.config import KiwoomConfig
from dockdack.exceptions import BrokerAPIError
from dockdack.http import KiwoomHTTPClient, order_send_guard
from dockdack.models import TradingMode


@dataclass
class Response:
    body: Any
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Transport:
    def __init__(self, clock, *responses):
        self.clock = clock
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, headers, json, timeout):
        self.calls.append({"time": self.clock(), "method": method, "url": url,
                           "headers": dict(headers), "body": dict(json), "timeout": timeout})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def token(value="fake-token"):
    return Response({"return_code": 0, "token": value, "expires_dt": "20991231235959"})


def limited(code=1700, status=200):
    return Response({"return_code": code,
                     "return_msg": "허용된 요청 개수를 초과하였습니다[1700:허용된 API 요청 개수를 초과하였습니다. 유량=1, API ID=usa20100]"}, status)


class ReadRetryTests(unittest.TestCase):
    def client(self, *responses, interval=0):
        clock = Clock()
        transport = Transport(clock, token(), *responses)
        config = KiwoomConfig("fake-app", "fake-secret", min_request_interval_seconds=interval)
        client = KiwoomHTTPClient(config, transport=transport, monotonic=clock, sleeper=clock.sleep)
        return client, transport, clock

    def test_known_read_retries_explicit_1700_and_preserves_page_payload(self):
        for error in (limited(), limited("1700"), limited(5), limited(1700, 429)):
            with self.subTest(error=error):
                client, transport, clock = self.client(
                    error, Response({"return_code": 0, "data": [1]},
                                    headers={"Cont-Yn": "Y", "Next-Key": "next-result"}))
                page = client.request(api_id="usa20100", path="/api/us/mrkcond",
                                      body={"stex_tp": "ND", "stk_cd": "AAPL"},
                                      cont_yn="Y", next_key="incoming-page")
                self.assertEqual(clock.sleeps, [2.5])
                self.assertEqual(len(transport.calls), 3)  # Token plus two reads.
                for key in ("headers", "body", "url", "method", "timeout"):
                    self.assertEqual(transport.calls[1][key], transport.calls[2][key])
                self.assertEqual(page.body["data"], [1])
                self.assertTrue(page.has_next)
                self.assertEqual(page.next_key, "next-result")

    def test_second_backoff_can_recover(self):
        client, transport, clock = self.client(limited(), limited(), Response({"return_code": 0}))
        client.request(api_id="usa06012", path="/api/us/chart")
        self.assertEqual(clock.sleeps, [2.5, 5.0])
        self.assertEqual(len(transport.calls), 4)

    def test_persistent_limit_stops_after_three_reads_with_original_error(self):
        client, transport, clock = self.client(limited(), limited(), limited(5), limited())
        with self.assertRaises(BrokerAPIError) as raised:
            client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(raised.exception.return_code, 5)
        self.assertEqual(raised.exception.status_code, 200)
        self.assertIn("1700:", str(raised.exception))
        self.assertEqual(clock.sleeps, [2.5, 5.0])
        self.assertLessEqual(sum(clock.sleeps), 7.5)
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(len(transport.responses), 1)

    def test_order_cancel_modify_and_unknown_operations_never_retry(self):
        endpoints = [(api_id, path)
                     for prefix, path in (("kt100", "/api/dostk/ordr"), ("ust200", "/api/us/ordr"))
                     for api_id in (prefix + suffix for suffix in ("00", "01", "02", "03"))]
        endpoints.extend([("future_read", "/api/dostk/stkinfo"),
                          ("ka10001", "/api/dostk/ordr"),
                          ("ust20000", "/api/us/acnt")])
        for api_id, path in endpoints:
            for error in (limited(), Response({"return_code": 1}, 401)):
                with self.subTest(api_id=api_id, path=path, status=error.status_code):
                    client, transport, clock = self.client(error, Response({"return_code": 0}))
                    with self.assertRaises(BrokerAPIError):
                        # Even the default retry_auth=True cannot retry a write.
                        client.request(api_id=api_id, path=path, body={"ord_qty": "1"})
                    self.assertEqual(len(transport.calls), 2)
                    self.assertEqual(clock.sleeps, [])

    def test_all_existing_reads_have_reviewed_retry_paths(self):
        endpoints = [
            ("ka10001", "/api/dostk/stkinfo"), ("ka10099", "/api/dostk/stkinfo"),
            ("ka10081", "/api/dostk/chart"), ("ka10032", "/api/dostk/rkinfo"),
            ("kt00018", "/api/dostk/acnt"), ("kt00001", "/api/dostk/acnt"),
            ("ka10075", "/api/dostk/acnt"), ("ka10076", "/api/dostk/acnt"),
            ("usa20100", "/api/us/mrkcond"), ("usa10098", "/api/us/stkinfo"),
            ("usa10099", "/api/us/stkinfo"), ("usa10104", "/api/us/stkinfo"),
            ("usa06012", "/api/us/chart"), ("usa20540", "/api/us/rkinfo"),
            ("ust21070", "/api/us/acnt"), ("ust21110", "/api/us/acnt"),
            ("ust21050", "/api/us/acnt"), ("ust21510", "/api/us/acnt"),
        ]
        for api_id, path in endpoints:
            with self.subTest(api_id=api_id):
                client, transport, _ = self.client(limited(), Response({"return_code": 0}))
                client.request(api_id=api_id, path=path)
                self.assertEqual(len(transport.calls), 3)

    def test_ambiguous_errors_do_not_retry(self):
        errors = [TimeoutError("read timed out"), ConnectionError("connection reset"),
                  Response(ValueError("not JSON")), Response([{"return_code": 1700}]),
                  Response({"return_code": 1700.5, "return_msg": "malformed code"}),
                  Response({"return_code": 5, "return_msg": "something 1700"}),
                  Response({"return_code": 5, "return_msg": "too many requests"}, 429),
                  limited(1700, 500), limited(1700, 302)]
        for error in errors:
            with self.subTest(error=error):
                client, transport, clock = self.client(error, Response({"return_code": 0}))
                with self.assertRaises(BrokerAPIError):
                    client.request(api_id="ka10001", path="/api/dostk/stkinfo")
                self.assertEqual(len(transport.calls), 2)
                self.assertEqual(clock.sleeps, [])

    def test_success_message_containing_1700_is_not_retried(self):
        client, transport, clock = self.client(limited(0))
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(clock.sleeps, [])

    def test_auth_refresh_for_known_read_preserves_retry_budget_and_payload(self):
        client, transport, clock = self.client(
            limited(), Response({"return_code": 1}, 401), token("new-token"),
            limited(), Response({"return_code": 0}))
        client.request(api_id="usa20100", path="/api/us/mrkcond",
                       body={"stk_cd": "AAPL"}, cont_yn="Y", next_key="page")
        reads = [call for call in transport.calls if "api-id" in call["headers"]]
        self.assertEqual(len(reads), 4)
        self.assertEqual(clock.sleeps, [2.5, 1.875, 1.875, 5.0])
        self.assertEqual(client.rate_status()["retry_count"], 2)
        for call in reads:
            self.assertEqual(call["body"], {"stk_cd": "AAPL"})
            self.assertEqual(call["headers"]["next-key"], "page")
            self.assertEqual(call["headers"]["cont-yn"], "Y")
        self.assertEqual(reads[-1]["headers"]["authorization"], "Bearer new-token")

    def test_auth_refresh_is_at_most_once_and_can_be_disabled(self):
        client, transport, _ = self.client(Response({"return_code": 1}, 401), token(),
                                           Response({"return_code": 1}, 401))
        with self.assertRaises(BrokerAPIError):
            client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(len(transport.calls), 4)
        client, transport, _ = self.client(Response({"return_code": 1}, 401))
        with self.assertRaises(BrokerAPIError):
            client.request(api_id="ka10001", path="/api/dostk/stkinfo", retry_auth=False)
        self.assertEqual(len(transport.calls), 2)

    def test_token_endpoint_is_never_rate_limit_retried(self):
        clock = Clock()
        transport = Transport(clock, limited(), token())
        client = KiwoomHTTPClient(KiwoomConfig("fake-app", "fake-secret"), transport=transport,
                                  monotonic=clock, sleeper=clock.sleep)
        with self.assertRaises(BrokerAPIError):
            client.get_access_token()
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(clock.sleeps, [])


class SharedThrottleTests(unittest.TestCase):
    def client(self, clock, *, app_key="fake-app", mode=TradingMode.DEMO, interval=1.05, transport=None):
        transport = transport or Transport(clock, token(), Response({"return_code": 0}))
        config = KiwoomConfig(app_key, "fake-secret", mode=mode,
                              min_request_interval_seconds=interval)
        return KiwoomHTTPClient(config, transport=transport, monotonic=clock, sleeper=clock.sleep)

    def test_same_credential_clients_share_request_spacing(self):
        clock = Clock()
        first, second = self.client(clock), self.client(clock)
        self.assertIs(first._request_gate, second._request_gate)
        first.request(api_id="ka10001", path="/api/dostk/stkinfo")
        second.request(api_id="usa20100", path="/api/us/mrkcond")
        times = [call["time"] for client in (first, second) for call in client.transport.calls]
        for actual, expected in zip(times, (0.0, 1.05, 2.1, 3.15)):
            self.assertAlmostEqual(actual, expected)

    def test_baseline_single_client_spacing_is_unchanged(self):
        clock = Clock()
        client = self.client(clock)
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(clock.sleeps, [1.05])

    def test_other_accounts_hosts_and_clocks_are_independent(self):
        clock = Clock()
        first = self.client(clock)
        for other in (self.client(clock, app_key="other-app"),
                      self.client(clock, mode=TradingMode.REAL), self.client(Clock())):
            self.assertIsNot(first._request_gate, other._request_gate)

    def test_shared_gate_is_released_after_last_client(self):
        clock = Clock()
        first, second = self.client(clock), self.client(clock)
        gate_ref = weakref.ref(first._request_gate)
        del first
        gc.collect()
        self.assertIsNotNone(gate_ref())
        del second
        gc.collect()
        self.assertIsNone(gate_ref())

    def test_shared_gate_does_not_shorten_previous_clients_interval(self):
        clock = Clock()
        first, second = self.client(clock, interval=1.05), self.client(clock, interval=0)
        first.get_access_token()
        second.get_access_token()
        self.assertEqual(clock.sleeps, [1.05])

    def test_overlapping_clients_cannot_send_in_parallel(self):
        clock = Clock()
        entered, release, second_started, second_entered = (threading.Event() for _ in range(4))

        class BlockingTransport(Transport):
            def request(self, *args, **kwargs):
                if threading.current_thread().name == "first-http":
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("test did not release first request")
                elif threading.current_thread().name == "second-http":
                    second_entered.set()
                return super().request(*args, **kwargs)

        transport = BlockingTransport(clock, token(), token(), Response({"return_code": 0}),
                                      Response({"return_code": 0}))
        first, second = self.client(clock, transport=transport), self.client(clock, transport=transport)
        first.get_access_token()
        second.get_access_token()
        errors = []

        def request(client, *, started=None):
            try:
                if started is not None:
                    started.set()
                client.request(api_id="ka10001", path="/api/dostk/stkinfo")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=request, args=(first,), name="first-http"),
                   threading.Thread(target=request, args=(second,),
                                    kwargs={"started": second_started}, name="second-http")]
        try:
            threads[0].start()
            self.assertTrue(entered.wait(1))
            threads[1].start()
            self.assertTrue(second_started.wait(1))
            self.assertFalse(second_entered.wait(0.05))
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(2)
        self.assertFalse(errors)
        self.assertTrue(second_entered.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        times = [call["time"] for call in transport.calls]
        self.assertTrue(all(later - earlier >= 1.049 for earlier, later in zip(times, times[1:])))


class CalculatedScheduleTests(unittest.TestCase):
    def client(self, clock, transport, *, mode=TradingMode.DEMO):
        return KiwoomHTTPClient(KiwoomConfig("fake-app", "fake-secret", mode=mode),
                                transport=transport, monotonic=clock, sleeper=clock.sleep)

    def test_default_spacing_is_from_response_completion_not_send_start(self):
        clock = Clock()

        class SlowTransport(Transport):
            def request(self, *args, **kwargs):
                result = super().request(*args, **kwargs)
                clock.now += 0.8
                return result

        transport = SlowTransport(clock, token(), *(Response({"return_code": 0}) for _ in range(3)))
        client = self.client(clock, transport)
        for _ in range(3):
            client.request(api_id="usa20100", path="/api/us/mrkcond")
        times = [call["time"] for call in transport.calls]
        for earlier, later in zip(times, times[1:]):
            self.assertAlmostEqual(later - earlier, 0.8 + 1.25)
        self.assertEqual(client.rate_status()["effective_interval_seconds"], 1.25)

    def test_real_default_also_respects_stricter_us_peak_rate(self):
        clock = Clock()
        transport = Transport(clock, token(), Response({"return_code": 0}))
        client = self.client(clock, transport, mode=TradingMode.REAL)
        client.request(api_id="usa20100", path="/api/us/mrkcond")
        self.assertEqual(clock.sleeps, [0.4])

    def test_catalog_pages_share_minute_limit_across_clients(self):
        clock = Clock()
        transport = Transport(clock, token(), token(), *(Response({"return_code": 0}) for _ in range(6)))
        first, second = self.client(clock, transport), self.client(clock, transport)
        first.get_access_token()
        second.get_access_token()
        for client in (first, second) * 3:
            client.request(api_id="usa10099", path="/api/us/stkinfo", cont_yn="Y", next_key="page")
        calls = [call for call in transport.calls if call["headers"].get("api-id") == "usa10099"]
        times = [call["time"] for call in calls]
        for earlier, later in zip(times, times[1:]):
            self.assertAlmostEqual(later - earlier, 12.2)
        self.assertGreater(times[5] - times[0], 60.0)
        self.assertEqual(first.rate_status()["catalog_interval_seconds"], 12.2)

    def test_slow_wakeup_and_long_idle_do_not_accumulate_burst_credits(self):
        class LateClock(Clock):
            def sleep(self, seconds):
                super().sleep(seconds)
                if len(self.sleeps) == 1:
                    self.now += 10.0

        clock = LateClock()
        transport = Transport(clock, token(), *(Response({"return_code": 0}) for _ in range(4)))
        client = self.client(clock, transport)
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        clock.now += 100.0
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        times = [call["time"] for call in transport.calls]
        self.assertAlmostEqual(times[2] - times[1], 1.25)
        self.assertAlmostEqual(times[4] - times[3], 1.25)

    def test_rate_limit_cooldown_is_shared_across_read_apis_and_clients(self):
        clock = Clock()
        transport = Transport(clock, token(), token(), limited(), limited(), limited(),
                              Response({"return_code": 0}))
        first, second = self.client(clock, transport), self.client(clock, transport)
        first.get_access_token()
        second.get_access_token()
        with self.assertRaises(BrokerAPIError):
            first.request(api_id="usa20100", path="/api/us/mrkcond")
        rejected_at = clock()
        status = second.rate_status()
        self.assertEqual(status["cooldown_remaining_seconds"], 10.0)
        self.assertEqual(status["rate_limit_count"], 3)
        self.assertEqual(status["retry_count"], 2)
        self.assertEqual(status["scope_id"], first.rate_status()["scope_id"])
        second.request(api_id="ka10081", path="/api/dostk/chart")
        self.assertAlmostEqual(transport.calls[-1]["time"] - rejected_at, 10.0)

    def test_persistent_limits_have_bounded_cooldown_and_retry_counts(self):
        clock = Clock()
        transport = Transport(clock, token(), *(limited() for _ in range(6)))
        client = self.client(clock, transport)
        for _ in range(2):
            with self.assertRaises(BrokerAPIError):
                client.request(api_id="usa20100", path="/api/us/mrkcond")
        status = client.rate_status()
        self.assertEqual(status["retry_count"], 4)
        self.assertEqual(status["rate_limit_count"], 6)
        self.assertEqual(status["cooldown_remaining_seconds"], 20.0)
        self.assertLessEqual(status["effective_interval_seconds"], 5.0)
        self.assertTrue(all(delay <= 20.0 for delay in clock.sleeps))
        self.assertEqual(len(transport.calls), 7)  # Token plus exactly six reads.

    def test_adaptive_spacing_recovers_after_quiet_minute(self):
        clock = Clock()
        transport = Transport(clock, token(), limited(), Response({"return_code": 0}),
                              Response({"return_code": 0}), Response({"return_code": 0}))
        client = self.client(clock, transport)
        client.request(api_id="usa20100", path="/api/us/mrkcond")
        self.assertEqual(client.rate_status()["effective_interval_seconds"], 1.875)
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertAlmostEqual(transport.calls[-1]["time"] - transport.calls[-2]["time"], 1.875)
        clock.now += 60.0
        status = client.rate_status()
        self.assertEqual(status["effective_interval_seconds"], 1.25)
        self.assertEqual(status["adaptive_remaining_seconds"], 0.0)
        self.assertEqual(status["cooldown_remaining_seconds"], 0.0)
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")

    def test_rate_status_remains_responsive_during_scheduled_sleep(self):
        entered, release, status_ready = (threading.Event() for _ in range(3))

        class BlockingClock(Clock):
            def sleep(self, seconds):
                entered.set()
                if not release.wait(2):
                    raise AssertionError("scheduled wait not released")
                super().sleep(seconds)

        clock = BlockingClock()
        transport = Transport(clock, token(), Response({"return_code": 0}))
        client = self.client(clock, transport)
        client.get_access_token()
        results, errors = [], []

        def request():
            try:
                client.request(api_id="usa20100", path="/api/us/mrkcond")
            except Exception as exc:
                errors.append(exc)

        def read_status():
            results.append(client.rate_status())
            status_ready.set()

        worker, reader = threading.Thread(target=request), threading.Thread(target=read_status)
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            reader.start()
            self.assertTrue(status_ready.wait(0.5), "GUI status blocked behind request sleep")
            self.assertEqual(results[0]["wait_remaining_seconds"], 1.25)
            self.assertFalse(results[0]["in_flight"])
            self.assertEqual(results[0]["active_api_id"], "usa20100")
        finally:
            release.set()
            worker.join(2)
            if reader.ident is not None:
                reader.join(2)
        self.assertFalse(errors)
        self.assertFalse(worker.is_alive())
        self.assertFalse(client.rate_status()["in_flight"])
        self.assertEqual(client.rate_status()["wait_remaining_seconds"], 0.0)

    def test_status_is_memory_only_and_response_json_is_not_parsed_twice(self):
        class CountingResponse(Response):
            calls = 0

            def json(self):
                self.calls += 1
                return super().json()

        clock = Clock()
        response = CountingResponse({"return_code": 0})
        transport = Transport(clock, token(), response)
        client = self.client(clock, transport)
        initial = client.rate_status()
        self.assertEqual(transport.calls, [])
        self.assertEqual(initial["request_count"], 0)
        client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(response.calls, 1)
        for _ in range(20):
            status = client.rate_status()
        self.assertEqual(status["request_count"], 2)
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn("fake-app", str(status))
        self.assertNotIn("fake-secret", str(status))


class OrderSendGuardTests(unittest.TestCase):
    class Denied(RuntimeError):
        pass

    def client(self, *responses):
        clock = Clock()
        transport = Transport(clock, token(), *responses)
        client = KiwoomHTTPClient(KiwoomConfig("guard-app", "fake-secret"), transport=transport,
                                  monotonic=clock, sleeper=clock.sleep)
        return client, transport, clock

    def submit(self, client, api_id="kt10000", path="/api/dostk/ordr"):
        return client.request(api_id=api_id, path=path, body={"ord_qty": "1"}, retry_auth=False)

    def test_denial_runs_after_wait_without_sending_or_incrementing_counter(self):
        client, transport, clock = self.client(Response({"return_code": 0}))
        checks = []

        def deny():
            checks.append(clock())
            raise self.Denied("OFF or stale after pacing")

        with order_send_guard(deny):
            with self.assertRaisesRegex(self.Denied, "OFF or stale"):
                self.submit(client)
        self.assertEqual(checks, [1.25])
        self.assertEqual(len(transport.calls), 1)  # Only initial authentication.
        self.assertEqual(len(transport.responses), 1)
        status = client.rate_status()
        self.assertEqual(status["request_count"], 1)
        self.assertEqual(status["rate_limit_count"], 0)
        self.assertEqual(status["retry_count"], 0)
        self.assertFalse(status["in_flight"])
        self.assertIsNone(status["active_api_id"])
        self.assertEqual(status["wait_remaining_seconds"], 0.0)

    def test_guard_covers_all_order_operations_and_unknown_api_fail_closed(self):
        for api_id, path in [(prefix + suffix, path)
                             for prefix, path in (("kt100", "/api/dostk/ordr"), ("ust200", "/api/us/ordr"))
                             for suffix in ("00", "01", "02", "03")] + [("future_api", "/api/us/acnt")]:
            with self.subTest(api_id=api_id):
                client, transport, _ = self.client(Response({"return_code": 0}))

                def deny():
                    raise self.Denied("not sent")

                with order_send_guard(deny), self.assertRaises(self.Denied):
                    self.submit(client, api_id, path)
                self.assertEqual(len(transport.calls), 1)

    def test_reads_and_auth_are_exempt_even_with_scoped_denial(self):
        client, transport, _ = self.client(Response({"return_code": 0}))

        def deny():
            raise self.Denied("not sent")

        with order_send_guard(deny):
            client.request(api_id="ka10001", path="/api/dostk/stkinfo")
        self.assertEqual(len(transport.calls), 2)

    def test_nested_guard_restores_outer_and_exception_exit_removes_guard(self):
        client, transport, _ = self.client(*(Response({"return_code": 0}) for _ in range(4)))
        checks = []
        with self.assertRaises(self.Denied):
            with order_send_guard(lambda: checks.append("outer")):
                self.submit(client)
                with order_send_guard(lambda: checks.append("inner")):
                    self.submit(client)
                self.submit(client)
                raise self.Denied("leave scope")
        self.submit(client)
        # Nested service permission checks must not mask AutoTrader's outer
        # OFF/freshness/session guard on the same paced transport send.
        self.assertEqual(checks, ["outer", "outer", "inner", "outer"])
        self.assertEqual(len(transport.calls), 5)

    def test_independent_context_and_thread_do_not_inherit_scoped_denial(self):
        client, transport, _ = self.client(*(Response({"return_code": 0}) for _ in range(3)))
        errors = []

        def deny():
            raise self.Denied("parent context")

        def threaded_submit():
            try:
                self.submit(client)
            except Exception as exc:
                errors.append(exc)

        with order_send_guard(deny):
            Context().run(self.submit, client)
            thread = threading.Thread(target=lambda: Context().run(threaded_submit))
            thread.start()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            with self.assertRaises(self.Denied):
                self.submit(client)
        self.submit(client)
        self.assertFalse(errors)
        self.assertEqual(len(transport.calls), 4)


if __name__ == "__main__":
    unittest.main()
