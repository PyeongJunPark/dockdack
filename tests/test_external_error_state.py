from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dockdack.autotrade import AutoTrader
from dockdack.watchlist import WatchStore
from test_autotrade import FakeTradingService, NOW


class ExternalErrorStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = WatchStore(Path(self.temp.name) / "watch.sqlite3")
        self.service = FakeTradingService()
        self.engine = AutoTrader(self.service, self.store, clock=lambda: NOW)
        self.engine.external_only = True

    def test_error_count_survives_recovery_without_rearming_or_stale_message(self):
        def broken():
            raise ValueError("invalid signal JSON")

        self.assertEqual((self.engine.external_error, self.engine.external_error_count), ("", 0))
        self.engine.external_reader = broken
        # Simulate an already armed session, without broker access or submissions.
        self.engine._armed.set()
        self.engine._read_external()
        self.assertFalse(self.engine.orders_enabled)
        self.assertEqual((self.engine.external_error, self.engine.external_error_count), ("invalid signal JSON", 1))
        self.assertIn("external:file", self.engine._messages)
        self.engine._read_external()
        self.assertEqual(self.engine.external_error_count, 2)
        self.assertEqual(len(self.store.events(category="system")), 1)

        self.engine.external_reader = lambda: None
        self.engine._read_external()
        self.assertEqual((self.engine.external_error, self.engine.external_error_count), ("", 2))
        self.assertNotIn("external:file", self.engine._messages)
        self.assertFalse(self.engine.orders_enabled)

        self.engine.external_reader = broken
        self.engine._read_external()
        self.assertEqual(self.engine.external_error_count, 3)
        self.assertEqual(len(self.store.events(category="system")), 2)
        self.assertEqual(self.service.submitted, [])
        self.assertEqual(self.store.order_history(), ())

    def test_empty_error_stays_detectable_and_absent_reader_is_not_recovery(self):
        def broken():
            raise TimeoutError()

        self.engine.external_reader = broken
        self.engine._read_external()
        self.assertEqual(self.engine.external_error, "TimeoutError")
        self.engine.external_reader = None
        self.engine._read_external()
        self.assertEqual((self.engine.external_error, self.engine.external_error_count), ("TimeoutError", 1))
        self.engine.external_reader = lambda: {"hold": 1}
        self.engine._read_external()
        self.assertEqual((self.engine.external_error, self.engine.external_error_count), ("", 1))
        self.assertEqual(self.service.submitted, [])


if __name__ == "__main__":
    unittest.main()
