"""Desktop startup loads holdings without enabling trading."""
from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None

if HAS_QT:
    from PySide6.QtWidgets import QApplication

    from dockdack.models import AccountSnapshot, Market
    from dockdack.v00_app import V00Window, main
    from test_autotrade import FakeTradingService, position


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class StartupPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_fetches_holdings_once_per_market_without_starting_trading(self):
        with tempfile.TemporaryDirectory() as folder:
            service = FakeTradingService()
            service.storage_scope = "demo"
            account_calls = []

            def account(instrument):
                account_calls.append(instrument.market)
                holdings = (position(),) if instrument.market is Market.DOMESTIC else ()
                return AccountSnapshot(instrument.market, instrument.currency, holdings,
                                       available_to_order=Decimal("10000"))

            service.safety_account = account
            observed = []

            def run_events(app):
                window = next(widget for widget in app.topLevelWidgets() if isinstance(widget, V00Window))
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    app.processEvents()
                    if (window.worker is None and window._workspace_worker is None
                            and window._activity_worker is None):
                        break
                    time.sleep(.01)
                else:
                    self.fail("Startup portfolio refresh did not complete")
                app.processEvents()
                observed.append((window.portfolio_panel.tables[Market.DOMESTIC].rowCount(),
                                 window.portfolio_panel.tables[Market.US].rowCount(),
                                 window.monitoring, window.engine.orders_enabled))
                window.close()
                app.processEvents()
                return 0

            with patch("dockdack.ui.v00_app.TradingService", return_value=service), \
                    patch("dockdack.runtime_paths.app_home", return_value=Path(folder)), \
                    patch("dockdack.ui.v00_app.set_windows_app_id"), \
                    patch("dockdack.gui_service.KiwoomConfig.from_env",
                          side_effect=AssertionError("No real credentials in startup test")), \
                    patch.object(QApplication, "exec", run_events):
                result = main(["--store", str(Path(folder) / "watch.sqlite3"), "--no-model"])

            self.assertEqual(result, 0)
            self.assertEqual(account_calls, [Market.DOMESTIC, Market.US])
            self.assertEqual(observed, [(1, 0, False, False)])
            self.assertEqual(service.quote_calls, 0)
            self.assertEqual(service.history_calls, 0)
            self.assertEqual(service.submitted, [])


if __name__ == "__main__":
    unittest.main()
