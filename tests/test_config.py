from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from dockdack import KiwoomConfig, Market, TradingMode


class EnvironmentConfigTests(unittest.TestCase):
    def test_loads_market_specific_demo_credentials(self) -> None:
        values = {
            "DOCKDACK_TRADING_MODE": "demo",
            "DOCKDACK_KIWOOM_DEMO_DOMESTIC_APP_KEY": "domestic-app",
            "DOCKDACK_KIWOOM_DEMO_DOMESTIC_SECRET_KEY": "domestic-secret",
            "DOCKDACK_KIWOOM_DEMO_US_APP_KEY": "us-app",
            "DOCKDACK_KIWOOM_DEMO_US_SECRET_KEY": "us-secret",
        }
        with patch.dict(os.environ, values, clear=True):
            domestic = KiwoomConfig.from_env(market=Market.DOMESTIC)
            us = KiwoomConfig.from_env(market=Market.US)

        self.assertEqual(domestic.app_key, "domestic-app")
        self.assertEqual(us.app_key, "us-app")
        self.assertIs(domestic.mode, TradingMode.DEMO)
        self.assertIs(us.mode, TradingMode.DEMO)

    def test_market_credentials_fall_back_to_shared_environment_key(self) -> None:
        values = {
            "DOCKDACK_TRADING_MODE": "real",
            "DOCKDACK_KIWOOM_REAL_APP_KEY": "shared-app",
            "DOCKDACK_KIWOOM_REAL_SECRET_KEY": "shared-secret",
        }
        with patch.dict(os.environ, values, clear=True):
            domestic = KiwoomConfig.from_env(market=Market.DOMESTIC)
            us = KiwoomConfig.from_env(market=Market.US)

        self.assertEqual(domestic.app_key, "shared-app")
        self.assertEqual(us.app_key, "shared-app")
        self.assertIs(domestic.mode, TradingMode.REAL)
        self.assertIs(us.mode, TradingMode.REAL)


if __name__ == "__main__":
    unittest.main()
