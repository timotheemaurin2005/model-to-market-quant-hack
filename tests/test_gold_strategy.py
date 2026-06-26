import unittest
from datetime import datetime, timezone
import os
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# MOCK METATRADER5 FOR MACOS RUN
# ---------------------------------------------------------------------------
mock_mt5 = MagicMock()
mock_mt5.initialize.return_value = True
mock_mt5.TIMEFRAME_M15 = 15
mock_mt5.ORDER_TYPE_BUY = 0
mock_mt5.ORDER_TYPE_SELL = 1
mock_mt5.POSITION_TYPE_BUY = 0
mock_mt5.POSITION_TYPE_SELL = 1
mock_mt5.TRADE_ACTION_DEAL = 1
mock_mt5.ORDER_TIME_GTC = 0
mock_mt5.ORDER_FILLING_FOK = 1
mock_mt5.ORDER_FILLING_IOC = 2
mock_mt5.ORDER_FILLING_RETURN = 0

# Mock values returned by terminal_info / account_info / symbol_info
mock_terminal = MagicMock()
mock_terminal.trade_allowed = True
mock_mt5.terminal_info.return_value = mock_terminal

mock_account = MagicMock()
mock_account.login = 10301
mock_account.equity = 1000000.0
mock_account.leverage = 30
mock_account.margin = 0.0
mock_mt5.account_info.return_value = mock_account

# Set it in sys.modules before importing directional_sleeve
sys.modules['MetaTrader5'] = mock_mt5

# Add sleeves/ and trading_engine/ to path to import directional_sleeve + mt5_executor
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(_ROOT, "sleeves"))
sys.path.append(os.path.join(_ROOT, "trading_engine"))

# Set mock password so mt5_executor doesn't throw on import
os.environ["MT5_PASSWORD"] = "mock_pass"

import directional_sleeve as ds

class TestGoldStrategy(unittest.TestCase):
    def setUp(self):
        # Clear state file if exists
        if os.path.exists(ds.STATE_FILE):
            os.remove(ds.STATE_FILE)

    def tearDown(self):
        # Clean up state file
        if os.path.exists(ds.STATE_FILE):
            os.remove(ds.STATE_FILE)

    def test_state_load_save(self):
        # Test initial state
        state = ds.load_gold_state()
        self.assertFalse(state["halted"])
        self.assertEqual(len(state["tranches"]), 0)

        # Modify and save
        state["halted"] = True
        state["tranches"].append({"price": 2350.0, "lots": 1.5, "time": "2026-06-23T00:00:00"})
        ds.save_gold_state(state)

        # Load again and verify
        loaded = ds.load_gold_state()
        self.assertTrue(loaded["halted"])
        self.assertEqual(len(loaded["tranches"]), 1)
        self.assertEqual(loaded["tranches"][0]["price"], 2350.0)

    def test_news_blackout_active(self):
        # Test outside window
        t_outside = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
        active, news_type = ds.is_news_blackout_active(t_outside)
        self.assertFalse(active)

        # Test GDP window (June 24, 00:00 - 14:00 UTC)
        t_gdp_inside = datetime(2026, 6, 24, 8, 30, tzinfo=timezone.utc)
        active, news_type = ds.is_news_blackout_active(t_gdp_inside)
        self.assertTrue(active)
        self.assertEqual(news_type, "GDP")

        # Test PCE window (June 26, 00:00 - 14:00 UTC)
        t_pce_inside = datetime(2026, 6, 26, 12, 15, tzinfo=timezone.utc)
        active, news_type = ds.is_news_blackout_active(t_pce_inside)
        self.assertTrue(active)
        self.assertEqual(news_type, "PCE")

if __name__ == "__main__":
    unittest.main()
