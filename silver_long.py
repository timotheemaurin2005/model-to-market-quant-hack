import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timezone

SYMBOL = "XAGUSD"
DIRECTION = -1                # Short
TOTAL_RISK_PCT = 0.04         # 4% equity risk
HARD_STOP_PRICE = 66.20       # Adjusted based on June 22 chart peak
MAGIC_SILVER_TEST = 999999    # Unique test magic number

# ==========================================
# 🛑 STRICT DRY RUN MODE
# True = Script calculates everything but WILL NOT place a live trade.
# ==========================================
DRY_RUN = False                

def check_combined_margin(equity):
    """Checks total account margin. Returns False if nearing 85% penalty."""
    acc = mt5.account_info()
    if not acc: return False
    
    if (acc.margin / equity) > 0.80:
        print("⚠️ GUARDRAIL REJECT: Combined margin > 80%.")
        return False
    return True

def get_test_volume(equity, price):
    """Calculates MT5 volume sizing based on raw dollar risk to the hard stop."""
    info = mt5.symbol_info(SYMBOL)
    if not info: return 0.0
    
    # We are testing "Tranche 1" (Half of the 4% risk = 2% risk)
    tranche_risk_usd = (equity * TOTAL_RISK_PCT) / 2.0
    risk_per_unit = abs(HARD_STOP_PRICE - price)
    
    if risk_per_unit == 0: return 0.0
    
    raw_volume = tranche_risk_usd / (risk_per_unit * (info.trade_tick_value / info.trade_tick_size))
    volume = round(raw_volume / info.volume_step) * info.volume_step
    return max(info.volume_min, min(volume, info.volume_max))

def run_silver_test():
    print("==================================================")
    print("🔬 INITIALIZING SILVER DRY-RUN TEST")
    print("==================================================")
    
    if not mt5.initialize():
        print("❌ MT5 Init Failed")
        return
        
    acc = mt5.account_info()
    if not acc: 
        print("❌ Failed to get account info")
        return
        
    equity = acc.equity
    
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick: 
        print(f"❌ Failed to get live tick for {SYMBOL}")
        return
        
    current_price = tick.bid
    print(f"📊 Live {SYMBOL} Price: {current_price}")
    
    if not check_combined_margin(equity): 
        return
        
    vol = get_test_volume(equity, current_price)
    
    print("\n--- TEST EXECUTION CALCULATION ---")
    print(f"Target Action:  SELL (Short)")
    print(f"Target Volume:  {vol} lots")
    print(f"Entry Price:    {current_price}")
    print(f"Hard Stop:      {HARD_STOP_PRICE}")
    print("----------------------------------")
    
    if DRY_RUN:
        print("\n✅ DRY RUN COMPLETE. No live order was placed.")
        print("The sizing logic and margin checks are working correctly.")
    else:
        print("\n⚠️ WARNING: DRY_RUN is False. Attempting to place live trade...")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": vol,
            "type": mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "sl": HARD_STOP_PRICE,
            "deviation": 20,
            "magic": MAGIC_SILVER_TEST,
            "comment": "Silver Test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Live Test Trade Executed Successfully! Ticket: {result.order}")
        else:
            print(f"❌ Order failed: {result.comment if result else mt5.last_error()}")
        
    mt5.shutdown()

if __name__ == "__main__":
    run_silver_test()