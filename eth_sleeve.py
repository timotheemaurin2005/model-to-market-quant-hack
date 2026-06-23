import MetaTrader5 as mt5
import pandas as pd
import time
import os
import json
from datetime import datetime, timezone

SYMBOL = "ETHUSD"
DIRECTION = -1                # Short
MAX_TRANCHES = 2              # 2 Tranches: Half now, half after macro event
TOTAL_RISK_PCT = 0.02         # 2% equity risk (Tactical satellite shot)
HARD_STOP_PRICE = 1515.0      # Hard stop above the $1500 breakdown reclaim level
KILL_SWITCH_USD = -20000.0    # Capped kill switch for this specific 2% allocation
MAGIC_ETH = 20260706          # Unique magic number for tracking
STATE_FILE = "eth_state.json"
MACRO_EVENT_TIME_UTC = datetime(2026, 6, 23, 13, 45, tzinfo=timezone.utc)

def load_state():
    """Loads previous execution state to manage tranches safely."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"cumulative_pnl": 0.0, "tranches_open": 0, "last_entry_price": 0.0}

def save_state(state):
    """Persists current state to disk."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def check_combined_margin(equity):
    """
    Checks total account margin across all running scripts.
    Returns False if account is nearing the 85% competition penalty line.
    """
    acc = mt5.account_info()
    if not acc: return False
    
    # 80% limit gives a 5% buffer before the 85% competition penalty
    if (acc.margin / equity) > 0.80:
        print("⚠️ GUARDRAIL REJECT: Combined margin > 80%.")
        return False
    return True

def get_tranche_volume(equity, price):
    """Calculates MT5 volume sizing based on raw dollar risk to the hard stop."""
    info = mt5.symbol_info(SYMBOL)
    if not info: return 0.0
    
    # Divide total 2% risk by the number of tranches (1% per tranche)
    tranche_risk_usd = (equity * TOTAL_RISK_PCT) / MAX_TRANCHES
    risk_per_unit = abs(HARD_STOP_PRICE - price)
    
    if risk_per_unit == 0: return 0.0
    
    raw_volume = tranche_risk_usd / (risk_per_unit * (info.trade_tick_value / info.trade_tick_size))
    volume = round(raw_volume / info.volume_step) * info.volume_step
    return max(info.volume_min, min(volume, info.volume_max))

def run_eth_sleeve():
    """Main execution block for the ETH trend-following short."""
    if not mt5.initialize():
        print("MT5 Init Failed")
        return
        
    acc = mt5.account_info()
    if not acc: return
    equity = acc.equity
    
    state = load_state()
    if state["cumulative_pnl"] <= KILL_SWITCH_USD:
        print("💀 KILL SWITCH TRIPPED. ETH sleeve halted forever.")
        return
        
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick: return
    current_price = tick.bid
    
    positions = mt5.positions_get(symbol=SYMBOL)
    eth_pos = [p for p in positions if p.magic == MAGIC_ETH]
    
    if len(eth_pos) == 0:
        state["tranches_open"] = 0
        
    if state["tranches_open"] == 0:
        if not check_combined_margin(equity): return
        
        vol = get_tranche_volume(equity, current_price)
        print(f"🚀 [TRANCHE 1] Opening tactical {vol} lots SHORT on {SYMBOL} @ {current_price}")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": vol,
            "type": mt5.ORDER_TYPE_SELL,
            "price": current_price,
            "sl": HARD_STOP_PRICE,
            "deviation": 20,
            "magic": MAGIC_ETH,
            "comment": "ETH T1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ Tranche 1 Executed Successfully!")
            state["tranches_open"] = 1
            state["last_entry_price"] = current_price
            save_state(state)
        else:
            print(f"❌ Order failed: {result.comment if result else mt5.last_error()}")

    elif state["tranches_open"] == 1:
        now = datetime.now(timezone.utc)
        # Add size only if macro event passed and we are in profit (lower price)
        if now > MACRO_EVENT_TIME_UTC and current_price < state["last_entry_price"]:
            if not check_combined_margin(equity): return
            
            vol = get_tranche_volume(equity, current_price)
            print(f"🔥 [TRANCHE 2] Trend confirmed post-event. Adding {vol} lots SHORT @ {current_price}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": vol,
                "type": mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "sl": HARD_STOP_PRICE,
                "deviation": 20,
                "magic": MAGIC_ETH,
                "comment": "ETH T2",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print("✅ Tranche 2 Executed Successfully!")
                state["tranches_open"] = 2
                save_state(state)
            else:
                print(f"❌ Order failed: {result.comment if result else mt5.last_error()}")
        else:
            print(f"⏳ Holding Tranche 1. Waiting for macro clear & lower lows. Current: {current_price}")

    mt5.shutdown()

if __name__ == "__main__":
    run_eth_sleeve()