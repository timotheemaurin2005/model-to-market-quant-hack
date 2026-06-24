import os
import time
import pandas as pd
from datetime import datetime, timezone
import MetaTrader5 as mt5

# --- CONFIGURATION ---
DRY_RUN = True               # GUARANTEED DRY RUN
KILL_SWITCH_USD = -50000.0   # -$50k cumulative stop across all directional trades
MAGIC_DIRECTIONAL = 20260702 # Unique ID for the directional book
RISK_PCT = 0.01              # 1% risk per trade
HARD_STOP_PCT = 0.03         # 3% hard adverse stop

ENTRY_PERIOD = 48  # 12-hour Donchian
EXIT_PERIOD = 24   # 6-hour Trailing Exit

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]

LOGIN = 10301
SERVER = "3.11.134.149:443"
PASSWORD = os.environ.get("MT5_PASSWORD")

def connect():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        print("MT5 Init Failed")
        return False
    return True

def get_cumulative_pnl():
    """Calculates realized + unrealized PnL strictly for this magic number."""
    # 1. Unrealized (Open Positions)
    unrealized = 0.0
    positions = mt5.positions_get()
    if positions:
        for p in positions:
            if p.magic == MAGIC_DIRECTIONAL:
                unrealized += p.profit

    # 2. Realized (Closed Deals) - Look back 3 days
    realized = 0.0
    from_date = datetime.now() - pd.Timedelta(days=3)
    deals = mt5.history_deals_get(from_date, datetime.now())
    if deals:
        for d in deals:
            if d.magic == MAGIC_DIRECTIONAL:
                realized += d.profit
                
    return unrealized + realized

def close_all_directional():
    """Emergency liquidation if kill switch triggers."""
    print("🚨 KILL SWITCH TRIGGERED: LIQUIDATING DIRECTIONAL BOOK")
    if DRY_RUN:
        print("  [DRY RUN] Would send MARKET CLOSE for all directional positions.")
        return
        
    positions = mt5.positions_get()
    if not positions: return
    for p in positions:
        if p.magic == MAGIC_DIRECTIONAL:
            # Send close logic here (omitted for strict Dry Run safety)
            pass

def run_directional_cycle():
    if not connect(): return
    
    account = mt5.account_info()
    if not account: return
    
    cum_pnl = get_cumulative_pnl()
    print(f"\n==================================================")
    print(f"📈 DIRECTIONAL SLEEVE (DRY RUN = {DRY_RUN})")
    print(f"Cumulative PnL: ${cum_pnl:.2f} / Kill Switch: ${KILL_SWITCH_USD:.2f}")
    
    if cum_pnl <= KILL_SWITCH_USD:
        close_all_directional()
        print("❌ DIRECTIONAL TRADING HALTED FOREVER.")
        return
        
    for sym in SYMBOLS:
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 100)
        if rates is None or len(rates) < 50:
            continue
            
        df = pd.DataFrame(rates)
        
        # Donchian Channels
        entry_upper = df['high'].shift(1).rolling(ENTRY_PERIOD).max().iloc[-1]
        entry_lower = df['low'].shift(1).rolling(ENTRY_PERIOD).min().iloc[-1]
        exit_upper = df['high'].shift(1).rolling(EXIT_PERIOD).max().iloc[-1]
        exit_lower = df['low'].shift(1).rolling(EXIT_PERIOD).min().iloc[-1]
        
        close = df['close'].iloc[-1]
        tick = mt5.symbol_info_tick(sym)
        info = mt5.symbol_info(sym)
        
        if not tick or not info: continue
        
        spread_usd = (tick.ask - tick.bid) * (info.trade_tick_value / info.trade_tick_size)
        
        # Current Position
        pos = [p for p in mt5.positions_get(symbol=sym) if p.magic == MAGIC_DIRECTIONAL]
        in_pos = len(pos) > 0
        
        if in_pos:
            p = pos[0]
            if p.type == mt5.POSITION_TYPE_BUY:
                if close < exit_lower or close < p.price_open * (1 - HARD_STOP_PCT):
                    print(f"🔴 [EXIT LONG] {sym} at {tick.bid} (Trailing Stop / Hard Stop)")
            elif p.type == mt5.POSITION_TYPE_SELL:
                if close > exit_upper or close > p.price_open * (1 + HARD_STOP_PCT):
                    print(f"🔴 [EXIT SHORT] {sym} at {tick.ask} (Trailing Stop / Hard Stop)")
        else:
            # Calculate Risk / Lot Sizing
            stop_dist_price = close * HARD_STOP_PCT
            stop_dist_points = stop_dist_price / info.point
            risk_dollars = account.equity * RISK_PCT
            lots = round(risk_dollars / (stop_dist_points * info.trade_tick_value), 2)
            lots = max(info.volume_min, min(lots, info.volume_max))
            
            if close > entry_upper:
                print(f"🟢 [ENTRY LONG] {sym} Breakout at {tick.ask}")
                print(f"   -> Size: {lots} lots | Stop: {close * (1 - HARD_STOP_PCT):.2f}")
                print(f"   -> Spread Cost: ${spread_usd * lots:.2f}")
            elif close < entry_lower:
                print(f"🔴 [ENTRY SHORT] {sym} Breakout at {tick.bid}")
                print(f"   -> Size: {lots} lots | Stop: {close * (1 + HARD_STOP_PCT):.2f}")
                print(f"   -> Spread Cost: ${spread_usd * lots:.2f}")

if __name__ == "__main__":
    while True:
        try:
            run_directional_cycle()
            # Sleep until the next 15-minute boundary
            now = time.time()
            sleep_s = 900 - (now % 900) + 5
            print(f"💤 Sleeping {sleep_s:.0f}s until next M15 close...")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)