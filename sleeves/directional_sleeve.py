"""
directional_sleeve.py — Gold-only short scaling strategy for Model to Market.

Runs ON the VPS in its own terminal or process. It executes a macro-driven 
short scaling strategy on Gold (XAUUSD) using the shared mt5_executor execution layer.
"""

import os
import sys
import json
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trading_engine"))

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import mt5_executor as ex

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DRY_RUN = True                # Independent of the FX core. Flip when verified.
MAGIC_GOLD = 20260703         # Unique magic number for this Gold strategy
SYMBOL = "XAUUSD"
TRANCHE_MARGIN_USD = 50000.0  # $50k margin budget per tranche
MAX_TRANCHES = 3
STOP_LOSS_PRICE = 4250.0      # Hard stop price
KILL_SWITCH_USD = -100000.0   # -$100k cumulative loss kill switch
STATE_FILE = "gold_state.json"
COMPETITION_START = datetime(2026, 6, 21, 22, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------
def load_gold_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading state file: {e}")
    return {"halted": False, "tranches": []}

def save_gold_state(state):
    dir_name = os.path.dirname(os.path.abspath(STATE_FILE)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        print(f"⚠️ Error saving state file: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def reconcile_tranches(state):
    if DRY_RUN:
        # In dry run, we don't actually place trades at the broker, so we don't reconcile
        # to avoid clearing our simulated state.
        return state

    positions = mt5.positions_get(symbol=SYMBOL) or []
    gold_pos = [p for p in positions if p.magic == MAGIC_GOLD]
    
    if not gold_pos:
        if state["tranches"]:
            print("  [reconcile] No open Gold positions found at broker. Clearing state tranches.")
            state["tranches"] = []
            save_gold_state(state)
    else:
        # We have a position at the broker. Make sure the sum of lots matches.
        pos = gold_pos[0]
        broker_volume = pos.volume
        state_volume = sum(t["lots"] for t in state["tranches"])
        
        # If they don't match, sync or warn.
        if not np.isclose(broker_volume, state_volume):
            print(f"  [reconcile] Warning: Broker volume ({broker_volume}) does not match state volume ({state_volume}).")
            if not state["tranches"]:
                print(f"  [reconcile] Ingesting broker position as Tranche 1.")
                state["tranches"].append({
                    "price": pos.price_open,
                    "lots": pos.volume,
                    "time": datetime.now(timezone.utc).isoformat()
                })
                save_gold_state(state)
    return state

# ---------------------------------------------------------------------------
# RISK & P&L CHECKS
# ---------------------------------------------------------------------------
def get_cumulative_pnl():
    """Calculates realized + unrealized PnL strictly for MAGIC_GOLD."""
    # 1. Unrealized (Open Positions)
    unrealized = sum(p.profit for p in (mt5.positions_get() or [])
                     if p.magic == MAGIC_GOLD)
    
    # 2. Realized (Closed Deals) - Look back since COMPETITION_START
    realized = 0.0
    now_utc = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(COMPETITION_START, now_utc)
    if deals:
        realized = sum(d.profit for d in deals if d.magic == MAGIC_GOLD)
        
    return unrealized + realized

def close_all_gold():
    """Emergency liquidation of all Gold positions under MAGIC_GOLD."""
    print("🚨 LIQUIDATING ALL GOLD SLEEVE POSITIONS")
    positions = mt5.positions_get(symbol=SYMBOL) or []
    for p in positions:
        if p.magic == MAGIC_GOLD:
            close_dir = -1 if p.type == mt5.POSITION_TYPE_BUY else +1
            ex.place_order(p.symbol, p.volume, close_dir,
                           comment="gold-liquidate",
                           magic=MAGIC_GOLD, dry_run=DRY_RUN)

# ---------------------------------------------------------------------------
# MACRO NEWS Blackout & RALLY DETECTION
# ---------------------------------------------------------------------------
def is_news_blackout_active(now_utc):
    # GDP Caution: June 24, 2026, 00:00 to 14:00 UTC (release is at ~13:30 UTC)
    gdp_start = datetime(2026, 6, 24, 0, 0, tzinfo=timezone.utc)
    gdp_end = datetime(2026, 6, 24, 14, 0, tzinfo=timezone.utc)
    
    # PCE Caution: June 26, 2026, 00:00 to 14:00 UTC (release is at ~13:30 UTC)
    pce_start = datetime(2026, 6, 26, 0, 0, tzinfo=timezone.utc)
    pce_end = datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc)
    
    if gdp_start <= now_utc <= gdp_end:
        return True, "GDP"
    if pce_start <= now_utc <= pce_end:
        return True, "PCE"
    return False, ""

def is_gold_rallying():
    # Fetch M15 bars to see if the price has been going up
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 50)
    if rates is None or len(rates) < 48:
        print("  [rally check] Insufficient bars to check rally. Defaulting to False (not rallying).")
        return False
        
    df = pd.DataFrame(rates)
    current_close = float(df["close"].iloc[-1])
    
    # Check 4 hours ago (16 bars of 15m) and 12 hours ago (48 bars of 15m)
    close_4h = float(df["close"].iloc[-17])
    close_12h = float(df["close"].iloc[-49])
    
    rally_4h = current_close > close_4h
    rally_12h = current_close > close_12h
    
    # Check EMA20
    ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    rally_ema = current_close > ema20
    
    is_rallying = rally_4h or rally_12h or rally_ema
    print(f"  [rally check] Current close: {current_close:.2f} | 4h ago: {close_4h:.2f} | 12h ago: {close_12h:.2f} | EMA20: {ema20:.2f}")
    print(f"  [rally check] rally_4h={rally_4h}, rally_12h={rally_12h}, rally_ema={rally_ema} -> is_rallying={is_rallying}")
    return is_rallying

# ---------------------------------------------------------------------------
# CYCLE RUN
# ---------------------------------------------------------------------------
def run_directional_cycle():
    state = load_gold_state()
    state = reconcile_tranches(state)
    
    now = datetime.now(timezone.utc)
    print(f"\n==================================================")
    print(f"🏆 GOLD SLEEVE (DRY RUN = {DRY_RUN}) | {now:%Y-%m-%d %H:%M:%S} UTC")
    
    if state.get("halted", False):
        print("❌ GOLD SLEEVE IS HALTED. No actions will be taken.")
        return
        
    # Calculate cumulative PnL
    cum_pnl = get_cumulative_pnl()
    print(f"Cumulative PnL: ${cum_pnl:,.2f}  |  Kill Switch: ${KILL_SWITCH_USD:,.2f}")
    
    # Kill switch check
    if cum_pnl <= KILL_SWITCH_USD:
        print(f"🚨 KILL SWITCH HIT! Cumulative loss ${cum_pnl:,.2f} <= ${KILL_SWITCH_USD:,.2f}")
        close_all_gold()
        state["halted"] = True
        save_gold_state(state)
        return

    # Check XAUUSD ticks/quotes
    mt5.symbol_select(SYMBOL, True)
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if not tick or not info or tick.ask == 0:
        print(f"  [skip] No live quote for {SYMBOL}")
        return
        
    current_ask = tick.ask
    current_bid = tick.bid
    print(f"Current Gold quote: Bid={current_bid:.2f} | Ask={current_ask:.2f}")

    # Hard Stop Loss check (since we are short, we buy back at the ask price)
    if current_ask >= STOP_LOSS_PRICE:
        print(f"🚨 STOP LOSS PRICE HIT! Price {current_ask:.2f} >= Stop Loss {STOP_LOSS_PRICE:.2f}")
        close_all_gold()
        state["halted"] = True
        save_gold_state(state)
        return

    # Evaluate scaling/entry
    num_tranches = len(state["tranches"])
    if num_tranches >= MAX_TRANCHES:
        print(f"  [status] At max tranches ({MAX_TRANCHES}). Position size is fully scaled.")
        return

    # Entry decision:
    should_enter = False
    reason = ""
    
    if num_tranches == 0:
        should_enter = True
        reason = "Initial tranche entry (no active tranches)"
    else:
        # Check if price is lower than previous tranche entry price
        last_tranche = state["tranches"][-1]
        last_price = last_tranche["price"]
        # Since we are shorting, the entry price is the bid price
        if current_bid < last_price:
            should_enter = True
            reason = f"Trend confirmation: current bid {current_bid:.2f} < previous entry {last_price:.2f}"
        else:
            reason = f"Price check failed: current bid {current_bid:.2f} is not lower than previous entry {last_price:.2f}"

    if should_enter:
        # Check News Blackout Caution
        is_blackout, news_type = is_news_blackout_active(now)
        if is_blackout:
            print(f"  [news] Pre-news blackout window active for {news_type} print.")
            if is_gold_rallying():
                print(f"  [news] BLOCKED: Gold is rallying leading into {news_type} print.")
                should_enter = False
            else:
                print(f"  [news] ALLOWED: Gold is not rallying leading into {news_type} print.")

    if should_enter:
        print(f"👉 Entry Signal: {reason}")
        # Calculate size by margin
        lots, est_margin = ex.size_by_margin(SYMBOL, TRANCHE_MARGIN_USD, -1) # -1 for Short
        if lots is None or est_margin is None:
            print("  [error] Size calculation failed.")
            return
            
        print(f"  Proposed Tranche {num_tranches + 1}: Short {lots} lots | Margin: ${est_margin:,.2f}")
        
        # Check account-wide guardrails
        ok, gr_reason, stats = ex.check_guardrails([(SYMBOL, lots, -1)])
        print(f"  Guardrail check: {'PASS' if ok else 'REJECT - ' + gr_reason}")
        if not ok:
            return
            
        # Place order with Stop Loss of 4250.0
        res = ex.place_order(SYMBOL, lots, -1, 
                             comment=f"gold-tranche-{num_tranches+1}", 
                             magic=MAGIC_GOLD, 
                             dry_run=DRY_RUN, 
                             sl=STOP_LOSS_PRICE)
        
        # Record tranche if execution succeeded (or if dry run)
        if res is not None:
            state["tranches"].append({
                "price": current_bid,
                "lots": lots,
                "time": now.isoformat()
            })
            save_gold_state(state)
            print(f"✅ Tranche {len(state['tranches'])} recorded in state.")
    else:
        print(f"  [status] No entry signal. {reason}")

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ex.connect()
    print(f"Gold scaling directional loop started. DRY_RUN = {DRY_RUN}. Ctrl+C to stop.")
    try:
        while True:
            try:
                run_directional_cycle()
            except Exception as e:
                print(f"⚠️ [cycle error] {type(e).__name__}: {e}")
                
            # Sleep until the next 15-minute boundary
            now = time.time()
            sleep_s = 900 - (now % 900) + 5
            print(f"💤 Sleeping {sleep_s:.0f}s until next M15 close...")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print(f"Shutdown complete. DRY_RUN = {DRY_RUN}")