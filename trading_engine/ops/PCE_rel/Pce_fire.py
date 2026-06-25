"""
pce_fire.py — PCE Auto-Fire Protocol
PURPOSE: Fire all PCE trades simultaneously on command.
USAGE: 
    python pce_fire.py hot    # Fire all shorts (hot PCE)
    python pce_fire.py cool   # Fire all longs (cool PCE)
    python pce_fire.py flat   # Fire reduced shorts (in-line PCE)

Run at 12:28 UK time. Watch DXY for 90 seconds after 12:30.
Then run with correct argument based on dollar direction.
"""

import MetaTrader5 as mt5
import sys
import time
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
MAGIC = 20260625
ACCOUNT_SIZE = 1_004_321  # Current balance

# ── HOT PCE TRADES (dollar up, everything shorts) ─────────────
HOT_TRADES = [
    {
        "symbol": "XAUUSD",
        "direction": "sell",
        "lots": 15,
        "sl_offset": 40,    # $40 above entry
        "tp_offset": 120,   # $120 below entry
        "comment": "PCE_HOT_XAU"
    },
    {
        "symbol": "XAGUSD",
        "direction": "sell",
        "lots": 15,
        "sl_offset": 1.30,
        "tp_offset": 2.50,
        "comment": "PCE_HOT_XAG"
    },
    {
        "symbol": "AUDUSD",
        "direction": "sell",
        "lots": 10,
        "sl_offset": 0.0060,
        "tp_offset": 0.0150,
        "comment": "PCE_HOT_AUD"
    },
    {
        "symbol": "SOLUSD",
        "direction": "sell",
        "lots": 2000,
        "sl_offset": 3.00,
        "tp_offset": 5.00,
        "comment": "PCE_HOT_SOL"
    },
    {
        "symbol": "BTCUSD",
        "direction": "sell",
        "lots": 10,
        "sl_offset": 1500,
        "tp_offset": 3000,
        "comment": "PCE_HOT_BTC"
    },
]

# ── COOL PCE TRADES (dollar down, flip to longs) ──────────────
COOL_TRADES = [
    {
        "symbol": "XAUUSD",
        "direction": "buy",
        "lots": 10,
        "sl_offset": 40,
        "tp_offset": 120,
        "comment": "PCE_COOL_XAU"
    },
    {
        "symbol": "SOLUSD",
        "direction": "buy",
        "lots": 2000,
        "sl_offset": 3.00,
        "tp_offset": 5.00,
        "comment": "PCE_COOL_SOL"
    },
    {
        "symbol": "EURUSD",
        "direction": "buy",
        "lots": 20,
        "sl_offset": 0.0060,
        "tp_offset": 0.0150,
        "comment": "PCE_COOL_EUR"
    },
]

# ── FLAT PCE TRADES (in-line, reduced size) ───────────────────
FLAT_TRADES = [
    {
        "symbol": "XAUUSD",
        "direction": "sell",
        "lots": 8,
        "sl_offset": 40,
        "tp_offset": 120,
        "comment": "PCE_FLAT_XAU"
    },
    {
        "symbol": "AUDUSD",
        "direction": "sell",
        "lots": 5,
        "sl_offset": 0.0060,
        "tp_offset": 0.0150,
        "comment": "PCE_FLAT_AUD"
    },
]


# ── ORDER EXECUTION ───────────────────────────────────────────
def fire_order(trade):
    symbol = trade["symbol"]
    direction = trade["direction"]
    lots = trade["lots"]

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ No tick for {symbol}")
        return False

    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"❌ No info for {symbol}")
        return False

    digits = info.digits

    if direction == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = round(price - trade["sl_offset"], digits)
        tp = round(price + trade["tp_offset"], digits)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = round(price + trade["sl_offset"], digits)
        tp = round(price - trade["tp_offset"], digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": MAGIC,
        "comment": trade["comment"],
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        arrow = "🟢 LONG" if direction == "buy" else "🔴 SHORT"
        print(f"  ✅ {arrow} {symbol} | {lots} lots @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        return True
    else:
        print(f"  ❌ FAILED {symbol}: {result.retcode} — {result.comment}")
        return False


# ── MARGIN CHECK ──────────────────────────────────────────────
def check_margin_safe():
    account = mt5.account_info()
    if account is None:
        return False
    margin_level = account.equity / max(account.margin, 1) * 100
    if margin_level < 200:
        print(f"⚠️  Margin level {margin_level:.0f}% — too low to fire safely")
        return False
    print(f"✅ Margin level: {margin_level:.0f}% — safe to fire")
    return True


# ── MAIN FIRE SEQUENCE ────────────────────────────────────────
def fire_all(scenario):
    print("\n" + "="*60)
    print(f"🚀 PCE FIRE PROTOCOL — {scenario.upper()} SCENARIO")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')} UK")
    print(f"   Account: ${mt5.account_info().equity:,.2f}")
    print("="*60)

    if not check_margin_safe():
        print("❌ Aborting — margin too low")
        return

    if scenario == "hot":
        trades = HOT_TRADES
        print("\n📉 Firing HOT PCE trades (all shorts)...")
    elif scenario == "cool":
        trades = COOL_TRADES
        print("\n📈 Firing COOL PCE trades (flip to longs)...")
    elif scenario == "flat":
        trades = FLAT_TRADES
        print("\n➡️  Firing FLAT PCE trades (reduced shorts)...")
    else:
        print(f"❌ Unknown scenario: {scenario}")
        print("   Usage: python pce_fire.py [hot|cool|flat]")
        return

    print()
    success = 0
    for trade in trades:
        if fire_order(trade):
            success += 1
        time.sleep(0.5)  # Small delay between orders

    print(f"\n{'='*60}")
    print(f"✅ {success}/{len(trades)} orders fired successfully")
    print(f"   Account equity: ${mt5.account_info().equity:,.2f}")
    print(f"{'='*60}\n")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pce_fire.py [hot|cool|flat]")
        print("  hot  = Dollar up, fire all shorts")
        print("  cool = Dollar down, flip to longs")
        print("  flat = In-line, reduced shorts only")
        sys.exit(1)

    scenario = sys.argv[1].lower()

    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"✅ MT5 connected | Login: {mt5.account_info().login}")

    fire_all(scenario)

    mt5.shutdown()