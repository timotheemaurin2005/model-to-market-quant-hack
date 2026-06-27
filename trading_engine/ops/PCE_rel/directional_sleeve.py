"""
directional_sleeve.py — Top 10 Offensive Protocol
PURPOSE: Captures volatility breakouts following PCE data.
RULES:
1. Entry: Price breaks 20-period high/low on M15.
2. Sizing: 2% of total equity risk (Aggressive for Rank 10 push).
3. Stop Loss: 2x ATR (Volatility-adjusted).
4. Take Profit: 3x ATR.
5. Safety: Must be combined with pce_circuit_breaker.py.
6. Concentration: Never exceed 85% single instrument exposure.
"""

import MetaTrader5 as mt5
import time
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────
RISK_PCT = 0.02          # 2% risk per trade
ACCOUNT_EQUITY = 1_000_000  # Fixed for competition
SYMBOLS = ["XAUUSD", "BTCUSD", "SOLUSD"]
MAGIC_OFFENSIVE = 20260625
CONCENTRATION_LIMIT = 0.85  # Max 85% single instrument


# ── BREAKOUT SIGNAL ───────────────────────────────────────────
def calculate_breakout(symbol):
    """
    Returns:
        signal: 1 = long breakout, -1 = short breakout, 0 = no signal
        atr: Average True Range for position sizing
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if rates is None or len(rates) < 25:
        print(f"⚠️ Insufficient data for {symbol}")
        return 0, 0

    df = pd.DataFrame(rates)

    # 20-period breakout (use shift to avoid lookahead bias)
    upper = df['high'].shift(1).rolling(20).max().iloc[-1]
    lower = df['low'].shift(1).rolling(20).min().iloc[-1]

    # ATR (14-period)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    if atr == 0:
        return 0, 0

    curr = df['close'].iloc[-1]

    if curr > upper:
        return 1, atr
    if curr < lower:
        return -1, atr
    return 0, atr


# ── LOT SIZE CALCULATOR ───────────────────────────────────────
def calculate_lot_size(symbol, atr):
    """
    Risk-adjusted lot sizing using symbol-specific tick value.
    Risk = ACCOUNT_EQUITY * RISK_PCT
    Stop distance = 2 * ATR
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.01

    tick_value = info.trade_tick_value  # $ value per 1 tick move
    point = info.point

    if tick_value == 0 or point == 0 or atr == 0:
        return 0.01

    # Dollar risk per lot = (ATR / point) * tick_value
    risk_per_lot = (atr / point) * tick_value * 2  # 2x ATR stop

    if risk_per_lot == 0:
        return 0.01

    lot_size = round((ACCOUNT_EQUITY * RISK_PCT) / risk_per_lot, 2)

    # Clamp to broker limits
    lot_size = max(info.volume_min, min(lot_size, info.volume_max))
    return lot_size


# ── CONCENTRATION CHECK ───────────────────────────────────────
def check_concentration(symbol, lot_size):
    """
    Returns True if trade is safe to place (under 85% concentration).
    """
    positions = mt5.positions_get()
    if not positions:
        return True

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    new_notional = lot_size * tick.last

    total_notional = sum([
        p.volume * mt5.symbol_info_tick(p.symbol).last
        for p in positions
        if mt5.symbol_info_tick(p.symbol) is not None
    ])

    if total_notional == 0:
        return True

    concentration = new_notional / (total_notional + new_notional)

    if concentration > CONCENTRATION_LIMIT:
        print(f"⚠️ Concentration limit hit: {concentration:.1%} — skipping {symbol}")
        return False

    return True


# ── MAIN LOOP ─────────────────────────────────────────────────
def run_offensive_cycle():
    print("🚀 OFFENSIVE SLEEVE ACTIVE — Watching for PCE breakouts...")
    print(f"   Symbols: {SYMBOLS}")
    print(f"   Risk per trade: {RISK_PCT*100}% of ${ACCOUNT_EQUITY:,}")
    print(f"   Concentration limit: {CONCENTRATION_LIMIT*100}%")
    print("-" * 50)

    try:
        while True:
            for sym in SYMBOLS:
                signal, atr = calculate_breakout(sym)

                if signal == 0:
                    continue

                # Check no existing position for this symbol
                existing = mt5.positions_get(symbol=sym, magic=MAGIC_OFFENSIVE)
                if existing:
                    continue

                # Calculate lot size
                lot_size = calculate_lot_size(sym, atr)
                if lot_size <= 0:
                    print(f"⚠️ Invalid lot size for {sym}")
                    continue

                # Concentration guardrail
                if not check_concentration(sym, lot_size):
                    continue

                # Get current price
                tick = mt5.symbol_info_tick(sym)
                if tick is None:
                    print(f"⚠️ No tick data for {sym}")
                    continue

                price = tick.ask if signal == 1 else tick.bid
                order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

                sl = price - (2 * atr) if signal == 1 else price + (2 * atr)
                tp = price + (3 * atr) if signal == 1 else price - (3 * atr)

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": sym,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": round(sl, mt5.symbol_info(sym).digits),
                    "tp": round(tp, mt5.symbol_info(sym).digits),
                    "deviation": 20,
                    "magic": MAGIC_OFFENSIVE,
                    "comment": "TOP_10_OFFENSIVE",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    direction = "LONG 🟢" if signal == 1 else "SHORT 🔴"
                    print(f"🔥 BREAKOUT FIRED: {sym} | {direction} | Size: {lot_size} | Price: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
                else:
                    print(f"❌ Order failed for {sym}: {result.retcode} — {result.comment}")

            time.sleep(15)  # Check every 15 seconds (aligns with Sharpe calculation window)

    except KeyboardInterrupt:
        print("\n⛔ Offensive sleeve stopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")


# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        exit(1)

    print(f"✅ MT5 connected | Account: {mt5.account_info().login}")
    run_offensive_cycle()