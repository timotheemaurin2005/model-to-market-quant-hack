#!/usr/bin/env python3
"""
momentum_trader.py — Donchian/ATR breakout engine (DRY_RUN LEARNING BUILD)

PURPOSE: a LEARNING tool, run with DRY_RUN=True alongside the live FX book.
It generates breakout signals on metals + crypto, logs every would-be trade to
a CSV (including the live spread cost at signal time), and places NOTHING. Read
the log to understand how directional momentum behaves vs your market-neutral FX
stat-arb — that contrast is the whole point.

This is deliberately NOT wired for live trading. It uses a separate magic number
and never imports or touches live_trader / mt5_executor state. Keep DRY_RUN=True.

WHAT THIS BUILD DOES vs the first draft:
  * Startup instrument validation: every symbol is checked against the broker
    via symbol_info()/tick before any signal is trusted. Invalid or untradeable
    symbols are dropped with a printed reason (the key habit to learn).
  * Signal logging WITH COST: every would-be entry is appended to
    momentum_signals.csv with the live bid/ask, spread in points, and the
    round-trip spread cost in dollars on the sized position — because breakouts
    fire exactly when spreads are widest, and close-price EV is a mirage.
  * Guaranteed teardown: a top-level try/finally closes the MT5 connection on
    ANY exit path (crash, exception, or Ctrl+C) — no dangling connection.
  * Removed the `Magic_Num =` alias and the fragile filling-mode expression.

DEPS:  pip install MetaTrader5 numpy pandas
"""

import os
import csv
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DRY_RUN = True              # KEEP TRUE. This is a learning/observation build.

# Confirmed competition instruments (metals + crypto):
TRADE_UNIVERSE = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "BARUSD"]

MAGIC_MOMENTUM = 20260701   # isolated; unrelated to the FX book's magic
SIGNAL_LOG     = "momentum_signals.csv"

CHANNEL_PERIOD = 20
ATR_PERIOD     = 14
RISK_PER_TRADE = 0.015
ATR_MULTIPLIER = 2.0

TIMEFRAME = mt5.TIMEFRAME_M15

LOGIN    = 10301
SERVER   = "3.11.134.149:443"
PASSWORD = os.environ.get("MT5_PASSWORD")


# ---------------------------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------------------------
def connect():
    if PASSWORD is None:
        raise RuntimeError("MT5_PASSWORD env var is not set.")
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5.account_info()


# ---------------------------------------------------------------------------
# STARTUP INSTRUMENT VALIDATION  (the habit worth learning)
# ---------------------------------------------------------------------------
def validate_universe(symbols):
    """Confirm each symbol is real, selectable, and has sane tick economics.
    Returns the subset that's actually tradeable. Prints why any is dropped."""
    good = []
    print("\n--- instrument validation ---")
    for s in symbols:
        if not mt5.symbol_select(s, True):
            print(f"  {s:8} DROPPED: symbol_select failed (not in this broker's book)")
            continue
        info = mt5.symbol_info(s)
        if info is None:
            print(f"  {s:8} DROPPED: symbol_info returned None")
            continue
        if info.trade_tick_size == 0 or info.trade_tick_value == 0:
            print(f"  {s:8} DROPPED: zero tick_size/tick_value (can't size safely)")
            continue
        tick = mt5.symbol_info_tick(s)
        if tick is None or tick.bid == 0:
            print(f"  {s:8} DROPPED: no live tick / zero bid")
            continue
        spread_pts = (tick.ask - tick.bid) / info.trade_tick_size if info.trade_tick_size else float("inf")
        note = "  (WIDE SPREAD — watch for false breakouts)" if spread_pts > 50 else ""
        print(f"  {s:8} OK   bid={tick.bid} spread~{spread_pts:.0f}pts "
              f"vol_min={info.volume_min}{note}")
        good.append(s)
    print(f"--- {len(good)}/{len(symbols)} tradeable ---\n")
    return good


# ---------------------------------------------------------------------------
# DATA / SIGNALS
# ---------------------------------------------------------------------------
def get_market_data(symbol, n=100):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["t"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("t")


def calculate_breakout_signals(df):
    highs, lows, closes = df["high"], df["low"], df["close"]
    upper = highs.shift(1).rolling(CHANNEL_PERIOD).max()
    lower = lows.shift(1).rolling(CHANNEL_PERIOD).min()
    tr = pd.concat([highs - lows,
                    (highs - closes.shift(1)).abs(),
                    (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    price = float(closes.iloc[-1])
    up, lo, a = float(upper.iloc[-1]), float(lower.iloc[-1]), float(atr.iloc[-1])
    signal = 1 if price > up else (-1 if price < lo else 0)
    return signal, price, a, up, lo


def calculate_momentum_size(symbol, atr, equity):
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_size == 0 or atr <= 0:
        return None
    point_value = info.trade_tick_value / info.trade_tick_size
    risk_dollars = equity * RISK_PER_TRADE
    stop_dist = atr * ATR_MULTIPLIER
    raw = risk_dollars / (stop_dist * point_value)
    lots = round(round(raw / info.volume_step) * info.volume_step, 8)
    return max(info.volume_min, min(lots, info.volume_max))


# ---------------------------------------------------------------------------
# SIGNAL LOGGING (what you review afterwards)
# ---------------------------------------------------------------------------
def log_signal(row):
    new = not os.path.exists(SIGNAL_LOG)
    with open(SIGNAL_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "symbol", "dir", "price", "atr",
                        "upper", "lower", "lots",
                        "bid", "ask", "spread_pts", "spread_cost_usd", "risk_pct"])
        w.writerow(row)


# ---------------------------------------------------------------------------
# CYCLE  (logs only; never places an order while DRY_RUN)
# ---------------------------------------------------------------------------
def run_cycle(universe):
    acc = connect()
    equity = acc.equity
    now = datetime.now(timezone.utc)
    print(f"\n⚡ momentum DRY-RUN cycle {now:%H:%M}Z | equity ${equity:,.0f}")

    for s in universe:
        df = get_market_data(s)
        if df is None or len(df) < CHANNEL_PERIOD + ATR_PERIOD + 2:
            continue
        signal, price, atr, up, lo = calculate_breakout_signals(df)
        if signal == 0:
            continue
        lots = calculate_momentum_size(s, atr, equity)
        if not lots:
            continue

        # Capture live bid/ask AT SIGNAL TIME — the true cost of entry.
        # Breakouts fire when spreads are widest, so close-price EV is a mirage.
        info = mt5.symbol_info(s)
        tick = mt5.symbol_info_tick(s)
        if tick and info and info.trade_tick_size:
            bid, ask = tick.bid, tick.ask
            spread_pts = (ask - bid) / info.trade_tick_size
            point_value = info.trade_tick_value / info.trade_tick_size
            # round-trip spread cost in dollars on the sized position:
            spread_cost = (ask - bid) * point_value * lots
        else:
            bid = ask = spread_pts = spread_cost = float("nan")

        direction = "LONG" if signal > 0 else "SHORT"
        print(f"  [SIGNAL] {s:8} {direction} price={price} atr={atr:.4f} "
              f"lots={lots} spread~{spread_pts:.0f}pts cost~${spread_cost:.0f}"
              f"  (DRY-RUN: nothing placed)")
        log_signal([now.isoformat(), s, direction, price, round(atr, 6),
                    round(up, 6), round(lo, 6), lots,
                    round(bid, 6), round(ask, 6), round(spread_pts, 1),
                    round(spread_cost, 2), RISK_PER_TRADE])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Momentum DRY-RUN learning engine. DRY_RUN =", DRY_RUN)
    if not DRY_RUN:
        raise SystemExit("This build is for observation only. Keep DRY_RUN=True.")
    try:
        connect()
        universe = validate_universe(TRADE_UNIVERSE)
        if not universe:
            raise SystemExit("No tradeable instruments after validation.")
        while True:
            now = time.time()
            sleep_s = 900 - (now % 900) + 10
            print(f"...sleeping {sleep_s:.0f}s to next M15 close")
            time.sleep(sleep_s)
            try:
                run_cycle(universe)
            except Exception as e:
                print(f"[momentum error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mt5.shutdown()   # always closes the connection — crash, exception, or Ctrl+C