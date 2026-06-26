#!/usr/bin/env python3
"""
backtest_momentum.py — does the Donchian/ATR momentum idea actually survive costs?

Runs the SAME breakout logic as momentum_trader.py over the crypto history pulled
by fetch_binance.py, then scores it on the competition's metrics so you can compare
apples-to-apples against your FX book:
    • total return
    • Sharpe on 15-MINUTE returns   (matches platform scoring)
    • max drawdown

THE WHOLE POINT — COST REALISM:
  It runs TWICE: once at optimistic (tight, Binance-like) spread, once at a
  pessimistic (wide, broker-like) spread. A strategy that only wins under the
  optimistic cost is NOT viable for your competition. Watch the gap.

This is a SINGLE-INSTRUMENT directional backtest (no hedge) — exactly what the
momentum idea is. Compare its risk-adjusted result to your market-neutral FX book.

DEPS:  pip install pandas numpy pyarrow
USAGE: python backtest_momentum.py
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "crypto_data"

# strategy params — identical to momentum_trader.py
CHANNEL_PERIOD = 20
ATR_PERIOD     = 14
ATR_MULTIPLIER = 2.0
RISK_PER_TRADE = 0.015
TP_ATR         = 3.5            # take-profit distance (matches the live draft's R:R)

# cost scenarios, in FRACTION of price charged once per side (round trip = 2x)
# optimistic ~ Binance tight; pessimistic ~ wider broker crypto spread.
COST_OPTIMISTIC  = 0.0002       # 2 bps/side
COST_PESSIMISTIC = 0.0015       # 15 bps/side — deliberately harsh; tune to broker

START_EQUITY = 1_000_000.0
BARS_PER_YEAR = 96 * 365        # 15-min bars in a year, for Sharpe annualization


def donchian_atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    df = df.copy()
    df["upper"] = h.shift(1).rolling(CHANNEL_PERIOD).max()
    df["lower"] = l.shift(1).rolling(CHANNEL_PERIOD).min()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
                   axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df.dropna()


def backtest_one(df, cost_frac):
    """Event-driven single-instrument breakout backtest on M15 bars.
    Returns a per-bar equity series (mark-to-market)."""
    df = donchian_atr(df)
    equity = START_EQUITY
    pos = 0            # +1 long, -1 short, 0 flat
    entry = stop = tp = 0.0
    units = 0.0        # position size in price-units (risk-based)
    eq_curve = []

    for _, row in df.iterrows():
        price, atr = row["close"], row["atr"]

        # mark-to-market an open position on this bar
        if pos != 0:
            # exit checks first (stop / tp), executed at the level, cost applied
            hit_stop = (pos == 1 and row["low"] <= stop) or (pos == -1 and row["high"] >= stop)
            hit_tp   = (pos == 1 and row["high"] >= tp)  or (pos == -1 and row["low"]  <= tp)
            if hit_stop or hit_tp:
                exit_px = stop if hit_stop else tp
                pnl = pos * (exit_px - entry) * units
                pnl -= cost_frac * exit_px * units      # exit cost
                equity += pnl
                pos = 0

        # entry on breakout if flat
        if pos == 0:
            long_sig  = price > row["upper"]
            short_sig = price < row["lower"]
            if long_sig or short_sig:
                pos = 1 if long_sig else -1
                entry = price
                stop  = entry - pos * ATR_MULTIPLIER * atr
                tp    = entry + pos * TP_ATR * atr
                risk_dollars = equity * RISK_PER_TRADE
                stop_dist = ATR_MULTIPLIER * atr
                units = risk_dollars / stop_dist if stop_dist > 0 else 0.0
                equity -= cost_frac * entry * units      # entry cost

        # record mark-to-market equity
        mtm = equity
        if pos != 0:
            mtm += pos * (price - entry) * units
        eq_curve.append(mtm)

    return pd.Series(eq_curve, index=df.index)


def score(eq):
    """Competition-style metrics on a 15-min equity curve."""
    ret_total = eq.iloc[-1] / eq.iloc[0] - 1.0
    r = eq.pct_change().dropna()
    sharpe = (r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 0 else 0.0
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min()
    return ret_total, sharpe, max_dd


def run_scenario(label, cost):
    print(f"\n===== {label} cost ({cost*1e4:.0f} bps/side) =====")
    combined = None
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet"))):
        sym = os.path.basename(path).replace(".parquet", "")
        df = pd.read_parquet(path)
        eq = backtest_one(df, cost)
        ret, sharpe, dd = score(eq)
        print(f"  {sym:9} return={ret:+7.2%}  sharpe(15m,ann)={sharpe:+6.2f}  maxDD={dd:7.2%}")
        # equal-weight portfolio of the per-symbol equity curves
        norm = eq / eq.iloc[0]
        combined = norm if combined is None else combined.add(norm, fill_value=0)
    if combined is not None:
        combined = combined / combined.iloc[0] * START_EQUITY
        ret, sharpe, dd = score(combined)
        print(f"  {'PORTFOLIO':9} return={ret:+7.2%}  sharpe(15m,ann)={sharpe:+6.2f}  maxDD={dd:7.2%}")


def main():
    if not glob.glob(os.path.join(DATA_DIR, "*.parquet")):
        raise SystemExit(f"No data in {DATA_DIR}/ — run fetch_binance.py first.")
    print("Momentum backtest — Donchian/ATR, single-instrument directional.")
    print("Compare the PORTFOLIO risk-adjusted result to your FX book's score.")
    run_scenario("OPTIMISTIC", COST_OPTIMISTIC)
    run_scenario("PESSIMISTIC", COST_PESSIMISTIC)
    print("\nIf the strategy is only viable under OPTIMISTIC costs, it is NOT viable")
    print("for this competition. The honest answer is the PESSIMISTIC portfolio row,")
    print("compared against your market-neutral FX baseline on the same metrics.")


if __name__ == "__main__":
    main()