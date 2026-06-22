"""
donchian_backtest.py — validate the breakout signal on MT5 data BEFORE going live.

Run this ON THE VPS (it pulls bars from the same MT5 feed you'll trade).
It does NOT trade. It tells you whether Donchian breakout has an edge on YOUR
venue's crypto data, on a held-out tail, after pessimistic costs.

ACCEPTANCE (printed per instrument):
  PASS only if  net_return > buy&hold  AND  profit_factor > 1.5  AND  trades >= 15
Trend systems have LOW win rates (~35-45%) — judge on profit factor & net return,
NOT win rate.
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import mt5_executor as ex

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
ENTRY_PERIOD = 48
EXIT_PERIOD  = 24
COST_BPS   = 0.0010      # 10 bps per side (spread)
SLIP_BPS   = 0.0008      # +8 bps per side (breakout fills are not gentle)
HOLDOUT_FRAC = 0.30      # test on the last 30%, unseen
N_BARS = 5000            # ~52 days of 15m bars


def load(symbol):
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, N_BARS)
    if rates is None or len(rates) < 500:
        return None
    return pd.DataFrame(rates)


def backtest(symbol):
    df = load(symbol)
    if df is None:
        print(f"\n{symbol}: insufficient data"); return

    df["eu"] = df["high"].shift(1).rolling(ENTRY_PERIOD).max()
    df["el"] = df["low"].shift(1).rolling(ENTRY_PERIOD).min()
    df["xu"] = df["high"].shift(1).rolling(EXIT_PERIOD).max()
    df["xl"] = df["low"].shift(1).rolling(EXIT_PERIOD).min()
    df = df.dropna().reset_index(drop=True)

    split = int(len(df) * (1 - HOLDOUT_FRAC))
    test = df.iloc[split:].reset_index(drop=True)

    cost = COST_BPS + SLIP_BPS
    pos, entry, trades = 0, 0.0, []
    for _, r in test.iterrows():
        if pos == 1 and r["low"] < r["xl"]:
            px = r["xl"] * (1 - cost)
            trades.append((px - entry) / entry); pos = 0
        elif pos == -1 and r["high"] > r["xu"]:
            px = r["xu"] * (1 + cost)
            trades.append((entry - px) / entry); pos = 0
        if pos == 0:
            if r["close"] > r["eu"]:
                pos, entry = 1, r["close"] * (1 + cost)
            elif r["close"] < r["el"]:
                pos, entry = -1, r["close"] * (1 - cost)

    bh = (test["close"].iloc[-1] - test["close"].iloc[0]) / test["close"].iloc[0]

    if not trades:
        print(f"\n{symbol}: 0 trades on holdout"); return
    t = np.array(trades)
    win_rate = (t > 0).mean()
    gw = t[t > 0].sum(); gl = abs(t[t <= 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    net = np.prod(1 + t) - 1
    eq = np.cumprod(1 + t); dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()

    passed = (net > bh) and (pf > 1.5) and (len(t) >= 15)
    print(f"\n{symbol}  [holdout {len(test)} bars]")
    print(f"  trades        {len(t)}    {'(>=15 ok)' if len(t)>=15 else '(<15 UNRELIABLE)'}")
    print(f"  win rate      {win_rate:.1%}   (low is normal for trend)")
    print(f"  profit factor {pf:.2f}   (need > 1.5)")
    print(f"  net return    {net:+.2%}")
    print(f"  buy & hold    {bh:+.2%}")
    print(f"  max drawdown  {dd:.2%}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    ex.connect()
    print("Donchian backtest on MT5 crypto data (holdout, costs+slippage)")
    for s in SYMBOLS:
        backtest(s)
    mt5.shutdown()
