#!/usr/bin/env python3
"""
fetch_binance.py — pull ~3 months of M15 klines for backtesting (Mac, no key).

Uses Binance's public klines endpoint (no auth). Paginates backward to cover the
requested window, saves one parquet per symbol to ./crypto_data/.

IMPORTANT FOR THE BACKTEST: Binance is one of the TIGHTEST-spread venues on
earth. Your MT5 broker's crypto spreads are wider (you saw BTC/ETH/SOL print
oddly tight and BARUSD proportionally huge). So use this data to test SIGNAL
QUALITY, but in the backtest apply PESSIMISTIC (wide) spread costs from your
broker — never Binance-tight costs — or you'll flatter the strategy.

Covers 4 of your 7 instruments. Metals (XAU/XAG) and BARUSD are NOT on Binance.

DEPS:  pip install requests pandas pyarrow
USAGE: python fetch_binance.py
"""

import os
import time
import requests
import pandas as pd

SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL  = "15m"
DAYS_BACK = 90
OUT_DIR   = "crypto_data"
BASE      = "https://api.binance.com/api/v3/klines"

BARS_PER_DAY = 96                       # 24h / 15m
TARGET_BARS  = DAYS_BACK * BARS_PER_DAY
MS_PER_BAR   = 15 * 60 * 1000


def fetch_symbol(symbol):
    """Paginate backward from now until we have ~TARGET_BARS bars."""
    end_time = int(time.time() * 1000)
    rows = []
    while len(rows) < TARGET_BARS:
        params = {"symbol": symbol, "interval": INTERVAL,
                  "limit": 1000, "endTime": end_time}
        r = requests.get(BASE, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows = batch + rows                       # prepend older data
        oldest_open = batch[0][0]
        end_time = oldest_open - 1                # step back before oldest
        print(f"  {symbol}: {len(rows)} bars (back to "
              f"{pd.to_datetime(oldest_open, unit='ms')})")
        time.sleep(0.3)                           # gentle on the public API
        if len(batch) < 1000:                     # ran out of history
            break

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["t"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("t").drop(columns=["open_time"])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching ~{DAYS_BACK}d of {INTERVAL} bars for {len(SYMBOLS)} symbols...\n")
    for sym in SYMBOLS:
        df = fetch_symbol(sym)
        path = os.path.join(OUT_DIR, f"{sym}.parquet")
        df.to_parquet(path)
        print(f"  saved {len(df)} bars -> {path}\n")
    print("Done. Metals (XAU/XAG) and BARUSD are not on Binance — "
          "crypto leg only.")


if __name__ == "__main__":
    main()