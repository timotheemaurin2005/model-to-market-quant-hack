"""
metals_screener.py — XAUUSD/XAGUSD cointegration screen via yfinance.

The tick parquet has no metals data (signal_server.py docstring: "XAUUSD/XAGUSD
is NOT in this dataset"). This fetches GC=F / SI=F intraday bars from yfinance,
labels them XAUUSD/XAGUSD for output, and runs them through pair_screener.py's
UNCHANGED screen_pair() / ou_half_life() — same ADF p<0.05, same 120-1440 min
half-life band, same 60-day lookback discipline. No threshold here is touched;
passes_both alone decides watchlist admission (CLAUDE.md hard rule).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trading_engine"))

import pandas as pd
import yfinance as yf

import pair_screener as ps

YF_METALS_MAP = {"XAUUSD": "GC=F", "XAGUSD": "SI=F"}


def fetch_metals(interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    """Fetch GC=F/SI=F from yfinance, label columns XAUUSD/XAGUSD."""
    series = {}
    for label, ticker in YF_METALS_MAP.items():
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {ticker} ({label})")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        idx = pd.to_datetime(s.index)
        s.index = idx.tz_localize(None) if idx.tz is not None else idx
        s.name = label
        series[label] = s

    wide = pd.DataFrame(series).dropna().sort_index()
    return wide


def infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    """Median spacing between consecutive bars, in minutes (robust to session gaps)."""
    diffs = index.to_series().diff().dropna()
    median_sec = diffs.dt.total_seconds().median()
    return int(round(median_sec / 60.0))


def main():
    wide = fetch_metals()
    bar_minutes = infer_bar_minutes(wide.index)
    print(f"Fetched {len(wide)} aligned bars, {wide.index.min()} -> {wide.index.max()}")
    print(f"Inferred bar size from data: {bar_minutes} min")

    # Same 60-day lookback discipline as pair_screener.main() — sliced by timestamp.
    cutoff = wide.index.max() - pd.Timedelta(days=ps.LOOKBACK_DAYS)
    windowed = wide.loc[wide.index >= cutoff]
    span_days = (windowed.index.max() - windowed.index.min()).days
    print(f"Window: last {ps.LOOKBACK_DAYS}d (data spans {span_days}d), {len(windowed)} bars")

    # BAR_MINUTES discipline (CLAUDE.md hard rule): pair_screener's screen_pair /
    # ou_half_life read the module-level BAR_MINUTES global. Set it to the value
    # WE JUST MEASURED from the fetched data — never hardcode — then assert.
    measured = bar_minutes
    ps.BAR_MINUTES = measured
    assert ps.BAR_MINUTES == measured, "BAR_MINUTES must equal the fetched bar size"

    if len(windowed) < ps.MIN_OBS:
        print(f"\nXAUUSD/XAGUSD: only {len(windowed)} bars (< MIN_OBS={ps.MIN_OBS}) -- cannot screen")
        return

    result = ps.screen_pair("XAUUSD", "XAGUSD", windowed)
    if result is None:
        print("\nXAUUSD/XAGUSD: screen_pair returned None (insufficient overlap)")
        return

    print("\n--- XAUUSD/XAGUSD screen (pair_screener.py screen_pair, UNCHANGED) ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["passes_both"]:
        print("\nDECISION: passes_both=True -> XAUUSD/XAGUSD QUALIFIES for the watchlist.")
        print("(Not wired in this session — reporting only, per instructions.)")
    else:
        killers = []
        if not result["passes_coint"]:
            killers.append(f"cointegration (ADF p={result['adf_pvalue']} >= {ps.COINT_PVALUE_MAX})")
        if not result["passes_halflife"]:
            killers.append(
                f"half-life ({result['half_life_min']} min outside "
                f"[{ps.HALF_LIFE_MIN_MINUTES}, {ps.HALF_LIFE_MAX_MINUTES}])"
            )
        print(f"\nDECISION: passes_both=False -> KEEP OUT. Killed by: {', '.join(killers)}")


if __name__ == "__main__":
    main()
