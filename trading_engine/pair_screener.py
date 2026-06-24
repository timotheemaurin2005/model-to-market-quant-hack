"""
pair_screener.py — cointegration + half-life screen for Model to Market.

Pipeline:
    parquet -> wide price frame -> Engle-Granger cointegration (ADF on spread)
            -> OU half-life (correct coefficient + correct units)
            -> filter (p < 0.05 AND half_life >= 120 min) -> ranked CSV

Two things people get wrong, handled here explicitly:
  1. The half-life coefficient comes from a Dickey-Fuller-form regression of
     d(spread) on lagged spread level -- NOT from statsmodels.adfuller output.
  2. Half-life is in BARS. Multiply by the parquet's bar size (BAR_MINUTES),
     which is the data's sampling interval, not your 15-min polling loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ----------------------------------------------------------------------------
# CONFIG  -- the only block you should need to touch
# ----------------------------------------------------------------------------
PARQUET_PATH = "/Users/timotheemaurin/My Drive/Backtesting Log/prices.parquet"  # EDIT
OUTPUT_CSV = "/Users/timotheemaurin/trading/outputs/ranked_pairs.csv"           # EDIT

BAR_MINUTES = 15          # <- SET to the parquet's bar size, NOT the poll rate.
HALF_LIFE_MIN_MINUTES = 120    # floor: too fast for 15-min polling + human exec
HALF_LIFE_MAX_MINUTES = 1440   # ceiling: ~2 half-lives = time to exit (Z 2.0->0.5),
                               # so 1440 = ~48h round-trip. 7200 (5d) is too loose:
                               # it admits pairs that can't close in the competition.
                               # PROVISIONAL -- set from the screener's distribution.
COINT_PVALUE_MAX = 0.05        # ADF p-value ceiling on the spread
MIN_OBS = 500                 # require enough overlapping bars to trust the fit
LOOKBACK_DAYS = 60            # screen on recent regime only; sliced BY TIMESTAMP
                              # (60 calendar days for every symbol, regardless of
                              # 24/5 FX vs 24/7 crypto session hours)

# Curated candidates from the vault. Crypto pairs are included on purpose so the
# filter EXCLUDES them with evidence rather than by assumption (better for the
# tech-prize writeup). Set SCREEN_ALL_PAIRS = True to brute-force the universe.
CANDIDATE_PAIRS = [
    ("XAUUSD", "XAGUSD"),
    ("EURUSD", "EURGBP"),
    ("EURUSD", "EURCHF"),
    ("EURGBP", "EURCHF"),
    ("BTCUSD", "ETHUSD"),
    ("ETHUSD", "SOLUSD"),
]
SCREEN_ALL_PAIRS = False
UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "EURCHF", "EURGBP", "XAUUSD", "XAGUSD",
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",
]


# ----------------------------------------------------------------------------
# DATA LOADING  -- wire this up to your parquet schema
# ----------------------------------------------------------------------------
def load_prices(path: str) -> pd.DataFrame:
    """Return a WIDE frame: DatetimeIndex, one price column per symbol.

    Adapt the body to your file. Two common layouts:

    LONG  (columns: timestamp, symbol, close):
        df = pd.read_parquet(path)
        wide = df.pivot(index="timestamp", columns="symbol", values="close")

    WIDE already (timestamp + one column per symbol):
        wide = pd.read_parquet(path).set_index("timestamp")

    Use mid price if you have bid/ask; close is fine for screening.
    """
    df = pd.read_parquet(path)
    # --- EDIT FROM HERE if your schema differs -------------------------------
    if {"symbol", "close"}.issubset(df.columns):
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        wide = df.pivot(index=ts_col, columns="symbol", values="close")
    else:
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        wide = df.set_index(ts_col)
    # -------------------------------------------------------------------------
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# ----------------------------------------------------------------------------
# STATS ENGINE  -- schema-agnostic; this is the part that has to be right
# ----------------------------------------------------------------------------
def hedge_ratio_and_spread(a: pd.Series, b: pd.Series):
    """OLS hedge ratio for screening: a = alpha + beta*b + spread.

    NOTE: this static beta is for SELECTION ONLY. Live trading uses the
    dynamic Kalman beta. Do not feed this beta into live Z-scores.
    """
    x = sm.add_constant(b.values)
    model = sm.OLS(a.values, x).fit()
    alpha, beta = model.params[0], model.params[1]
    spread = a - (alpha + beta * b)
    return beta, spread


def ou_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion, returned in BARS.

    Dickey-Fuller form:  d(s_t) = c + lambda * s_{t-1} + e_t
    Mean reversion requires -1 < lambda < 0.
    Exact OU half-life:  -ln(2) / ln(1 + lambda)
    (The -ln(2)/lambda in the vault is the first-order approximation.)
    """
    s = spread.dropna()
    s_lag = s.shift(1)
    delta = s - s_lag
    df = pd.concat([delta, s_lag], axis=1).dropna()
    df.columns = ["delta", "s_lag"]
    if len(df) < 10:
        return np.inf
    x = sm.add_constant(df["s_lag"].values)
    lam = sm.OLS(df["delta"].values, x).fit().params[1]
    if lam >= 0 or (1.0 + lam) <= 0:   # no reversion / oscillatory -> reject
        return np.inf
    return float(-np.log(2.0) / np.log(1.0 + lam))


def screen_pair(name_a: str, name_b: str, wide: pd.DataFrame) -> dict | None:
    if name_a not in wide.columns or name_b not in wide.columns:
        return None
    pair = wide[[name_a, name_b]].dropna()
    if len(pair) < MIN_OBS:
        return None

    a, b = pair[name_a], pair[name_b]
    beta, spread = hedge_ratio_and_spread(a, b)
    adf_p = adfuller(spread.dropna(), autolag="AIC")[1]

    hl_bars = ou_half_life(spread)
    hl_minutes = hl_bars * BAR_MINUTES  # <-- the units fix

    mu, sigma = spread.mean(), spread.std()
    current_z = (spread.iloc[-1] - mu) / sigma if sigma > 0 else np.nan

    passes_coint = adf_p < COINT_PVALUE_MAX
    passes_hl = (np.isfinite(hl_minutes)
                 and HALF_LIFE_MIN_MINUTES <= hl_minutes <= HALF_LIFE_MAX_MINUTES)

    return {
        "pair": f"{name_a}/{name_b}",
        "n_obs": len(pair),
        "beta": round(beta, 4),
        "adf_pvalue": round(adf_p, 5),
        "half_life_min": round(hl_minutes, 1) if np.isfinite(hl_minutes) else np.inf,
        "half_life_hours": round(hl_minutes / 60.0, 2) if np.isfinite(hl_minutes) else np.inf,
        "current_z": round(current_z, 2) if np.isfinite(current_z) else np.nan,
        "passes_coint": passes_coint,
        "passes_halflife": passes_hl,
        "passes_both": passes_coint and passes_hl,
    }


def main():
    wide = load_prices(PARQUET_PATH)

    # Slice to the lookback window BY TIMESTAMP. This gives exactly LOOKBACK_DAYS
    # calendar days for FX (24/5), metals, and crypto (24/7) alike -- no
    # bars-per-day assumption, no weekend-gap miscounting.
    cutoff = wide.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
    wide = wide.loc[wide.index >= cutoff]
    span_days = (wide.index.max() - wide.index.min()).days
    print(f"Window: last {LOOKBACK_DAYS}d (data spans {span_days}d, "
          f"ending {wide.index.max():%Y-%m-%d %H:%M}). "
          f"{wide.shape[0]} bars x {wide.shape[1]} symbols: {list(wide.columns)}")

    if SCREEN_ALL_PAIRS:
        from itertools import combinations
        pairs = list(combinations([s for s in UNIVERSE if s in wide.columns], 2))
    else:
        pairs = CANDIDATE_PAIRS

    rows = [r for (na, nb) in pairs if (r := screen_pair(na, nb, wide)) is not None]
    if not rows:
        print("No pairs produced results -- check symbol names against the loaded columns.")
        return

    out = pd.DataFrame(rows).sort_values(
        ["passes_both", "adf_pvalue"], ascending=[False, True]
    ).reset_index(drop=True)

    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows -> {OUTPUT_CSV}")
    print(f"Tradeable (passes_both): {int(out['passes_both'].sum())}")


if __name__ == "__main__":
    main()
