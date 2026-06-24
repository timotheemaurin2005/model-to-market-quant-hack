"""
kalman_calibration.py — offline Kalman tuning for Model to Market.

Turns ranked_pairs.csv (from pair_screener.py) into kalman_config.json: a frozen
per-pair parameter set the LIVE Northflank filter loads and steps with
filter_update() every 15 min. EM runs HERE, offline, once -- never in the loop.

Corrections vs. the naive approach:
  * EM runs on the HEDGE-RATIO state [alpha, beta] with a time-varying observation
    matrix [1, x_t] -- not on the scalar spread. The learned transition_covariance
    is therefore the drift of beta, which is what the live beta-filter needs.
  * Warm start: beta0/alpha0/P0 from OLS on the recent tail (no cold start, so
    "48-hour warm-up" never happens regardless of how Q/R were chosen).
  * EM output is CONSTRAINED: R floored at a fraction of residual variance, and
    the effective delta = Q[beta,beta]/R clamped to a band, so a 30-day fit can't
    drive observation noise to zero and overfit microstructure.
  * HELD-OUT validation: EM fits on the train slice; spread quality is judged on
    an unseen tail. A pair that only looks stationary in-sample is flagged REVIEW.

Run locally during the prep window. Requires pair_screener.py in the same dir.
"""

from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter

from pair_screener import ou_half_life  # reuse corrected half-life logic

# ---- CONFIG ----------------------------------------------------------------
# Real data on disk is per-symbol tick parquet (bid/ask), not the single wide
# prices.parquet pair_screener.py was written against — load_prices() there
# doesn't match this schema, so we load ticks directly (see load_prices_from_ticks).
_DEFAULT_DATA_DIR = str(Path(__file__).parent / "pricer-output-2026-05-11_2026-06-10")
DATA_DIR   = Path(_DEFAULT_DATA_DIR)               # EDIT if data moves
CONFIG_OUT = Path(__file__).parent / "outputs" / "kalman_config.json"

BAR_MINUTES     = 15     # must match signal_server.py's BAR_MINUTES (parquet path)
CALIB_DAYS      = 30     # calibration window (recent regime)
VALIDATION_DAYS = 7      # held-out tail inside the calib window (unseen by EM)
WARMSTART_DAYS  = 5      # OLS tail for beta0 / alpha0 / P0
EM_ITERS        = 10
MIN_OBS         = 500

# EM output guards -- these are the safety rails, not cosmetic
R_FLOOR_FRAC = 0.25     # R >= 0.25 * residual variance (block R -> 0 overfit)
DELTA_MAX    = 1e-3     # cap beta-drift/R: reactive ceiling (don't chase noise)
DELTA_MIN    = 1e-6     # floor: don't freeze beta into static OLS

# Held-out acceptance
VAL_ADF_PMAX   = 0.05
VAL_HL_MIN_MIN = 120    # held-out half-life still tradeable (minutes)

# Watchlist pairs to calibrate. Must match signal_server.py's WATCHLIST
# (same symbol order) so the live server's f"{sym_a}/{sym_b}" config lookup
# key matches. XAUUSD/XAGUSD is not in this dataset (signal_server.py docstring);
# these 5 are the curated watchlist from the real 28-pair screen on this data.
WATCHLIST = [
    ("AUDUSD", "USDJPY"),
    ("USDCAD", "USDJPY"),
    ("USDCHF", "USDJPY"),
    ("USDCAD", "USDCHF"),
    ("EURGBP", "EURUSD"),
]


# ---- helpers ---------------------------------------------------------------
def load_prices_from_ticks(data_dir: Path, symbols: list[str], bar_minutes: int) -> pd.DataFrame:
    """Build a WIDE mid-price frame from per-symbol tick parquet files.

    Mirrors signal_server._load_symbol_bars: reads {symbol}_*.parquet (time,
    bid, ask), takes the mid, resamples to bar_minutes bars (last value).
    """
    series = {}
    for sym in symbols:
        files = sorted(glob.glob(str(data_dir / f"{sym}_*.parquet")))
        if not files:
            continue
        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f, engine="fastparquet", columns=["time", "bid", "ask"])
            except Exception:
                continue
            df["time"] = pd.to_datetime(df["time"])
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            dfs.append(df[["time", "mid"]])
        if not dfs:
            continue
        all_ticks = pd.concat(dfs).sort_values("time").set_index("time")
        series[sym] = all_ticks["mid"].resample(f"{bar_minutes}min").last().dropna()

    wide = pd.DataFrame(series).sort_index()
    return wide


def ols_warmstart(y: pd.Series, x: pd.Series):
    """Static OLS for the warm start: y = alpha + beta*x. Returns state in
    [alpha, beta] order to match the Kalman state vector."""
    X = sm.add_constant(x.values)
    res = sm.OLS(y.values, X).fit()
    alpha, beta = float(res.params[0]), float(res.params[1])
    P0 = np.asarray(res.cov_params(), float)        # 2x2 cov of [alpha, beta]
    resid_var = float(np.var(res.resid))            # -> seeds R, NOT P0
    return alpha, beta, P0, resid_var


def obs_matrices(x: pd.Series) -> np.ndarray:
    """Time-varying observation matrix [1, x_t], shape (n, 1, 2)."""
    n = len(x)
    m = np.ones((n, 1, 2))
    m[:, 0, 1] = np.asarray(x.values, float)
    return m


def run_filter(px: pd.DataFrame, a: str, b: str, alpha0, beta0, P0, Q, R):
    """Run the frozen filter forward (causal). Returns (beta_t, alpha_t)."""
    kf = KalmanFilter(
        n_dim_state=2, n_dim_obs=1,
        transition_matrices=np.eye(2),
        observation_matrices=obs_matrices(px[b]),
        initial_state_mean=[alpha0, beta0],
        initial_state_covariance=P0,
        transition_covariance=Q,
        observation_covariance=np.array([[R]]),
    )
    means, _ = kf.filter(px[a].values)
    return means[:, 1], means[:, 0]


def calibrate_pair(a: str, b: str, wide: pd.DataFrame) -> dict | None:
    if a not in wide.columns or b not in wide.columns:
        return None
    px = wide[[a, b]].dropna()
    end = px.index.max()
    calib = px.loc[px.index >= end - pd.Timedelta(days=CALIB_DAYS)]
    if len(calib) < MIN_OBS:
        return None

    val_start = end - pd.Timedelta(days=VALIDATION_DAYS)
    train = calib.loc[calib.index < val_start]
    ws = train.loc[train.index >= train.index.max() - pd.Timedelta(days=WARMSTART_DAYS)]
    alpha0, beta0, P0, resid_var = ols_warmstart(ws[a], ws[b])

    # --- EM on TRAIN only: learn the two noise matrices, keep warm-start state
    kf = KalmanFilter(
        n_dim_state=2, n_dim_obs=1,
        transition_matrices=np.eye(2),
        observation_matrices=obs_matrices(train[b]),
        initial_state_mean=[alpha0, beta0],
        initial_state_covariance=P0,
        transition_covariance=np.eye(2) * resid_var * 1e-3,   # seed
        observation_covariance=np.array([[resid_var]]),       # seed
        em_vars=["transition_covariance", "observation_covariance"],
    ).em(train[a].values, n_iter=EM_ITERS)

    Q = np.asarray(kf.transition_covariance, float)
    R = float(np.asarray(kf.observation_covariance).ravel()[0])

    # --- constrain EM output (the load-bearing safety step) ------------------
    R = max(R, R_FLOOR_FRAC * resid_var)
    delta = Q[1, 1] / R
    if delta > DELTA_MAX:
        Q *= DELTA_MAX / delta
    elif 0 < delta < DELTA_MIN:
        Q *= DELTA_MIN / delta
    delta_eff = float(Q[1, 1] / R)

    # --- held-out validation: run frozen filter over calib, judge the tail ---
    beta_t, alpha_t = run_filter(calib, a, b, alpha0, beta0, P0, Q, R)
    dyn_spread = pd.Series(calib[a].values - (alpha_t + beta_t * calib[b].values),
                           index=calib.index)
    val = dyn_spread.loc[dyn_spread.index >= val_start].dropna()
    adf_p = float(adfuller(val)[1])
    hl_min = ou_half_life(val) * BAR_MINUTES
    passed = (adf_p < VAL_ADF_PMAX and np.isfinite(hl_min) and hl_min >= VAL_HL_MIN_MIN)

    return {
        "pair": f"{a}/{b}",
        "alpha0": alpha0, "beta0": beta0,
        "P0": P0.tolist(),
        "Q": Q.tolist(),
        "R": R,
        "delta_eff": delta_eff,
        "bar_minutes": BAR_MINUTES,
        "validation": {
            "adf_pvalue": round(adf_p, 5),
            "half_life_min": round(hl_min, 1) if np.isfinite(hl_min) else None,
            "n_val_obs": int(len(val)),
        },
        "status": "PASS" if passed else "REVIEW",
    }


def main():
    symbols = sorted({s for pair in WATCHLIST for s in pair})
    wide = load_prices_from_ticks(DATA_DIR, symbols, BAR_MINUTES)
    pairs = WATCHLIST
    print(f"Calibrating {len(pairs)} watchlist pair(s): {pairs}")

    configs = {}
    for a, b in pairs:
        cfg = calibrate_pair(a, b, wide)
        if cfg is None:
            print(f"  {a}/{b}: insufficient data, skipped")
            continue
        v = cfg["validation"]
        print(f"  {cfg['pair']:<16} delta={cfg['delta_eff']:.2e}  "
              f"R={cfg['R']:.3g}  val_adf_p={v['adf_pvalue']}  "
              f"val_hl={v['half_life_min']}min  -> {cfg['status']}")
        configs[cfg["pair"]] = cfg

    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_OUT, "w") as f:
        json.dump(configs, f, indent=2)
    n_pass = sum(c["status"] == "PASS" for c in configs.values())
    print(f"\nWrote {len(configs)} configs -> {CONFIG_OUT}  ({n_pass} PASS, "
          f"{len(configs) - n_pass} REVIEW)")
    print("Trade only PASS pairs live. REVIEW = looked good in-sample, "
          "degraded out-of-sample.")


if __name__ == "__main__":
    main()
