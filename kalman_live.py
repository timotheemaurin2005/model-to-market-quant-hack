"""
kalman_live.py — Live Kalman filter for dynamic hedge-ratio tracking.

Loads frozen per-pair configs from kalman_config.json (produced by
kalman_calibration.py) and steps a causal Kalman filter forward to produce
a time-varying β (hedge ratio) for each pair. EM is NEVER run here — only
the frozen Q, R from offline calibration are used.

State vector: [alpha, beta]
Observation:  y_t = H_t @ state + noise,  H_t = [1, x_t]
Transition:   state_{t+1} = I @ state_t + process noise

Key design:
  - filter_update() is a single causal step (no lookahead).
  - current_beta() warm-starts from config and steps across the full window.
  - If the pair has no config or status != "PASS", the caller falls back to OLS.
  - bar_minutes must match between config and server — asserted, not assumed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("kalman_live")


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, dict]:
    """Load kalman_config.json → dict keyed by pair name (e.g. "AUDUSD/USDJPY").

    Returns an empty dict (with a warning) if the file doesn't exist or is invalid.
    Never crashes — missing config means OLS fallback for all pairs.

    Each config entry has keys:
        alpha0, beta0, P0 (2×2 list), Q (2×2 list), R (float),
        bar_minutes (int), status ("PASS" or "REVIEW"),
        validation (dict with adf_pvalue, half_life_min, n_val_obs).
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Kalman config not found at {path} — OLS fallback for all pairs")
        return {}

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse Kalman config at {path}: {e} — OLS fallback")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"Kalman config at {path} is not a dict — OLS fallback")
        return {}

    # Convert P0, Q from lists to numpy arrays for convenience
    configs: dict[str, dict] = {}
    for pair_key, cfg in raw.items():
        try:
            configs[pair_key] = {
                "pair": cfg.get("pair", pair_key),
                "alpha0": float(cfg["alpha0"]),
                "beta0": float(cfg["beta0"]),
                "P0": np.array(cfg["P0"], dtype=np.float64),
                "Q": np.array(cfg["Q"], dtype=np.float64),
                "R": float(cfg["R"]),
                "bar_minutes": int(cfg["bar_minutes"]),
                "status": cfg.get("status", "REVIEW"),
                "validation": cfg.get("validation", {}),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed config for {pair_key}: {e}")

    logger.info(f"Loaded Kalman configs for {len(configs)} pair(s) from {path}")
    for pair_key, cfg in configs.items():
        logger.info(
            f"  {pair_key}: status={cfg['status']}, "
            f"β₀={cfg['beta0']:.6f}, bar_min={cfg['bar_minutes']}"
        )
    return configs


# ── Single-step Kalman update ────────────────────────────────────────────────

def filter_update(
    state: np.ndarray,
    P: np.ndarray,
    x_t: float,
    y_t: float,
    Q: np.ndarray,
    R: float,
    predict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Single causal Kalman filter step.

    Parameters
    ----------
    state : (2,) array — current state [alpha, beta]
    P     : (2,2) array — current state covariance
    x_t   : float — predictor value (price of symbol B at time t)
    y_t   : float — observation value (price of symbol A at time t)
    Q     : (2,2) array — transition (process) noise covariance (FROZEN from EM)
    R     : float — observation noise variance (FROZEN from EM)
    predict : bool — run the predict step (P += Q) before correcting.
        Must be False for the very FIRST observation: pykalman's filter()
        uses initial_state_mean/covariance directly as the t=0 prior (no
        transition, no +Q) and corrects with observation 0. Passing
        predict=True there double-applies a transition that never
        happened upstream and the two implementations diverge from t=0.

    Returns
    -------
    state_new : (2,) array — updated state
    P_new     : (2,2) array — updated covariance

    Notes
    -----
    - F = I (random walk on [alpha, beta])
    - H = [1, x_t] (observation matrix)
    - predict: P_pred = P + Q (skipped when predict=False)
    - innovation: v = y_t - H @ state
    - S = H @ P_pred @ H^T + R
    - K = P_pred @ H^T / S
    - update: state_new = state + K * v;  P_new = (I - K @ H) @ P_pred
    """
    state = np.asarray(state, dtype=np.float64).copy()
    P = np.asarray(P, dtype=np.float64).copy()
    Q = np.asarray(Q, dtype=np.float64)

    # Observation vector
    H = np.array([[1.0, x_t]])  # (1, 2)

    # Predict
    P_pred = P + Q if predict else P

    # Innovation
    v = (y_t - (H @ state)).item()  # scalar

    # Innovation covariance
    S = (H @ P_pred @ H.T).item() + R  # scalar

    # Kalman gain
    K = (P_pred @ H.T) / S  # (2, 1)

    # Update
    state_new = state + K.ravel() * v
    P_new = (np.eye(2) - K @ H) @ P_pred

    return state_new, P_new


# ── Full-window filter pass ──────────────────────────────────────────────────

def current_beta(
    cfg: dict,
    series_a: pd.Series,
    series_b: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Run the Kalman filter across a price window and return dynamic β.

    Warm-starts from cfg.alpha0/beta0/P0 and steps filter_update across
    aligned bars using FROZEN cfg.Q, cfg.R (no EM, no recalibration).

    Parameters
    ----------
    cfg      : dict from load_config (must have alpha0, beta0, P0, Q, R)
    series_a : pd.Series — price bars for symbol A (time-indexed)
    series_b : pd.Series — price bars for symbol B (time-indexed)

    Returns
    -------
    alpha_t : float — latest filtered intercept
    beta_t  : float — latest filtered hedge ratio
    dyn_spread : pd.Series — dynamic spread A - (alpha_t + beta_t * B) at each step
    """
    # Align series
    px = pd.DataFrame({"A": series_a, "B": series_b}).dropna()

    if len(px) < 2:
        raise ValueError("Need at least 2 overlapping bars for Kalman filtering")

    state = np.array([cfg["alpha0"], cfg["beta0"]], dtype=np.float64)
    P = np.array(cfg["P0"], dtype=np.float64)
    Q = np.array(cfg["Q"], dtype=np.float64)
    R = float(cfg["R"])

    alphas = np.zeros(len(px))
    betas = np.zeros(len(px))

    for i in range(len(px)):
        x_t = float(px["B"].iloc[i])
        y_t = float(px["A"].iloc[i])
        state, P = filter_update(state, P, x_t, y_t, Q, R, predict=(i > 0))
        alphas[i] = state[0]
        betas[i] = state[1]

    # Dynamic spread: A - (alpha_t + beta_t * B)
    dyn_spread = pd.Series(
        px["A"].values - (alphas + betas * px["B"].values),
        index=px.index,
        name="dyn_spread",
    )

    return float(state[0]), float(state[1]), dyn_spread
