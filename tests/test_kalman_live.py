"""
tests/test_kalman_live.py — Tests for kalman_live.py.

Tests:
  1. Equivalence: filter_update stepped over a window matches pykalman
     KalmanFilter.filter() to ~1e-8 — proves the incremental step matches
     the offline calibrator.
  2. With Q=0 the filter stays at the warm-start β (frozen state).
  3. Missing/partial config falls back cleanly with a logged warning.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalman_live import load_config, filter_update, current_beta


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_data(n: int = 200, seed: int = 42):
    """Generate synthetic price series A, B with known relationship."""
    rng = np.random.default_rng(seed)
    B = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    true_alpha = 5.0
    true_beta = 0.8
    noise = rng.normal(0, 0.3, n)
    A = true_alpha + true_beta * B + noise
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.Series(A, index=idx, name="A"), pd.Series(B, index=idx, name="B")


def _make_config(alpha0=5.0, beta0=0.8, P0=None, Q=None, R=0.1,
                 bar_minutes=15, status="PASS"):
    """Build a config dict matching kalman_config.json format."""
    if P0 is None:
        P0 = (np.eye(2) * 0.01).tolist()
    if Q is None:
        Q = (np.eye(2) * 1e-5).tolist()
    return {
        "pair": "A/B",
        "alpha0": alpha0,
        "beta0": beta0,
        "P0": P0 if isinstance(P0, list) else P0.tolist(),
        "Q": Q if isinstance(Q, list) else Q.tolist(),
        "R": R,
        "bar_minutes": bar_minutes,
        "status": status,
        "validation": {
            "adf_pvalue": 0.01,
            "half_life_min": 200.0,
            "n_val_obs": 100,
        },
    }


# ── Test 1: Equivalence with pykalman ────────────────────────────────────────

def test_equivalence_with_pykalman():
    """filter_update stepped over a window matches pykalman KalmanFilter.filter()
    to ~1e-8 — proves the incremental step matches the offline calibrator."""
    try:
        from pykalman import KalmanFilter
    except ImportError:
        pytest.skip("pykalman not installed")

    A, B = _make_synthetic_data(n=200)

    alpha0, beta0 = 5.0, 0.8
    P0 = np.eye(2) * 0.01
    Q = np.eye(2) * 1e-5
    R = 0.1

    # ── pykalman batch filter ────────────────────────────────────────
    n = len(A)
    obs_matrices = np.ones((n, 1, 2))
    obs_matrices[:, 0, 1] = B.values

    kf = KalmanFilter(
        n_dim_state=2, n_dim_obs=1,
        transition_matrices=np.eye(2),
        observation_matrices=obs_matrices,
        initial_state_mean=[alpha0, beta0],
        initial_state_covariance=P0,
        transition_covariance=Q,
        observation_covariance=np.array([[R]]),
    )
    means_pykalman, covs_pykalman = kf.filter(A.values)

    # ── Incremental filter_update ────────────────────────────────────
    state = np.array([alpha0, beta0])
    P = P0.copy()
    means_live = np.zeros((n, 2))
    covs_live = np.zeros((n, 2, 2))

    for i in range(n):
        state, P = filter_update(
            state, P, float(B.iloc[i]), float(A.iloc[i]), Q, R, predict=(i > 0)
        )
        means_live[i] = state
        covs_live[i] = P

    # ── Compare: should match to ~1e-8 precision ───────────────────
    # pykalman's filter() uses initial_state_mean/covariance directly as
    # the t=0 prior (no transition, no +Q) and corrects with observation 0.
    # filter_update must skip its predict step on the first observation
    # (predict=False above) to align with that convention; once aligned,
    # both implementations agree to numerical precision.
    np.testing.assert_allclose(
        means_live, means_pykalman, atol=1e-8,
        err_msg="filter_update means diverge from pykalman"
    )
    np.testing.assert_allclose(
        covs_live, covs_pykalman, atol=1e-8,
        err_msg="filter_update covariances diverge from pykalman"
    )

    print(f"\n  Equivalence test: max mean diff = {np.max(np.abs(means_live - means_pykalman)):.2e}")
    print(f"  Equivalence test: max cov diff = {np.max(np.abs(covs_live - covs_pykalman)):.2e}")


# ── Test 2: Q=0 freezes state ───────────────────────────────────────────────

def test_q_zero_freezes_beta():
    """With Q=0 (no process noise), the filter stays at the warm-start β."""
    A, B = _make_synthetic_data(n=100)

    alpha0, beta0 = 5.0, 0.8
    P0 = np.zeros((2, 2))  # zero initial covariance
    Q = np.zeros((2, 2))   # zero process noise
    R = 0.1

    state = np.array([alpha0, beta0])
    P = P0.copy()

    for i in range(len(A)):
        state, P = filter_update(state, P, float(B.iloc[i]), float(A.iloc[i]), Q, R)

    # With Q=0 and P0=0, the Kalman gain is always zero → state never moves
    assert abs(state[0] - alpha0) < 1e-12, f"alpha drifted: {state[0]} vs {alpha0}"
    assert abs(state[1] - beta0) < 1e-12, f"beta drifted: {state[1]} vs {beta0}"

    print(f"\n  Q=0 freeze test: alpha={state[0]:.10f} (expected {alpha0})")
    print(f"  Q=0 freeze test: beta={state[1]:.10f} (expected {beta0})")


# ── Test 3: current_beta produces dynamic spread ────────────────────────────

def test_current_beta():
    """current_beta() returns dynamic α, β and spread with correct shape."""
    A, B = _make_synthetic_data(n=200)
    cfg = _make_config()

    alpha_t, beta_t, dyn_spread = current_beta(cfg, A, B)

    assert isinstance(alpha_t, float), f"alpha_t should be float, got {type(alpha_t)}"
    assert isinstance(beta_t, float), f"beta_t should be float, got {type(beta_t)}"
    assert len(dyn_spread) == len(A), "dyn_spread length mismatch"
    assert not dyn_spread.isna().any(), "dyn_spread contains NaN"

    # β should be close to true value (0.8) after 200 steps
    assert abs(beta_t - 0.8) < 0.1, f"beta_t={beta_t} too far from true 0.8"

    print(f"\n  current_beta test: alpha_t={alpha_t:.4f}, beta_t={beta_t:.4f}")
    print(f"  spread: mean={dyn_spread.mean():.4f}, std={dyn_spread.std():.4f}")


# ── Test 4: Missing config → empty dict, no crash ───────────────────────────

def test_missing_config():
    """load_config with nonexistent path returns empty dict + logs warning."""
    configs = load_config("/nonexistent/kalman_config.json")
    assert configs == {}, f"Expected empty dict, got {configs}"
    print("\n  Missing config test: returns empty dict (no crash)")


def test_invalid_json_config():
    """load_config with invalid JSON returns empty dict + logs warning."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not json {{{")
        f.flush()
        configs = load_config(f.name)
    assert configs == {}, f"Expected empty dict, got {configs}"
    print("\n  Invalid JSON config test: returns empty dict (no crash)")


def test_partial_config():
    """load_config with partial/malformed entries skips bad entries cleanly."""
    config_data = {
        "GOOD/PAIR": _make_config(),
        "BAD/PAIR": {"alpha0": "not a number"},  # malformed
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        configs = load_config(f.name)

    assert "GOOD/PAIR" in configs, "Good pair should load"
    assert "BAD/PAIR" not in configs, "Bad pair should be skipped"
    print(f"\n  Partial config test: loaded {len(configs)} pair(s), skipped malformed")


# ── Test 5: load_config parses numpy arrays correctly ────────────────────────

def test_config_numpy_arrays():
    """Verify P0, Q are loaded as numpy arrays with correct shape."""
    config_data = {
        "TEST/PAIR": _make_config(),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()
        configs = load_config(f.name)

    cfg = configs["TEST/PAIR"]
    assert isinstance(cfg["P0"], np.ndarray), f"P0 should be ndarray"
    assert cfg["P0"].shape == (2, 2), f"P0 should be 2x2"
    assert isinstance(cfg["Q"], np.ndarray), f"Q should be ndarray"
    assert cfg["Q"].shape == (2, 2), f"Q should be 2x2"

    print(f"\n  Config arrays test: P0 shape={cfg['P0'].shape}, Q shape={cfg['Q'].shape}")
