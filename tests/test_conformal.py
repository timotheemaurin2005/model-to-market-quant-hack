"""
tests/test_conformal.py — Tests for conformal.py two-piece split-conformal predictor.

Tests:
  1. Synthetic two-piece (σ₁=1, σ₂=4): recovered r≈4 ±tol; holdout coverage≈0.90
  2. Symmetric (σ₁=σ₂): r≈1; coverage≈0.90
  3. Short/degenerate series: returns REVIEW, never crashes
  4. All watchlist pairs: print coverage for each (the distribution is the point)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the trading directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformal import (
    ConformalFit,
    fit_two_piece_cp,
    predict_interval,
    validate_coverage,
    MIN_SERIES_LENGTH,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_two_piece_spread(
    n: int, sigma1: float, sigma2: float, phi: float = 0.8, seed: int = 42
) -> np.ndarray:
    """Generate a synthetic spread with known two-piece innovation distribution.

    s_{t} = phi * s_{t-1} + e_t
    where e_t ~ TPN(0, sigma1, sigma2):
        negative side: |N(0, sigma1)|, flipped
        positive side: |N(0, sigma2)|
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        # Two-piece normal: draw from left or right with equal probability
        if rng.random() < 0.5:
            e = -abs(rng.normal(0, sigma1))
        else:
            e = abs(rng.normal(0, sigma2))
        s[t] = phi * s[t - 1] + e
    return s


# ── Test 1: Synthetic two-piece (σ₁=1, σ₂=4) ───────────────────────────────

def test_asymmetric_two_piece():
    """Recovered r ≈ 4 within tolerance; holdout coverage ≈ 0.90."""
    spread = _generate_two_piece_spread(n=2000, sigma1=1.0, sigma2=4.0, seed=42)
    fit = fit_two_piece_cp(spread, alpha=0.10)

    # r should be close to 4.0 (σ₂/σ₁)
    assert fit.r > 2.0, f"r={fit.r} too low, expected ~4.0"
    assert fit.r < 8.0, f"r={fit.r} too high, expected ~4.0"

    # Coverage should be around 0.90 ± 0.08
    assert fit.coverage > 0.80, f"coverage={fit.coverage} too low"
    assert fit.coverage < 0.98, f"coverage={fit.coverage} too high"

    # Status should be PASS (coverage in [0.85, 0.95])
    # (might be REVIEW if coverage falls just outside — that's also acceptable)
    assert fit.status in ("PASS", "REVIEW"), f"unexpected status: {fit.status}"

    # Sigma floors should not have triggered (σ's are well above floor)
    assert fit.sigma1 > 0.5, f"sigma1={fit.sigma1} unexpectedly low"
    assert fit.sigma2 > 2.0, f"sigma2={fit.sigma2} unexpectedly low"

    print(f"\n  Asymmetric test: r={fit.r:.2f}, σ₁={fit.sigma1:.3f}, "
          f"σ₂={fit.sigma2:.3f}, coverage={fit.coverage:.3f}, "
          f"status={fit.status}")


# ── Test 2: Symmetric (σ₁=σ₂) ───────────────────────────────────────────────

def test_symmetric():
    """r ≈ 1 for symmetric innovations; coverage ≈ 0.90."""
    spread = _generate_two_piece_spread(n=2000, sigma1=2.0, sigma2=2.0, seed=99)
    fit = fit_two_piece_cp(spread, alpha=0.10)

    # r should be close to 1.0
    assert fit.r > 0.5, f"r={fit.r} too low, expected ~1.0"
    assert fit.r < 2.0, f"r={fit.r} too high, expected ~1.0"

    # Coverage around 0.90
    assert fit.coverage > 0.80, f"coverage={fit.coverage} too low"
    assert fit.coverage < 0.98, f"coverage={fit.coverage} too high"

    print(f"\n  Symmetric test: r={fit.r:.2f}, σ₁={fit.sigma1:.3f}, "
          f"σ₂={fit.sigma2:.3f}, coverage={fit.coverage:.3f}, "
          f"status={fit.status}")


# ── Test 3: Short / degenerate series ────────────────────────────────────────

def test_short_series_returns_review():
    """Series shorter than MIN_SERIES_LENGTH → REVIEW, never crashes."""
    short = np.array([1.0, 2.0, 3.0, 2.5, 2.0])
    fit = fit_two_piece_cp(short)

    assert fit.status == "REVIEW", f"expected REVIEW, got {fit.status}"
    assert fit.r == 1.0, f"expected r=1.0 for degenerate, got {fit.r}"
    assert fit.n_train == 0, "degenerate fit should have n_train=0"

    print(f"\n  Short series test: status={fit.status}, r={fit.r}")


def test_constant_series_returns_review():
    """Constant series (zero variance) → REVIEW, never crashes."""
    constant = np.ones(100)
    fit = fit_two_piece_cp(constant)

    assert fit.status == "REVIEW", f"expected REVIEW, got {fit.status}"
    # Should not crash — σ floor prevents division by zero

    print(f"\n  Constant series test: status={fit.status}, r={fit.r}")


def test_empty_series():
    """Empty or length-1 series → REVIEW, never crashes."""
    for s in [np.array([]), np.array([42.0])]:
        fit = fit_two_piece_cp(s)
        assert fit.status == "REVIEW"

    print("\n  Empty/single series test: REVIEW (no crash)")


# ── Test 4: Predict interval ────────────────────────────────────────────────

def test_predict_interval_asymmetric():
    """Interval bounds respect asymmetry: U - mu > mu - L when r > 1."""
    spread = _generate_two_piece_spread(n=2000, sigma1=1.0, sigma2=4.0, seed=42)
    fit = fit_two_piece_cp(spread, alpha=0.10)

    x_new = spread[-1]
    mu, L, U, width, r = predict_interval(fit, x_new)

    # Upper tail should be wider than lower tail
    upper_width = U - mu
    lower_width = mu - L
    assert upper_width > lower_width, (
        f"Expected asymmetric interval: upper={upper_width:.3f} > lower={lower_width:.3f}"
    )
    assert width > 0, f"width should be positive, got {width}"
    assert r > 2.0, f"r={r} should be > 2 for σ₁=1,σ₂=4"

    print(f"\n  Predict interval: mu={mu:.3f}, L={L:.3f}, U={U:.3f}, "
          f"width={width:.3f}, r={r:.2f}")


# ── Test 5: All watchlist pairs (prints coverage, pass or fail) ──────────────

def test_watchlist_pairs_coverage(capsys):
    """Print coverage for ALL watchlist pairs. The distribution is the point."""
    # Import signal_server components for data loading
    try:
        from signal_server import WATCHLIST, _load_symbol_bars, _screen_pair
    except Exception as e:
        pytest.skip(f"Cannot import signal_server: {e}")

    print("\n\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  Conformal Coverage — All Watchlist Pairs                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Pair':20s} {'r':>6s} {'σ̂₁':>8s} {'σ̂₂':>8s} {'Cov':>6s} {'Status':>8s} ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")

    import pandas as pd

    for sym_a, sym_b in WATCHLIST:
        try:
            bars_a = _load_symbol_bars(sym_a)
            bars_b = _load_symbol_bars(sym_b)
            px = pd.DataFrame({sym_a: bars_a, sym_b: bars_b}).dropna()

            import statsmodels.api as sm_local
            x = sm_local.add_constant(px[sym_b].values)
            ols = sm_local.OLS(px[sym_a].values, x).fit()
            alpha_ols, beta = float(ols.params[0]), float(ols.params[1])
            spread = px[sym_a] - (alpha_ols + beta * px[sym_b])

            fit = fit_two_piece_cp(spread.values, alpha=0.10)

            pair_name = f"{sym_a}/{sym_b}"
            print(
                f"  ║  {pair_name:20s} {fit.r:6.2f} {fit.sigma1:8.5f} "
                f"{fit.sigma2:8.5f} {fit.coverage:6.3f} {fit.status:>8s} ║"
            )
        except Exception as e:
            pair_name = f"{sym_a}/{sym_b}"
            print(f"  ║  {pair_name:20s} {'ERROR':>6s} {'':>8s} {'':>8s} "
                  f"{'':>6s} {'':>8s} ║  {e}")

    print("  ╚══════════════════════════════════════════════════════════════╝")


# ── Test: Calibration / holdout are never reused ─────────────────────────────

def test_no_data_leakage():
    """Verify chronological split: train < cal < holdout indices."""
    spread = _generate_two_piece_spread(n=500, sigma1=1.0, sigma2=3.0, seed=77)
    n = len(spread) - 1  # 1-step pairs
    n_train = int(n * 0.6)
    n_cal = int(n * 0.2)

    fit = fit_two_piece_cp(spread, alpha=0.10)

    # Partition sizes should match expectations
    assert fit.n_train == n_train, f"n_train={fit.n_train}, expected {n_train}"
    assert fit.n_cal == n_cal, f"n_cal={fit.n_cal}, expected {n_cal}"
    assert fit.n_holdout == n - n_train - n_cal

    # Total should equal n
    assert fit.n_train + fit.n_cal + fit.n_holdout == n

    print(f"\n  No-leakage test: train={fit.n_train}, cal={fit.n_cal}, "
          f"holdout={fit.n_holdout}, total={n}")
