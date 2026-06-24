"""
conformal.py — Two-piece modal split-conformal predictor.

Implements Algorithm 1 from "Asymmetric Conformal Prediction via Modal
Regression" (Rubio & Steel framework), adapted for 1-step spread innovations.

Key design decisions:
  - 1-step (not multi-step) so calibration points stay exchangeable.
  - Chronological train|calibration|holdout split (never random).
  - Per-side method-of-moments for σ̂₁/σ̂₂ (v1; MLE swappable later).
  - σ̂ floors prevent collapse to 0 (mirrors R_FLOOR in kalman_calibration).
  - REVIEW status for short series or out-of-band coverage — cap at LOW tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import statsmodels.api as sm


# ── Config ───────────────────────────────────────────────────────────────────

# Minimum fraction of full residual std for each per-side σ̂.
# Prevents σ̂ → 0 which would make r blow up or nonconformity scores explode.
SIGMA_FLOOR_FRAC = 0.05

# Minimum number of points required in any partition (train/cal/holdout).
MIN_PARTITION_SIZE = 5

# Minimum total series length to attempt fitting.
MIN_SERIES_LENGTH = 30

# Coverage acceptance band for holdout validation.
COVERAGE_LO = 0.85
COVERAGE_HI = 0.95


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConformalFit:
    """Result of fit_two_piece_cp."""
    intercept: float       # ĉ in μ̂(x) = ĉ + φ̂·x
    slope: float           # φ̂
    sigma1: float          # left scale (negative residuals)
    sigma2: float          # right scale (non-negative residuals)
    q: float               # conformal quantile from calibration
    r: float               # σ̂₂/σ̂₁ — the real asymmetry ratio
    n_train: int
    n_cal: int
    n_holdout: int
    coverage: float        # holdout coverage
    status: str            # "PASS" or "REVIEW"

    def mu_fn(self, x: float) -> float:
        """Conditional mode: μ̂(x) = intercept + slope · x."""
        return self.intercept + self.slope * x


def _safe_review_fit() -> ConformalFit:
    """Return a safe, conservative ConformalFit for degenerate cases."""
    return ConformalFit(
        intercept=0.0,
        slope=0.0,
        sigma1=1.0,
        sigma2=1.0,
        q=1.0,
        r=1.0,
        n_train=0,
        n_cal=0,
        n_holdout=0,
        coverage=0.0,
        status="REVIEW",
    )


# ── Core algorithm ──────────────────────────────────────────────────────────

def fit_two_piece_cp(
    spread: np.ndarray | "pd.Series",
    alpha: float = 0.10,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> ConformalFit:
    """Fit a two-piece split-conformal predictor on a spread series.

    Parameters
    ----------
    spread : array-like
        The OLS spread series (time-ordered, no NaNs).
    alpha : float
        Miscoverage level. Default 0.10 → 90% target coverage.
    train_frac, cal_frac : float
        Chronological split fractions. holdout_frac = 1 - train - cal.

    Returns
    -------
    ConformalFit
        Contains μ̂, σ̂₁, σ̂₂, q, r, coverage, status.
        Returns a safe REVIEW fit for degenerate/short series.

    Notes
    -----
    - Response: y_t = spread_t.  Predictor: x_t = spread_{t-1}.
    - 1-step innovation preserves exchangeability for conformal guarantee.
    - Chronological split: train → calibration → holdout. Never random.
    """
    s = np.asarray(spread, dtype=np.float64)

    # Guard: too short
    if len(s) < MIN_SERIES_LENGTH:
        return _safe_review_fit()

    # Build 1-step pairs: y_t = s[t], x_t = s[t-1]
    y = s[1:]
    x = s[:-1]
    n = len(y)

    if n < MIN_SERIES_LENGTH:
        return _safe_review_fit()

    # ── Chronological split ──────────────────────────────────────────
    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)
    n_holdout = n - n_train - n_cal

    if n_train < MIN_PARTITION_SIZE or n_cal < MIN_PARTITION_SIZE or n_holdout < MIN_PARTITION_SIZE:
        return _safe_review_fit()

    x_train, y_train = x[:n_train], y[:n_train]
    x_cal, y_cal = x[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
    x_hold, y_hold = x[n_train + n_cal:], y[n_train + n_cal:]

    # ── 1. Fit μ̂(x) = ĉ + φ̂·x on TRAIN only ────────────────────────
    # Guard: constant predictor → OLS degenerates
    if np.std(x_train) < 1e-12:
        return _safe_review_fit()

    X_train = sm.add_constant(x_train)
    if X_train.ndim == 1 or X_train.shape[1] < 2:
        return _safe_review_fit()

    ols = sm.OLS(y_train, X_train).fit()
    intercept, slope = float(ols.params[0]), float(ols.params[1])

    # ── 2. Two-piece scales from TRAIN residuals ─────────────────────
    residuals_train = y_train - (intercept + slope * x_train)
    res_std = float(np.std(residuals_train))
    floor = SIGMA_FLOOR_FRAC * res_std if res_std > 0 else 1e-8

    neg_mask = residuals_train < 0
    pos_mask = ~neg_mask  # includes zero

    if neg_mask.sum() < 2 or pos_mask.sum() < 2:
        # Not enough residuals on one side — degenerate
        return ConformalFit(
            intercept=intercept, slope=slope,
            sigma1=max(res_std, floor), sigma2=max(res_std, floor),
            q=1.0, r=1.0,
            n_train=n_train, n_cal=n_cal, n_holdout=n_holdout,
            coverage=0.0, status="REVIEW",
        )

    sigma1_raw = float(np.std(residuals_train[neg_mask]))
    sigma2_raw = float(np.std(residuals_train[pos_mask]))
    sigma1 = max(sigma1_raw, floor)
    sigma2 = max(sigma2_raw, floor)
    r = sigma2 / sigma1

    # ── 3. Nonconformity scores on CALIBRATION ───────────────────────
    residuals_cal = y_cal - (intercept + slope * x_cal)
    scales_cal = np.where(residuals_cal < 0, sigma1, sigma2)
    scores = np.abs(residuals_cal) / scales_cal

    # ── 4. Conformal quantile ────────────────────────────────────────
    # q = ⌈(1-α)·(n_cal+1)⌉-th order statistic
    k = int(np.ceil((1 - alpha) * (n_cal + 1)))
    k = min(k, n_cal)  # clamp to array bounds
    sorted_scores = np.sort(scores)
    q = float(sorted_scores[k - 1])  # 0-indexed

    # ── 5. Holdout coverage ──────────────────────────────────────────
    coverage, status = validate_coverage_arrays(
        intercept, slope, sigma1, sigma2, q, x_hold, y_hold
    )

    return ConformalFit(
        intercept=intercept,
        slope=slope,
        sigma1=sigma1,
        sigma2=sigma2,
        q=q,
        r=round(r, 4),
        n_train=n_train,
        n_cal=n_cal,
        n_holdout=n_holdout,
        coverage=round(coverage, 4),
        status=status,
    )


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_interval(
    fit: ConformalFit, x_new: float
) -> tuple[float, float, float, float, float]:
    """Predict with the fitted two-piece conformal model.

    Returns
    -------
    (mu, L, U, width, r)
        mu    : conditional mode μ̂(x_new)
        L     : lower bound = mu - q·σ̂₁
        U     : upper bound = mu + q·σ̂₂
        width : q·(σ̂₁ + σ̂₂)
        r     : σ̂₂/σ̂₁
    """
    mu = fit.mu_fn(x_new)
    L = mu - fit.q * fit.sigma1
    U = mu + fit.q * fit.sigma2
    width = fit.q * (fit.sigma1 + fit.sigma2)
    return mu, L, U, width, fit.r


# ── Validation ───────────────────────────────────────────────────────────────

def validate_coverage_arrays(
    intercept: float,
    slope: float,
    sigma1: float,
    sigma2: float,
    q: float,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> tuple[float, str]:
    """Compute holdout coverage and return (coverage, status).

    PASS if coverage ∈ [0.85, 0.95], else REVIEW.
    """
    mu = intercept + slope * x_holdout
    L = mu - q * sigma1
    U = mu + q * sigma2
    covered = (y_holdout >= L) & (y_holdout <= U)
    coverage = float(np.mean(covered))
    status = "PASS" if COVERAGE_LO <= coverage <= COVERAGE_HI else "REVIEW"
    return coverage, status


def validate_coverage(
    fit: ConformalFit,
    holdout_y: np.ndarray,
    holdout_x: np.ndarray,
) -> tuple[float, str]:
    """Convenience wrapper using a ConformalFit object."""
    return validate_coverage_arrays(
        fit.intercept, fit.slope, fit.sigma1, fit.sigma2, fit.q,
        holdout_x, holdout_y,
    )
