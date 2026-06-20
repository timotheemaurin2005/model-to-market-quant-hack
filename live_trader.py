"""
live_trader.py - live pairs mean-reversion loop for Model to Market (hardened).

Runs ON the VPS, in-process. Pulls 15-min bars from MT5, screens cointegration +
half-life every cycle, and trades the FX pairs that currently qualify. Routes all
orders through mt5_executor (DRY_RUN-gated).

HARDENING (vs the first draft):
  * NETTING-SAFE: the account is Netting (one position per symbol). Two held pairs
    that share a symbol would be merged by the broker, breaking per-pair exit.
    So we forbid entering a pair whose symbol is already used by a held pair.
  * FROZEN BASELINE EXIT: on entry we freeze alpha/beta/mu/sigma in state.json and
    judge exits against those, never a re-fit (no "moving goalposts").
  * STATE BY RECORD, NOT COMMENT: holdings live in state.json, reconciled against
    magic-tagged positions by symbol. Broker comment-mangling can't orphan a leg.
  * STOPS: divergence stop (frozen |Z| > 3.5 -> cointegration broken, cut) and
    time stop (held > 3x half-life, capped at one round).

Workflow: edit on your Mac -> git push -> git pull on the VPS.
Deps: MetaTrader5, numpy, pandas, statsmodels   (pip install pandas statsmodels)
"""

import os
import json
import time
import itertools
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import MetaTrader5 as mt5

import mt5_executor as ex   # connect/size/guardrail/place_order + the DRY_RUN switch

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FX_UNIVERSE = ["AUDUSD", "EURCHF", "EURGBP", "EURUSD",
               "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]
PAIRS = list(itertools.combinations(FX_UNIVERSE, 2))     # 28 candidate pairs

ENTRY_Z = 2.0
EXIT_Z  = 0.5
DIVERGENCE_Z = 3.5            # hard stop: frozen |Z| past this => cut (relationship broke)
MAX_HOLD_HL  = 3.0           # time stop: exit after this many half-lives...
ABS_MAX_HOLD_MIN = 1440      # ...but never hold longer than one round (1 day)

N_BARS  = 3000
ADF_PMAX = 0.05
HL_MIN_MIN, HL_MAX_MIN = 120, 1440
BAR_MIN = 15

MARGIN_PER_PAIR      = 15_000
MAX_CONCURRENT_PAIRS = 6
STATE_FILE = "state.json"


# ---------------------------------------------------------------------------
# STATE  (source of truth for holdings + frozen entry baselines)
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def reconcile(state):
    """Drop pairs whose legs are no longer open at the broker (live only).
    In DRY_RUN nothing actually opens, so we trust state.json so exit paths
    can still be exercised across runs."""
    if ex.DRY_RUN:
        return state
    open_syms = {p.symbol for p in (mt5.positions_get() or []) if p.magic == ex.MAGIC}
    alive = {}
    for tag, rec in state.items():
        if any(s in open_syms for s in rec["symbols"]):
            alive[tag] = rec
        else:
            print(f"   [state] {tag} no longer open at broker -> dropped")
    return alive


# ---------------------------------------------------------------------------
# DATA / STATS
# ---------------------------------------------------------------------------
def get_closes(symbol, n=N_BARS):
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["t"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("t")["close"]


def half_life(spread):
    s = spread.dropna()
    lag = s.shift(1)
    delta = (s - lag).dropna()
    lag = lag.loc[delta.index]
    lam = sm.OLS(delta.values, sm.add_constant(lag.values)).fit().params[1]
    if lam >= 0 or (1 + lam) <= 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + lam))


def fit_pair(a_close, b_close):
    df = pd.concat([a_close, b_close], axis=1).dropna()
    df.columns = ["a", "b"]
    if len(df) < 500:
        return None
    res = sm.OLS(df["a"].values, sm.add_constant(df["b"].values)).fit()
    alpha, beta = float(res.params[0]), float(res.params[1])
    spread = df["a"] - (alpha + beta * df["b"])
    mu, sd = float(spread.mean()), float(spread.std())
    z = float((spread.iloc[-1] - mu) / sd) if sd > 0 else np.nan
    adf_p = float(adfuller(spread.values, autolag="AIC")[1])
    hl_min = half_life(spread) * BAR_MIN
    return dict(alpha=alpha, beta=beta, mu=mu, sd=sd, z=z, adf_p=adf_p, hl_min=hl_min)


def passes_screen(f):
    return (f["adf_p"] < ADF_PMAX
            and np.isfinite(f["hl_min"])
            and HL_MIN_MIN <= f["hl_min"] <= HL_MAX_MIN)


def frozen_z(rec, a_close, b_close):
    """Z of the current spread against the FROZEN entry baseline."""
    a_now, b_now = float(a_close.iloc[-1]), float(b_close.iloc[-1])
    spread_now = a_now - (rec["alpha"] + rec["beta"] * b_now)
    return (spread_now - rec["mu"]) / rec["sigma"] if rec["sigma"] else float("nan")


# ---------------------------------------------------------------------------
# LEG SIZING (beta-weighted, contract/currency-correct via tick value)
# ---------------------------------------------------------------------------
def price_value_per_lot(symbol):
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_size == 0:
        return None
    return info.trade_tick_value / info.trade_tick_size


def build_legs(a, b, f):
    z = f["z"]
    spread_dir = -1 if z > 0 else +1
    dir_a = spread_dir
    dir_b = -spread_dir * (1 if f["beta"] >= 0 else -1)
    lots_a, _ = ex.size_by_margin(a, MARGIN_PER_PAIR, dir_a)
    if lots_a is None:
        return None
    pv_a, pv_b = price_value_per_lot(a), price_value_per_lot(b)
    if not pv_a or not pv_b:
        return None
    info_b = mt5.symbol_info(b)
    raw_b = lots_a * abs(f["beta"]) * pv_a / pv_b
    lots_b = round(round(raw_b / info_b.volume_step) * info_b.volume_step, 8)
    lots_b = max(info_b.volume_min, min(lots_b, info_b.volume_max))
    return [(a, lots_a, dir_a), (b, lots_b, dir_b)]


def close_pair(tag, rec):
    """Netting account: an opposite-direction deal of equal volume closes the leg."""
    for p in (mt5.positions_get() or []):
        if p.magic == ex.MAGIC and p.symbol in rec["symbols"]:
            close_dir = -1 if p.type == mt5.POSITION_TYPE_BUY else +1
            ex.place_order(p.symbol, p.volume, close_dir, comment=tag + "-exit")


# ---------------------------------------------------------------------------
# ONE CYCLE
# ---------------------------------------------------------------------------
def run_cycle():
    state = reconcile(load_state())
    now = datetime.now(timezone.utc)
    print(f"\n=== cycle {now:%Y-%m-%d %H:%M} UTC | holding {len(state)}: "
          f"{sorted(state) or 'none'} ===")

    closes = {s: get_closes(s) for s in FX_UNIVERSE}

    # ---- EXIT pass: held pairs, judged on FROZEN baseline ----
    for tag, rec in list(state.items()):
        a, b = rec["symbols"]
        if closes.get(a) is None or closes.get(b) is None:
            continue
        z = frozen_z(rec, closes[a], closes[b])
        entry_t = datetime.fromisoformat(rec["entry_time"])
        held_min = (now - entry_t).total_seconds() / 60.0
        max_hold = min(MAX_HOLD_HL * rec["half_life_min"], ABS_MAX_HOLD_MIN)

        reason = None
        if abs(z) > DIVERGENCE_Z:
            reason = f"DIVERGENCE STOP frozenZ={z:+.2f} (> {DIVERGENCE_Z})"
        elif abs(z) < EXIT_Z:
            reason = f"reverted frozenZ={z:+.2f} (< {EXIT_Z})"
        elif held_min > max_hold:
            reason = f"TIME STOP held {held_min:.0f}m (> {max_hold:.0f}m)"

        print(f"  [HELD] {tag:16} frozenZ={z:+.2f} held={held_min:.0f}m"
              + (f"  -> EXIT: {reason}" if reason else "  -> hold"))
        if reason:
            close_pair(tag, rec)
            del state[tag]

    # ---- ENTRY pass: flat pairs, fresh fit + screen ----
    held_syms = {s for r in state.values() for s in r["symbols"]}
    scan = []
    for a, b in PAIRS:
        if tag_of(a, b) in state:
            continue
        if closes.get(a) is None or closes.get(b) is None:
            continue
        f = fit_pair(closes[a], closes[b])
        if f is None:
            continue
        f.update(a=a, b=b, tag=tag_of(a, b), screened=passes_screen(f))
        scan.append(f)

    n_pass = sum(r["screened"] for r in scan)
    show = sorted([r for r in scan if r["screened"] or r["adf_p"] < 0.10],
                  key=lambda r: -abs(r["z"]))
    print(f"{n_pass}/{len(scan)} flat pairs pass screen | PASS + near-miss:")
    for r in show:
        print(f"  {r['tag']:16} Z={r['z']:+.2f} beta={r['beta']:+.4f} "
              f"adf_p={r['adf_p']:.4f} hl={r['hl_min']:.0f}m "
              f"{'PASS' if r['screened'] else 'fail'}")

    entered = 0
    for r in sorted(scan, key=lambda r: -abs(r["z"])):
        if not (r["screened"] and abs(r["z"]) > ENTRY_Z):
            continue
        a, b, tag = r["a"], r["b"], r["tag"]
        if len(state) + entered >= MAX_CONCURRENT_PAIRS:
            print(f"   -> {tag} skipped (at {MAX_CONCURRENT_PAIRS}-pair cap)"); continue
        if a in held_syms or b in held_syms:
            print(f"   -> {tag} skipped (symbol overlap; netting account)"); continue
        legs = build_legs(a, b, r)
        if not legs:
            print(f"   -> {tag} entry aborted (sizing)"); continue
        ok, reason, _ = ex.check_guardrails(legs)
        desc = " / ".join(f"{s} {l} {'L' if d > 0 else 'S'}" for s, l, d in legs)
        print(f"   -> ENTRY {tag}: {desc} | guardrail {'PASS' if ok else 'REJECT: ' + reason}")
        if ok:
            for s, l, d in legs:
                ex.place_order(s, l, d, comment=tag)
            state[tag] = dict(alpha=r["alpha"], beta=r["beta"], mu=r["mu"],
                              sigma=r["sd"], entry_z=r["z"],
                              entry_time=now.isoformat(), half_life_min=r["hl_min"],
                              symbols=[a, b])
            held_syms.update([a, b])
            entered += 1

    save_state(state)


def tag_of(a, b):
    return f"{a}/{b}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ex.connect()
    run_cycle()                       # ONE cycle, for testing
    # For LIVE, replace the single call with an aligned loop:
    #   while True:
    #       run_cycle()
    #       time.sleep(900)
    mt5.shutdown()
    print(f"\nDRY_RUN = {ex.DRY_RUN}  (flip in mt5_executor.py when ready to go live)")
    print("Tip: delete state.json for a clean slate between dry-run tests.")
