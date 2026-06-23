"""
live_trader.py - live pairs mean-reversion loop for Model to Market (OPTIMIZED).

Runs ON the VPS, in-process. Pulls 15-min bars from MT5, screens cointegration +
half-life every cycle, and trades the pairs that currently qualify. Routes all
orders through mt5_executor (DRY_RUN-gated).

============================================================================
CHANGES vs the previous "hardened" version
============================================================================
TIER 1 - safety / correctness (must-have before live):
  [1] Concurrency cap no longer double-counts. Was `len(state) + entered >= CAP`,
      which stopped at CAP/2 because each fill bumps both terms. Now `len(state)`.
  [2] COEXISTENCE GUARD: refuse to enter any pair whose symbol is already held by a
      position the bot did NOT open (magic != ex.MAGIC). Stops this loop from netting
      into a manual/other-EA directional book on the same account.
  [3] STATE PERSISTED PER-ENTRY: state.json is written immediately after each pair is
      opened (atomic write, cheap), not once at end-of-cycle. A crash mid-cycle can no
      longer orphan a recorded-nowhere position. Startup adopt/flatten of orphans added.
  [4] EXIT->RE-ENTRY SEAM CLOSED: symbols touched by an exit this cycle are deferred to
      the next bar, so a not-yet-filled close can't be re-entered and netted.
  [5] HEDGE-RATIO DRIFT CHECK: if lot rounding/clamping pushes the realised hedge ratio
      more than BETA_DRIFT_TOL off the intended |beta|, the entry is rejected instead of
      silently leaving net directional exposure on a "neutral" pair.
  [6] EMERGENCY BROKER-SIDE STOP: every entry order carries a wide catastrophe SL as a
      dead-man's switch for loop/VPS death. This is NOT the strategy stop (that's the
      frozen-Z divergence stop, evaluated each bar) - it's a backstop only.

TIER 2 - edge correctness (reduce false signals / overfitting):
  [7] PERSISTENCE FILTER: a pair must pass the screen for PERSIST_CYCLES consecutive
      cycles before it is tradable. Kills flickering pairs and most multiple-testing
      false positives at once.
  [8] SYMMETRIC COINTEGRATION: Engle-Granger run BOTH orderings; require the WORSE
      p-value to clear ADF_PMAX (max(p_ab, p_ba)). Removes order-dependence and is
      conservative against spurious pairs.
  [9] POSITIVE-BETA REQUIREMENT: within-class pairs with a negative hedge ratio are
      usually spurious; reject them (toggle REQUIRE_POSITIVE_BETA).
  [10] OUT-OF-SAMPLE ENTRY Z: fit alpha/beta/mu/sigma on a training window, evaluate the
      trigger z on a held-out tail. De-biases the in-sample-optimistic entry. Exit logic
      remains frozen-baseline, unchanged.
  [11] COST GATE (optional): require expected reversion capture to exceed a multiple of
      the round-trip spread cost before entering. Non-blocking if data missing.

MINOR: removed unused adfuller import; removed duplicate `import time`; N_BARS comment
corrected; assorted logging.

Workflow: edit on Mac -> git push -> git pull on VPS.
Deps: MetaTrader5, numpy, pandas, statsmodels
"""

import os
import json
import time
import tempfile
import itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint   # adfuller no longer used
import MetaTrader5 as mt5

import mt5_executor as ex   # connect/size/guardrail/place_order + DRY_RUN + MAGIC


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FX_UNIVERSE = ["AUDUSD", "EURCHF", "EURGBP", "EURUSD",
               "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]
METALS      = ["XAUUSD", "XAGUSD"]
CRYPTO      = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD"]
UNIVERSE    = FX_UNIVERSE + METALS + CRYPTO

# Pair WITHIN asset class only. FX: 28 combos, Metals: 1, Crypto: 6 -> 35 candidates.
PAIRS = (list(itertools.combinations(FX_UNIVERSE, 2))
         + list(itertools.combinations(METALS, 2))
         + list(itertools.combinations(CRYPTO, 2)))

# --- signal ---
ENTRY_Z = 1.25
EXIT_Z  = 0.25
DIVERGENCE_Z = 3.5           # strategy hard stop: frozen |Z| past this => relationship broke
MAX_HOLD_HL  = 3.0           # time stop: exit after this many half-lives...
ABS_MAX_HOLD_MIN = 1440      # ...but never hold longer than one round (1 day)

# --- screening ---
N_BARS  = 700                # M15 history caps ~628 bars at this broker; ask for a bit more
MIN_FIT_BARS = 250
ADF_PMAX = 0.05
HL_MIN_MIN, HL_MAX_MIN = 120, 1440
BAR_MIN = 15
REQUIRE_POSITIVE_BETA = True            # [9]
HOLDOUT_BARS = 32                       # [10] out-of-sample tail for the entry z (~8h)
PERSIST_CYCLES = 2                      # [7] consecutive screen-passes required to trade

# --- sizing / risk ---
MARGIN_PER_PAIR      = 15_000
MAX_CONCURRENT_PAIRS = 8
BETA_DRIFT_TOL = 0.15                   # [5] max realised-vs-intended hedge ratio drift
EMERGENCY_SL_FRAC = {                   # [6] catastrophe backstop, NOT the strategy stop
    "FX": 0.03, "METALS": 0.06, "CRYPTO": 0.15,
}

# --- cost gate [11] ---
COST_GATE_ENABLED = True
COST_EDGE_MULT = 1.0                     # require expected capture > this * round-trip cost

# --- orphan handling on startup [3] ---
# "warn"    -> log loudly, touch nothing (safe default; you decide)
# "flatten" -> close any bot-magic position not present in state.json
ADOPT_ORPHANS = "warn"

STATE_FILE   = "state.json"
STREAK_FILE  = "streaks.json"


# ---------------------------------------------------------------------------
# ASSET CLASS HELPERS
# ---------------------------------------------------------------------------
def asset_class(symbol):
    if symbol in METALS:
        return "METALS"
    if symbol in CRYPTO:
        return "CRYPTO"
    return "FX"


# ---------------------------------------------------------------------------
# STATE (atomic JSON)
# ---------------------------------------------------------------------------
def _atomic_write(path, obj):
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def load_state():
    return _load_json(STATE_FILE, {})


def save_state(state):
    _atomic_write(STATE_FILE, state)


def load_streaks():
    return _load_json(STREAK_FILE, {})


def save_streaks(streaks):
    _atomic_write(STREAK_FILE, streaks)


def reconcile(state):
    """Drop pairs whose legs are no longer open at the broker (live only)."""
    if ex.DRY_RUN:
        return state
    open_syms = {p.symbol for p in (mt5.positions_get() or []) if p.magic == ex.MAGIC}
    alive = {}
    for tag, rec in state.items():
        # [6/strict] require *all* legs still open; a one-legged pair is a naked
        # directional position, not a hedge -> flatten the survivor and drop it.
        if all(s in open_syms for s in rec["symbols"]):
            alive[tag] = rec
        else:
            survivors = [s for s in rec["symbols"] if s in open_syms]
            if survivors:
                print(f"   [state] {tag} only partially open {survivors} -> flattening survivor(s)")
                _flatten_symbols(survivors, comment=tag + "-orphan-leg")
            else:
                print(f"   [state] {tag} no longer open at broker -> dropped")
    return alive


def _flatten_symbols(symbols, comment="flatten"):
    for p in (mt5.positions_get() or []):
        if p.magic == ex.MAGIC and p.symbol in symbols:
            close_dir = -1 if p.type == mt5.POSITION_TYPE_BUY else +1
            ex.place_order(p.symbol, p.volume, close_dir, comment=comment)


def handle_orphans_on_startup(state):
    """[3] Positions tagged by this bot but not recorded in state = orphans
    (e.g. crash between place_order and save). We cannot rebuild a frozen baseline
    for them, so the only safe resolutions are warn or flatten."""
    if ex.DRY_RUN:
        return
    recorded = {s for rec in state.values() for s in rec["symbols"]}
    orphan_syms = {p.symbol for p in (mt5.positions_get() or [])
                   if p.magic == ex.MAGIC and p.symbol not in recorded}
    if not orphan_syms:
        return
    print(f"!! ORPHANS (bot-magic, not in state): {sorted(orphan_syms)}")
    if ADOPT_ORPHANS == "flatten":
        print("   ADOPT_ORPHANS=flatten -> closing them")
        _flatten_symbols(orphan_syms, comment="startup-orphan")
    else:
        print("   ADOPT_ORPHANS=warn -> leaving them; resolve manually before trusting state")


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
    """Fit on a TRAINING window; evaluate the trigger z OUT OF SAMPLE on the tail [10].
    Cointegration screened SYMMETRICALLY (worse of both orderings) [8]."""
    df = pd.concat([a_close, b_close], axis=1, sort=False).dropna()
    df.columns = ["a", "b"]
    if len(df) < MIN_FIT_BARS + HOLDOUT_BARS:
        return None

    train = df if HOLDOUT_BARS == 0 else df.iloc[:-HOLDOUT_BARS]
    if len(train) < MIN_FIT_BARS:
        return None

    # hedge regression on train only
    res = sm.OLS(train["a"].values, sm.add_constant(train["b"].values)).fit()
    alpha, beta = float(res.params[0]), float(res.params[1])

    spread_train = train["a"] - (alpha + beta * train["b"])
    mu, sd = float(spread_train.mean()), float(spread_train.std())
    if not (sd > 0):
        return None

    # out-of-sample current z: latest bar is NOT in the fit window
    spread_last = float(df["a"].iloc[-1] - (alpha + beta * df["b"].iloc[-1]))
    z = (spread_last - mu) / sd

    # symmetric Engle-Granger on train: require the worse direction to still pass
    p_ab = float(coint(train["a"].values, train["b"].values, trend="c")[1])
    p_ba = float(coint(train["b"].values, train["a"].values, trend="c")[1])
    adf_p = max(p_ab, p_ba)

    hl_min = half_life(spread_train) * BAR_MIN
    return dict(alpha=alpha, beta=beta, mu=mu, sd=sd, z=z, adf_p=adf_p, hl_min=hl_min)


def passes_screen(f):
    if not (f["adf_p"] < ADF_PMAX):
        return False
    if not (np.isfinite(f["hl_min"]) and HL_MIN_MIN <= f["hl_min"] <= HL_MAX_MIN):
        return False
    if REQUIRE_POSITIVE_BETA and f["beta"] <= 0:    # [9]
        return False
    return True


def frozen_z(rec, a_close, b_close):
    a_now, b_now = float(a_close.iloc[-1]), float(b_close.iloc[-1])
    spread_now = a_now - (rec["alpha"] + rec["beta"] * b_now)
    return (spread_now - rec["mu"]) / rec["sigma"] if rec["sigma"] else float("nan")


# ---------------------------------------------------------------------------
# LEG SIZING / ORDER HELPERS
# ---------------------------------------------------------------------------
def price_value_per_lot(symbol):
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_size == 0:
        return None
    return info.trade_tick_value / info.trade_tick_size


def _spread_price(symbol):
    """Current bid/ask spread in price units (for the cost gate)."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask == 0 or tick.bid == 0:
        return None
    return float(tick.ask - tick.bid)


def emergency_sl_price(symbol, direction):
    """[6] Wide catastrophe stop, in the adverse direction. Backstop only."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    px = tick.bid if direction < 0 else tick.ask
    frac = EMERGENCY_SL_FRAC[asset_class(symbol)]
    return px * (1 + frac) if direction < 0 else px * (1 - frac)


def build_legs(a, b, f):
    """Return [(sym, lots, dir), ...] or None. Rejects on sizing failure OR on a
    hedge-ratio drift beyond tolerance [5]."""
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
    if lots_b <= 0:
        return None

    # [5] did rounding/clamping break the hedge?
    intended = abs(f["beta"])
    realised = (lots_b * pv_b) / (lots_a * pv_a) if (lots_a and pv_a) else 0.0
    drift = abs(realised / intended - 1.0) if intended > 0 else 1.0
    if drift > BETA_DRIFT_TOL:
        print(f"   -> {tag_of(a, b)} entry aborted (hedge drift {drift:.0%} "
              f"intended_beta={intended:.4f} realised={realised:.4f})")
        return None

    return [(a, lots_a, dir_a), (b, lots_b, dir_b)]


def cost_gate_ok(legs, f):
    """[11] expected reversion capture (A-leg, in account ccy) vs round-trip spread cost.
    Non-blocking: returns True if anything needed is missing."""
    if not COST_GATE_ENABLED:
        return True
    try:
        (a, lots_a, _), (b, lots_b, _) = legs
        pv_a, pv_b = price_value_per_lot(a), price_value_per_lot(b)
        sp_a, sp_b = _spread_price(a), _spread_price(b)
        if None in (pv_a, pv_b, sp_a, sp_b):
            return True
        # capture: spread mean-reverts from |z| toward EXIT_Z; sd is in A price units
        capture_px = max(abs(f["z"]) - EXIT_Z, 0.0) * f["sd"]
        capture_ccy = capture_px * pv_a * lots_a
        roundtrip_cost = 2.0 * (sp_a * pv_a * lots_a + sp_b * pv_b * lots_b)
        ok = capture_ccy > COST_EDGE_MULT * roundtrip_cost
        if not ok:
            print(f"   -> cost gate: capture~{capture_ccy:,.0f} <= "
                  f"{COST_EDGE_MULT}x cost~{roundtrip_cost:,.0f}")
        return ok
    except Exception as e:
        print(f"   [cost gate skipped] {type(e).__name__}: {e}")
        return True


def place_pair(legs, tag):
    """Place both legs with an emergency SL on each [6].
    NOTE: requires mt5_executor.place_order to accept an `sl=` price kwarg and apply it.
    If your executor doesn't, either add it there or attach via mt5.order_send modify."""
    for s, l, d in legs:
        sl = emergency_sl_price(s, d)
        ex.place_order(s, l, d, comment=tag, sl=sl)


def close_pair(tag, rec):
    for p in (mt5.positions_get() or []):
        if p.magic == ex.MAGIC and p.symbol in rec["symbols"]:
            close_dir = -1 if p.type == mt5.POSITION_TYPE_BUY else +1
            ex.place_order(p.symbol, p.volume, close_dir, comment=tag + "-exit")


def externally_held_syms():
    """[2] symbols with an open position this bot did NOT open."""
    if ex.DRY_RUN:
        return set()
    return {p.symbol for p in (mt5.positions_get() or []) if p.magic != ex.MAGIC}


# ---------------------------------------------------------------------------
# ONE CYCLE
# ---------------------------------------------------------------------------
def run_cycle():
    state = reconcile(load_state())
    streaks = load_streaks()
    now = datetime.now(timezone.utc)
    print(f"\n=== cycle {now:%Y-%m-%d %H:%M} UTC | holding {len(state)}: "
          f"{sorted(state) or 'none'} ===")

    closes = {s: get_closes(s) for s in UNIVERSE}

    # ---- EXIT pass ----
    exited_syms = set()                          # [4]
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
            exited_syms.update(rec["symbols"])   # [4] defer these symbols this cycle
            del state[tag]
            save_state(state)                    # [3] persist immediately

    # ---- SCAN + persistence streak update [7] ----
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

    # rebuild streaks from this cycle's scan (prunes stale tags automatically)
    new_streaks = {}
    for r in scan:
        if r["screened"]:
            new_streaks[r["tag"]] = streaks.get(r["tag"], 0) + 1
        # failing pairs reset to 0 by omission
    streaks = new_streaks
    save_streaks(streaks)

    n_pass = sum(r["screened"] for r in scan)
    show = sorted([r for r in scan if r["screened"] or r["adf_p"] < 0.10],
                  key=lambda r: -abs(r["z"]))
    print(f"{n_pass}/{len(scan)} flat pairs pass screen | PASS + near-miss:")
    for r in show:
        print(f"  {r['tag']:16} Z={r['z']:+.2f} beta={r['beta']:+.4f} "
              f"adf_p={r['adf_p']:.4f} hl={r['hl_min']:.0f}m streak={streaks.get(r['tag'], 0)} "
              f"{'PASS' if r['screened'] else 'fail'}")

    # ---- ENTRY pass ----
    held_syms = {s for r in state.values() for s in r["symbols"]}
    ext_syms = externally_held_syms()            # [2]
    if ext_syms:
        print(f"   [coexistence] non-bot positions present on: {sorted(ext_syms)} (blocked)")

    for r in sorted(scan, key=lambda r: -abs(r["z"])):
        if not (r["screened"] and abs(r["z"]) > ENTRY_Z):
            continue
        if streaks.get(r["tag"], 0) < PERSIST_CYCLES:                 # [7]
            print(f"   -> {r['tag']} skipped (streak {streaks.get(r['tag'], 0)}/{PERSIST_CYCLES})")
            continue

        a, b, tag = r["a"], r["b"], r["tag"]
        if len(state) >= MAX_CONCURRENT_PAIRS:                        # [1] fixed
            print(f"   -> {tag} skipped (at {MAX_CONCURRENT_PAIRS}-pair cap)"); continue
        if a in held_syms or b in held_syms:
            print(f"   -> {tag} skipped (symbol overlap; netting account)"); continue
        if a in ext_syms or b in ext_syms:                           # [2]
            print(f"   -> {tag} skipped (symbol held by non-bot position)"); continue
        if a in exited_syms or b in exited_syms:                     # [4]
            print(f"   -> {tag} skipped (symbol exited this cycle; defer to next bar)"); continue

        legs = build_legs(a, b, r)
        if not legs:
            continue                                                  # build_legs logged why
        if not cost_gate_ok(legs, r):                                # [11]
            continue

        ok, reason, _ = ex.check_guardrails(legs)
        desc = " / ".join(f"{s} {l} {'L' if d > 0 else 'S'}" for s, l, d in legs)
        print(f"   -> ENTRY {tag}: {desc} | guardrail {'PASS' if ok else 'REJECT: ' + reason}")
        if ok:
            place_pair(legs, tag)                                    # [6] SL attached
            state[tag] = dict(alpha=r["alpha"], beta=r["beta"], mu=r["mu"],
                              sigma=r["sd"], entry_z=r["z"],
                              entry_time=now.isoformat(), half_life_min=r["hl_min"],
                              symbols=[a, b])
            held_syms.update([a, b])
            save_state(state)                                        # [3] persist per entry


def tag_of(a, b):
    return f"{a}/{b}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ex.connect()
    print(f"Live loop started. DRY_RUN={ex.DRY_RUN} MAGIC={getattr(ex, 'MAGIC', '??')}")
    handle_orphans_on_startup(load_state())      # [3]
    print("Aligning to 15-min bars. Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            sleep_s = 900 - (now % 900) + 5
            mins = int((900 - (now % 900)) // 60)
            print(f"...sleeping {sleep_s:.0f}s to next bar (~{mins}m)")
            time.sleep(sleep_s)
            try:
                run_cycle()
            except Exception as e:
                print(f"[cycle error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nLoop stopped by user.")
    finally:
        mt5.shutdown()
        print(f"DRY_RUN = {ex.DRY_RUN}")