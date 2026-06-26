"""
directional_trader.py — Donchian breakout sleeve for Model to Market.

Runs ON the VPS in its OWN terminal, alongside live_trader.py.

IT HAS ITS OWN DRY_RUN FLAG (below). Every order call passes dry_run=DRY_RUN to
mt5_executor, so this book can be dry-run-tested while the FX core trades live on
the SAME account. Set DRY_RUN=False here ONLY when you've verified entries, exits,
and the kill switch in dry-run.

PROTECTIONS
-----------
- Margin-based sizing (ex.size_by_margin) — no point/tick bug.
- Real exits — trailing + 3% hard stop place actual closing orders.
- Combined guardrail — every entry checks the WHOLE account's margin.
- Kill switch — cumulative book P&L <= -$50k -> liquidate + halt for the run.
- Own magic number — isolates this book for P&L and reconciliation.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trading_engine"))

import pandas as pd
import MetaTrader5 as mt5

import mt5_executor as ex

# ---------------------------------------------------------------------------
# CONFIG  — THIS book's switches
# ---------------------------------------------------------------------------
DRY_RUN = True                       # <-- independent of the FX core. Flip when verified.

KILL_SWITCH_USD = -50_000.0
COMPETITION_START = datetime(2026, 6, 21, 22, 0, tzinfo=timezone.utc)
MAGIC_DIRECTIONAL = 20260702

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]

ENTRY_PERIOD = 48                    # 12h breakout
EXIT_PERIOD  = 24                    # 6h trailing exit
HARD_STOP_PCT = 0.03                 # 3% hard stop
N_BARS = 500
MARGIN_PER_TRADE = 12_000            # margin budget per position

_HALTED = False


# ---------------------------------------------------------------------------
# P&L + KILL SWITCH
# ---------------------------------------------------------------------------
def directional_cum_pnl():
    unrealized = sum(p.profit for p in (mt5.positions_get() or [])
                     if p.magic == MAGIC_DIRECTIONAL)
    realized = 0.0
    deals = mt5.history_deals_get(COMPETITION_START, datetime.now(timezone.utc))
    if deals:
        realized = sum(d.profit for d in deals if d.magic == MAGIC_DIRECTIONAL)
    return realized + unrealized


def directional_positions(symbol=None):
    pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    return [p for p in (pos or []) if p.magic == MAGIC_DIRECTIONAL]


def close_position(p):
    close_dir = -1 if p.type == mt5.POSITION_TYPE_BUY else +1
    return ex.place_order(p.symbol, p.volume, close_dir,
                          comment=f"dir-exit-{p.ticket}",
                          magic=MAGIC_DIRECTIONAL, dry_run=DRY_RUN)


def liquidate_directional():
    print("KILL SWITCH: liquidating directional book")
    for p in directional_positions():
        close_position(p)


# ---------------------------------------------------------------------------
# DONCHIAN
# ---------------------------------------------------------------------------
def donchian_levels(symbol):
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, N_BARS)
    if rates is None or len(rates) < ENTRY_PERIOD + 5:
        return None
    df = pd.DataFrame(rates)
    return dict(
        entry_upper=df["high"].shift(1).rolling(ENTRY_PERIOD).max().iloc[-1],
        entry_lower=df["low"].shift(1).rolling(ENTRY_PERIOD).min().iloc[-1],
        exit_upper=df["high"].shift(1).rolling(EXIT_PERIOD).max().iloc[-1],
        exit_lower=df["low"].shift(1).rolling(EXIT_PERIOD).min().iloc[-1],
        close=float(df["close"].iloc[-1]),
    )


# ---------------------------------------------------------------------------
# CYCLE
# ---------------------------------------------------------------------------
def run_directional_cycle():
    global _HALTED
    acc = mt5.account_info()
    if acc is None:
        print("  [skip] no account info"); return

    cum = directional_cum_pnl()
    print(f"\n{'='*52}")
    print(f"DIRECTIONAL (DRY_RUN={DRY_RUN})  {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print(f"Book P&L: ${cum:,.2f}  |  kill switch: ${KILL_SWITCH_USD:,.0f}")

    if _HALTED:
        print("HALTED (kill switch already tripped)."); return
    if cum <= KILL_SWITCH_USD:
        liquidate_directional(); _HALTED = True
        print("DIRECTIONAL HALTED FOR THE REST OF THIS RUN."); return

    for sym in SYMBOLS:
        lv = donchian_levels(sym)
        if lv is None:
            print(f"  {sym}: insufficient data"); continue
        info, tick = mt5.symbol_info(sym), mt5.symbol_info_tick(sym)
        if info is None or tick is None or tick.ask == 0:
            print(f"  {sym}: no quote"); continue

        close = lv["close"]
        held = directional_positions(sym)

        # EXIT
        if held:
            p = held[0]
            exit_now, why = False, ""
            if p.type == mt5.POSITION_TYPE_BUY:
                if close < lv["exit_lower"]:
                    exit_now, why = True, f"trailing ({close:.2f} < {lv['exit_lower']:.2f})"
                elif close < p.price_open * (1 - HARD_STOP_PCT):
                    exit_now, why = True, f"HARD STOP -{HARD_STOP_PCT:.0%}"
            else:
                if close > lv["exit_upper"]:
                    exit_now, why = True, f"trailing ({close:.2f} > {lv['exit_upper']:.2f})"
                elif close > p.price_open * (1 + HARD_STOP_PCT):
                    exit_now, why = True, f"HARD STOP +{HARD_STOP_PCT:.0%}"
            side = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
            if exit_now:
                print(f"  {sym}: EXIT {side} -> {why}")
                close_position(p)
            else:
                print(f"  {sym}: hold {side} (open {p.price_open:.2f}, now {close:.2f}, "
                      f"P&L ${p.profit:,.2f})")
            continue

        # ENTRY
        direction = +1 if close > lv["entry_upper"] else (-1 if close < lv["entry_lower"] else 0)
        if direction == 0:
            print(f"  {sym}: no breakout (close {close:.2f}, "
                  f"chan [{lv['entry_lower']:.2f}, {lv['entry_upper']:.2f}])")
            continue

        lots, est_margin = ex.size_by_margin(sym, MARGIN_PER_TRADE, direction)
        if lots is None:
            print(f"  {sym}: sizing failed"); continue

        ok, reason, stats = ex.check_guardrails([(sym, lots, direction)])
        side = "LONG" if direction > 0 else "SHORT"
        stop_px = close * (1 - HARD_STOP_PCT) if direction > 0 else close * (1 + HARD_STOP_PCT)
        print(f"  {sym}: BREAKOUT {side} @ ~{close:.2f} lots={lots} "
              f"margin=${est_margin:,.0f} stop={stop_px:.2f}")
        print(f"     guardrail: {'PASS' if ok else 'REJECT — ' + reason} "
              f"(usage {stats.get('margin_usage',0):.1%}, level {stats.get('margin_level',0):.2f})")
        if ok:
            ex.place_order(sym, lots, direction, comment=f"dir-{sym}",
                           magic=MAGIC_DIRECTIONAL, dry_run=DRY_RUN)


# ---------------------------------------------------------------------------
# LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ex.connect()
    print(f"Directional loop started. DRY_RUN={DRY_RUN}. Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            sleep_s = 900 - (now % 900) + 7
            print(f"...sleeping {sleep_s:.0f}s to next bar")
            time.sleep(sleep_s)
            try:
                run_directional_cycle()
            except Exception as e:
                print(f"[cycle error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mt5.shutdown()
        print(f"DRY_RUN = {DRY_RUN}")
