"""
mt5_executor.py — sizing + guardrail + execution for Model to Market.

Shared by BOTH books:
  - live_trader.py        (FX mean-reversion core — your survival/rank engine)
  - directional_trader.py (Donchian breakout sleeve — your return engine)

KEY SAFETY DESIGN
-----------------
- Module-level DRY_RUN is the DEFAULT for any call that doesn't override it.
- place_order() takes a per-call `dry_run` override. This lets the FX core
  run LIVE (ex.DRY_RUN=False) while the directional book is independently
  dry-run-tested (passes dry_run=True) on the SAME account.
- Sizing/risk use the broker's own calculator (order_calc_margin), so currency
  conversion is correct, never hand-rolled.
- Credentials come from the environment (MT5_LOGIN, MT5_SERVER, MT5_PASSWORD),
  never the repo.

CHANGE (magic split + SL):
- MAGIC is no longer one shared tag. Each sleeve has its own so they can be told
  apart on a shared account:
      MAGIC_DIR = directional sleeve  (== the existing/legacy live positions)
      MAGIC_FX  = FX mean-reversion core (new, unique)
  `MAGIC` is kept as a backward-compat alias == MAGIC_DIR so directional_trader.py
  (which relied on the old default) is unchanged. place_order() now defaults to
  MAGIC_DIR; live_trader.py passes magic=MAGIC_FX explicitly.
- place_order() now accepts an optional `sl=` price and attaches it to the deal,
  so the FX core can set a broker-side emergency stop on every leg.
"""

import os
import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DRY_RUN = False            # FX core default. Live for Round 2. Per-call override exists.

LOGIN    = int(os.environ.get("MT5_LOGIN", "0"))
SERVER   = os.environ.get("MT5_SERVER")
PASSWORD = os.environ.get("MT5_PASSWORD")

MAX_MARGIN_USAGE = 0.85
MAX_SINGLE_INSTR = 0.80
STOPOUT_LEVEL    = 0.30

# --- order tags, one per sleeve so a shared account stays distinguishable ---
MAGIC_DIR = 20260621       # directional (Donchian) sleeve — ALSO the existing live positions
MAGIC_FX  = 20260631       # FX mean-reversion core (new, unique)
MAGIC     = MAGIC_DIR      # backward-compat alias for any code importing MAGIC


# ---------------------------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------------------------
def connect():
    if not LOGIN or not SERVER or PASSWORD is None:
        raise RuntimeError("MT5_LOGIN, MT5_SERVER, and MT5_PASSWORD env vars must all be set.")
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    term = mt5.terminal_info()
    if term is None or not term.trade_allowed:
        raise RuntimeError("Algo Trading is DISABLED in the terminal (press Ctrl+E). "
                           "Orders would be silently rejected. Enable it and re-run.")
    acc = mt5.account_info()
    print(f"Connected: {acc.login} | equity {acc.equity:,.2f} | "
          f"leverage {acc.leverage}x | margin used {acc.margin:,.2f}")
    return acc


# ---------------------------------------------------------------------------
# SIZING
# ---------------------------------------------------------------------------
def _round_to_step(lots, step):
    return round(round(lots / step) * step, 8)


def _price(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask == 0:
        return None
    return tick.ask if direction > 0 else tick.bid


def margin_for(symbol, lots, direction):
    price = _price(symbol, direction)
    if price is None:
        return None
    otype = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    return mt5.order_calc_margin(otype, symbol, float(lots), price)


def size_by_margin(symbol, margin_budget_usd, direction):
    """Lots that use ~margin_budget_usd of margin. Clamped to broker min/step/max."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None, None
    mt5.symbol_select(symbol, True)
    margin_1lot = margin_for(symbol, 1.0, direction)
    if not margin_1lot:
        return None, None
    raw = margin_budget_usd / margin_1lot
    lots = _round_to_step(raw, info.volume_step)
    lots = max(info.volume_min, min(lots, info.volume_max))
    return lots, margin_for(symbol, lots, direction)


# ---------------------------------------------------------------------------
# GUARDRAIL  (aggregate book: existing positions + proposed legs)
# ---------------------------------------------------------------------------
def check_guardrails(proposed):
    """proposed = list of (symbol, lots, direction). Evaluates the WHOLE account
    (FX core + directional + proposed), since margin/stop-out are account-wide."""
    acc = mt5.account_info()
    equity = acc.equity
    existing_margin = acc.margin

    add_margin = 0.0
    per_instr = {}
    for sym, lots, d in proposed:
        m = margin_for(sym, lots, d) or 0.0
        add_margin += m
        per_instr[sym] = per_instr.get(sym, 0.0) + m

    total_margin = existing_margin + add_margin
    if total_margin <= 0:
        return True, "no margin used", {"total_margin": 0.0}

    margin_usage = total_margin / equity
    margin_level = equity / total_margin
    max_instr_frac = (max(per_instr.values()) / total_margin) if per_instr else 0.0
    stopout_equity = STOPOUT_LEVEL * total_margin
    cushion_frac = (equity - stopout_equity) / equity if equity else 0.0

    stats = dict(equity=equity, total_margin=total_margin,
                 margin_usage=margin_usage, margin_level=margin_level,
                 max_instr_frac=max_instr_frac, cushion_to_stopout=cushion_frac)

    if margin_usage > MAX_MARGIN_USAGE:
        return False, f"margin usage {margin_usage:.1%} > cap {MAX_MARGIN_USAGE:.0%}", stats
    if max_instr_frac > MAX_SINGLE_INSTR:
        return False, f"single-instrument {max_instr_frac:.1%} > cap {MAX_SINGLE_INSTR:.0%}", stats
    return True, "ok", stats


# ---------------------------------------------------------------------------
# ORDER PLACEMENT  (per-call dry_run override + optional broker-side SL)
# ---------------------------------------------------------------------------
def _filling_mode(symbol):
    info = mt5.symbol_info(symbol)
    mode = info.filling_mode if info else 0
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def place_order(symbol, lots, direction, comment="m2m", magic=MAGIC_DIR,
                sl=None, dry_run=None):
    """Market order.

    magic:   defaults to MAGIC_DIR (directional sleeve / legacy). The FX core
             passes magic=MAGIC_FX so the two books stay distinguishable.
    sl:      optional stop-loss PRICE. Attached to the deal if provided. Used by
             the FX core as a wide emergency backstop on every leg.
    dry_run: None -> use module DRY_RUN. True/False -> override for THIS call.
    """
    effective_dry = DRY_RUN if dry_run is None else dry_run

    price = _price(symbol, direction)
    if price is None:
        print(f"[SKIP] {symbol}: no price"); return None
    otype = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": otype,
        "price": price,
        "deviation": 20,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(symbol),
    }
    if sl is not None:
        req["sl"] = float(sl)

    side = "BUY" if direction > 0 else "SELL"
    sl_txt = f" sl={sl}" if sl is not None else ""
    if effective_dry:
        print(f"[DRY-RUN] would send {side} {lots} {symbol} @ {price}{sl_txt} (magic={magic})")
        return {"dry_run": True, "request": req}
    result = mt5.order_send(req)
    print(f"[LIVE] {side} {lots} {symbol}{sl_txt}: retcode={getattr(result,'retcode',None)} "
          f"{getattr(result,'comment','')}")
    return result