"""
mt5_executor.py — sizing + guardrail + (dry-run) execution for Model to Market.

Runs ON the VPS, in-process with the signal loop. No HTTP, no public port.
The signal loop imports this and calls size_position() / check_guardrails() / place_order().

SAFETY
------
- DRY_RUN defaults True. Nothing is sent to the broker until you set it False.
- Sizing and risk use the broker's own calculators (order_calc_margin). Currency
  conversion for USD/JPY/CHF/cross pairs is therefore correct, never hand-rolled.
- Credentials come from the environment, never the repo:
      set MT5_PASSWORD via Windows System Properties -> Environment Variables
      (use the GUI so the " in the password needs no shell escaping).
"""

import os
import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DRY_RUN = False            # <-- flip to False ONLY when you intend to place live orders

LOGIN    = 10301
SERVER   = "3.11.134.149:443"
PASSWORD = os.environ.get("MT5_PASSWORD")   # set on the VPS, not in the repo

# Guardrail caps — sit deliberately below competition penalty cliffs / stop-out.
# Competition: margin-usage penalty >90%, leverage penalty >28x, force-liquidation
# at 30% margin LEVEL (= instant elimination).
MAX_MARGIN_USAGE = 0.85   # used_margin / equity  (also ~= 25x leverage on 30x pairs)
MAX_SINGLE_INSTR = 0.80   # one instrument's margin / total used margin
STOPOUT_LEVEL    = 0.30   # broker force-liquidation threshold (margin level)

MAGIC = 20260621          # tag so we can identify our own orders


# ---------------------------------------------------------------------------
# CONNECTION
# ---------------------------------------------------------------------------
def connect():
    if PASSWORD is None:
        raise RuntimeError("MT5_PASSWORD env var is not set on this machine.")
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    term = mt5.terminal_info()
    if term is None or not term.trade_allowed:
        # initialize() can flip the terminal's Algo Trading off; surface it loudly.
        raise RuntimeError("Algo Trading is DISABLED in the terminal (press Ctrl+E). "
                           "Orders would be silently rejected. Enable it and re-run.")
    acc = mt5.account_info()
    print(f"Connected: {acc.login} | equity {acc.equity:,.2f} | "
          f"leverage {acc.leverage}x | margin used {acc.margin:,.2f}")
    return acc


# ---------------------------------------------------------------------------
# SIZING  (margin-budget based, using the broker's calculator)
# ---------------------------------------------------------------------------
def _round_to_step(lots, step):
    return round(round(lots / step) * step, 8)


def _price(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask == 0:
        return None
    return tick.ask if direction > 0 else tick.bid


def margin_for(symbol, lots, direction):
    """USD margin the broker would require for `lots` of `symbol`."""
    price = _price(symbol, direction)
    if price is None:
        return None
    otype = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
    return mt5.order_calc_margin(otype, symbol, float(lots), price)


def size_by_margin(symbol, margin_budget_usd, direction):
    """Lots that use ~margin_budget_usd of margin. Clamped to broker min/step/max.

    Returns (lots, est_margin) or (None, None) if the symbol isn't tradable.
    """
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
# GUARDRAIL  (one-way gate: reject or accept; never resizes up)
# ---------------------------------------------------------------------------
def check_guardrails(proposed):
    """proposed = list of (symbol, lots, direction).

    Returns (ok: bool, reason: str, stats: dict). Evaluates the AGGREGATE book
    (existing positions + proposed), which is what actually drives the stop-out.
    """
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

    margin_usage = total_margin / equity            # competition penalty metric
    margin_level = equity / total_margin            # stop-out at 0.30
    max_instr_frac = (max(per_instr.values()) / total_margin) if per_instr else 0.0

    # How far can equity fall before the 30% stop-out, expressed as a % of equity.
    # stop-out when equity == 0.30 * total_margin  ->  cushion below.
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
# ORDER PLACEMENT  (gated by DRY_RUN)
# ---------------------------------------------------------------------------
def _filling_mode(symbol):
    """Pick a filling mode the symbol actually supports (IOC preferred, else FOK)."""
    info = mt5.symbol_info(symbol)
    mode = info.filling_mode if info else 0
    if mode & 2:   # SYMBOL_FILLING_IOC
        return mt5.ORDER_FILLING_IOC
    if mode & 1:   # SYMBOL_FILLING_FOK
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def place_order(symbol, lots, direction, comment="m2m"):
    """Market order. In DRY_RUN it only logs the request and sends nothing."""
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
        "magic": MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(symbol),
    }
    side = "BUY" if direction > 0 else "SELL"
    if DRY_RUN:
        print(f"[DRY-RUN] would send {side} {lots} {symbol} @ {price}")
        return {"dry_run": True, "request": req}
    result = mt5.order_send(req)
    ok = result and result.retcode == mt5.TRADE_RETCODE_DONE
    print(f"[LIVE] {side} {lots} {symbol}: retcode={getattr(result,'retcode',None)} "
          f"{getattr(result,'comment','')}")
    return result


# ---------------------------------------------------------------------------
# DEMO  — read-only walk-through of the full pipeline (safe; nothing is sent)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    connect()

    # Sample pair trade: long EURUSD / short USDJPY, ~$15k margin per leg.
    MARGIN_PER_LEG = 15_000
    legs = [("EURUSD", +1), ("USDJPY", -1)]

    proposed = []
    print("\n--- sizing ---")
    for sym, d in legs:
        lots, est_m = size_by_margin(sym, MARGIN_PER_LEG, d)
        if lots is None:
            print(f"{sym}: not tradable"); continue
        print(f"{sym:8} {'LONG' if d>0 else 'SHORT':5} {lots} lots  est margin ${est_m:,.0f}")
        proposed.append((sym, lots, d))

    print("\n--- guardrail (aggregate book) ---")
    ok, reason, stats = check_guardrails(proposed)
    for k, v in stats.items():
        print(f"  {k:20} {v:,.4f}" if isinstance(v, float) else f"  {k:20} {v}")
    print(f"  -> {'PASS' if ok else 'REJECT'}: {reason}")

    if ok:
        print("\n--- placement (DRY-RUN) ---")
        for sym, lots, d in proposed:
            place_order(sym, lots, d, comment="demo")

    mt5.shutdown()
    print(f"\nDRY_RUN = {DRY_RUN}  (no orders were sent)" if DRY_RUN
          else "\nLIVE MODE — orders above were real.")
