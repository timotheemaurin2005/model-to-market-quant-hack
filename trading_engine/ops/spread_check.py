"""
spread_check.py - transaction-cost sanity check for the pairs book.

For each currently-passing pair it compares the round-trip bid/ask cost to the
reversion the strategy actually captures (entry |Z|=2.0 -> exit |Z|=0.5 = 1.5 sigma).
Cost is expressed in the SAME units as the Z-score (spread sigmas), so the verdict
is apples-to-apples: if cost eats most of the 1.5-sigma capture, you'd bleed.

Round-trip cost in spread-price units = (ask-bid)_A + |beta| * (ask-bid)_B
  (a full bid/ask on each leg = the 2 crossings per leg, i.e. the reviewer's "4 crossings")

Run ON the VPS from the repo folder (it imports live_trader). Reads live MT5 spreads.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
import mt5_executor as ex
from live_trader import (FX_UNIVERSE, PAIRS, get_closes, fit_pair, passes_screen,
                         ENTRY_Z, EXIT_Z)

CAPTURE = ENTRY_Z - EXIT_Z      # sigmas of gross reversion captured per round trip


def leg_info(symbol):
    t = mt5.symbol_info_tick(symbol)
    i = mt5.symbol_info(symbol)
    if not t or not i or i.point == 0:
        return None
    spr_price = t.ask - t.bid
    return spr_price, spr_price / i.point          # (price, points)


if __name__ == "__main__":
    ex.connect()
    closes = {s: get_closes(s) for s in FX_UNIVERSE}

    rows = []
    for a, b in PAIRS:
        if closes.get(a) is None or closes.get(b) is None:
            continue
        f = fit_pair(closes[a], closes[b])
        if f is None or not passes_screen(f) or f["sd"] == 0:
            continue
        la, lb = leg_info(a), leg_info(b)
        if not la or not lb:
            continue
        cost_price = la[0] + abs(f["beta"]) * lb[0]
        cost_sigma = cost_price / f["sd"]
        ratio = cost_sigma / CAPTURE
        verdict = "HEALTHY" if ratio < 0.33 else "MARGINAL" if ratio < 0.66 else "BLEEDS"
        rows.append((f"{a}/{b}", la[1], lb[1], f["beta"], cost_sigma,
                     CAPTURE - cost_sigma, verdict))

    rows.sort(key=lambda r: r[4])      # cheapest first
    print(f"\nRound-trip cost vs {CAPTURE:.1f}-sigma capture "
          f"(entry|Z|={ENTRY_Z} -> exit|Z|={EXIT_Z})\n")
    if not rows:
        print("No pairs currently pass the screen -- nothing to cost-check right now.")
    else:
        print(f"{'pair':16}{'sprA_pts':>9}{'sprB_pts':>9}{'beta':>8}"
              f"{'cost_sig':>10}{'net_sig':>9}  verdict")
        for tag, pa, pb, beta, cs, ns, v in rows:
            print(f"{tag:16}{pa:9.1f}{pb:9.1f}{beta:+8.3f}{cs:10.3f}{ns:9.3f}  {v}")
        print("\n  HEALTHY  cost < 1/3 of capture  -> spreads aren't eating the edge")
        print("  MARGINAL cost 1/3-2/3 of capture -> widen capture (exit nearer Z=0) or raise entry Z")
        print("  BLEEDS   cost > 2/3 of capture   -> don't trade this pair as-is")

    mt5.shutdown()
