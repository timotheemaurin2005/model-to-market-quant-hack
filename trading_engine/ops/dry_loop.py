#!/usr/bin/env python3
"""
dry_loop.py — overnight DRY-RUN of the live pairs loop.

WHY THIS EXISTS: running `python live_trader.py` uses mt5_executor.DRY_RUN, which
is currently False (live). This wrapper pins DRY_RUN=True for the whole loop, so the
pairs engine screens + would-be-enters every 15 min and PLACES NOTHING. Lets you
gather overnight data on how often the crypto-extended universe passes the screen and
triggers at the new ENTRY_Z, with zero risk.

It does NOT edit mt5_executor.py — so tomorrow's switch to live is a deliberate,
explicit flip, not "did I remember to undo the overnight hack."

RUN:  python dry_loop.py
   or, to keep a reviewable log:  python dry_loop.py > dry_overnight.log 2>&1
STOP: Ctrl+C.
"""

import time
import mt5_executor as ex
ex.DRY_RUN = True                    # pin dry for this whole process, BEFORE importing the loop
import live_trader as lt
import MetaTrader5 as mt5

if __name__ == "__main__":
    ex.connect()
    print(f"DRY overnight loop started. DRY_RUN={ex.DRY_RUN} — nothing will be placed. "
          f"Universe={len(lt.UNIVERSE)} symbols, {len(lt.PAIRS)} pairs, ENTRY_Z={lt.ENTRY_Z}. "
          f"Ctrl+C to stop.")
    try:
        while True:
            now = time.time()
            sleep_s = 900 - (now % 900) + 5      # +5s so the bar has closed
            mins = int((900 - (now % 900)) // 60)
            print(f"...sleeping {sleep_s:.0f}s to next bar (~{mins}m)")
            time.sleep(sleep_s)
            try:
                lt.run_cycle()
            except Exception as e:
                print(f"[cycle error] {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        print("\nDry loop stopped by user.")
    finally:
        mt5.shutdown()
        print(f"shutdown. DRY_RUN was {ex.DRY_RUN} throughout.")