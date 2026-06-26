#!/usr/bin/env python3
"""
pinger.py  —  Mechanical Event Monitor (Channel 1)

Out-of-band watchdog for the Dual-Brain FX stat-arb system. Runs ALONGSIDE the
execution loop (never inside it). It diffs state.json (or trade_state.json) between polls
and checks live margin, then sends a Telegram alert ONLY when something mechanical and
pre-decided changes. It never makes or suggests trading decisions — it reports
facts that map to a rule you already set.

WHAT IT PINGS ON (deterministic triggers only):
  • New entry            — a pair appeared in state.json
  • Exit                 — a pair left state.json (closed: revert / stop / time)
  • GBP TRIPWIRE         — EURGBP/USDCAD exits a 2nd time => your rule says
                           "exclude GBP for today". Pinged loudly, once.
  • Margin usage > 70%   — real risk worth your eyes
  • Reconnect / error    — the monitor itself had trouble (so silence != safety)

SETUP:
  On the VPS set two env vars (System env vars):
        TELEGRAM_TOKEN   = 123456:ABC...
        TELEGRAM_CHAT_ID = 987654321

RUN (on the VPS, in the repo folder, in its OWN window — NOT the trading loop's):
    python pinger.py
"""

import os
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import MetaTrader5 as mt5

# Reuse connect() from your executor; NO order functions are used here.
import mt5_executor as ex

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Automatically detect if live_trader uses trade_state.json or state.json
if os.path.exists("trade_state.json"):
    STATE_FILE = "trade_state.json"
else:
    STATE_FILE = "state.json"

POLL_SECONDS        = 60                 # gentle; state changes on 15-min bars anyway
MARGIN_WARN         = 0.70               # margin usage fraction to alert on
GBP_TRIPWIRE_PAIR   = "EURGBP/USDCAD"    # 2nd exit of this => exclude-GBP rule fires
GBP_EXIT_TRIP_COUNT = 2

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def notify(msg: str) -> None:
    """Send a Telegram message. Prints to console too."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}Z] {msg}"
    print(line, flush=True)
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": line},
            timeout=10,
        )
        r.raise_for_status()  # Force exception on 4xx/5xx API responses
    except Exception as e:
        response_text = ""
        try:
            if 'r' in locals():
                response_text = f" | Response: {r.text}"
        except Exception:
            pass
        print(f"[pinger] telegram send failed: {e}{response_text}", flush=True)


# ---------------------------------------------------------------------------
# STATE HELPENS
# ---------------------------------------------------------------------------
def load_state() -> dict | None:
    """
    Loads state data. Returns None if there's a file read or JSON parse error.
    This prevents false exit/entry triggers during mid-write race conditions.
    """
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            txt = f.read().strip()
        return json.loads(txt) if txt else {}
    except Exception:
        return None            # Return None to signal a mid-write race condition


def margin_usage() -> float:
    acc = mt5.account_info()
    if not acc or acc.equity <= 0:
        return 0.0
    return acc.margin / acc.equity


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main() -> None:
    ex.connect()   # raises if MT5 / Algo Trading not ready; read-only use here
    notify(f"pinger online — watching {STATE_FILE} + margin (read-only).")

    initial_state = load_state()
    # If initial load fails because of race, retry until successful setup
    while initial_state is None:
        time.sleep(1)
        initial_state = load_state()

    prev = set(initial_state.keys())
    gbp_exit_count = 0
    margin_alerted = False     # latch so we don't spam every poll over 70%

    while True:
        try:
            state_data = load_state()
            if state_data is None:
                # Mid-write collision detected. Skip this check to prevent false notifications.
                time.sleep(1)
                continue

            cur = set(state_data.keys())

            # --- exits: pairs that were held last poll and are now gone ----
            for tag in prev - cur:
                notify(f"EXIT: {tag} closed (revert / divergence / time stop).")
                if tag == GBP_TRIPWIRE_PAIR:
                    gbp_exit_count += 1
                    if gbp_exit_count >= GBP_EXIT_TRIP_COUNT:
                        notify(
                            f"GBP TRIPWIRE: {GBP_TRIPWIRE_PAIR} has now exited "
                            f"{gbp_exit_count}x. Your rule: exclude GBP pairs for "
                            f"today. Decide deliberately."
                        )

            # --- entries: new pairs since last poll ------------------------
            for tag in cur - prev:
                notify(f"ENTRY: {tag} opened.")

            # --- margin watch (latched) ------------------------------------
            mu = margin_usage()
            if mu > MARGIN_WARN and not margin_alerted:
                notify(f"MARGIN: usage {mu:.0%} > {MARGIN_WARN:.0%}. Watch the book.")
                margin_alerted = True
            elif mu <= MARGIN_WARN * 0.9:
                margin_alerted = False   # reset once it drops back down

            prev = cur

        except Exception as e:
            notify(f"pinger error (will keep running): {type(e).__name__}: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npinger stopped by user.")
    finally:
        mt5.shutdown()