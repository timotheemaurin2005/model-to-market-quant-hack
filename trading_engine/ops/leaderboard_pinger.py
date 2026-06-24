#!/usr/bin/env python3
"""
leaderboard_pinger.py  —  Status Monitor (Channel 2)

Companion to pinger.py (Channel 1). Reads the Syphonix leaderboard API, finds
YOUR row, and reports a STATUS BUCKET — SAFE / AT RISK / ELIMINATED. It pings
ONLY when the bucket CHANGES, states the fact, and appends the relevant
dials-matrix REMINDER. It never recommends a specific Z value or any parameter
change — that decision stays with you, made deliberately at the dashboard.

WHY TOKEN-PASTE, NOT UNATTENDED:
  The leaderboard API authenticates with a short-lived Cognito Bearer token
  (expires in minutes). A hardcoded token dies almost immediately, so this tool
  is built to be RUN with a fresh token when you want a read — not to poll all
  day on stale auth. Grab a fresh token from your browser's Network tab
  (the leaderboard request -> Headers -> authorization), and pass it in.

USAGE:
  export SYPHONIX_TOKEN="eyJ..."        # paste a FRESH token (no 'Bearer ')
  python leaderboard_pinger.py          # single read + bucket check
  python leaderboard_pinger.py --watch  # re-read every 5 min until token dies

  Telegram (optional, shares Channel-1 vars):
  export TELEGRAM_TOKEN=...  TELEGRAM_CHAT_ID=...

DEPS:  pip install requests
"""

import os
import sys
import json
import time

import requests

COMPETITION_ID = "0e2336e4-eca5-4922-927e-dd670ee668e1"
API_URL = (
    f"https://quanthack.syphonix.com/api/v1/competitions/"
    f"{COMPETITION_ID}/leaderboard?page=1&page_size=99999"
)
AT_RISK_BUFFER = 20          # ranks above the cut line still flagged AT RISK
STATE_PATH     = "leaderboard_state.json"   # remembers last bucket across runs
POLL_SECONDS   = 300         # --watch cadence: 5 min, gentle on their endpoint

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# --- dials-matrix REMINDERS (facts you pre-wrote, not new advice) -----------
DIALS_REMINDER = {
    "SAFE": (
        "Dials matrix: R1 is observe-and-hold. Change nothing off rank noise. "
        "The 22:00 close is the only number that decides anything."
    ),
    "AT RISK": (
        "Dials matrix: if you choose to act, SAFE LEVERS FIRST — more concurrent "
        "pairs, then ENTRY_Z toward 2.0 (never below). MARGIN_PER_PAIR is the last, "
        "smallest lever. Never below ENTRY_Z 2.0. Never out of fear. Decide at the "
        "dashboard, deliberately — not off this ping."
    ),
    "ELIMINATED": (
        "If this is real, it's decided by the close, not this snapshot. Nothing to "
        "do now but review calmly for tomorrow."
    ),
}


def notify(msg: str) -> None:
    print(msg, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10,
            )
        except Exception as e:
            print(f"[lb] telegram failed: {e}", flush=True)


def fetch_board(token: str) -> dict:
    r = requests.get(
        API_URL,
        headers={"authorization": f"Bearer {token}",
                 "accept": "application/json"},
        timeout=15,
    )
    if r.status_code == 401:
        raise SystemExit(
            "401 Unauthorized — token expired (they die in minutes). "
            "Grab a fresh one from the Network tab and re-export SYPHONIX_TOKEN."
        )
    r.raise_for_status()
    return r.json()


def bucket_for(rank: int, line: int, eliminated: bool) -> str:
    if eliminated or rank > line:
        return "ELIMINATED"
    if rank > line - AT_RISK_BUFFER:
        return "AT RISK"
    return "SAFE"


def load_last() -> str | None:
    try:
        with open(STATE_PATH) as f:
            return json.load(f).get("bucket")
    except Exception:
        return None


def save_last(bucket: str) -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump({"bucket": bucket}, f)
    except Exception:
        pass


def check_once(token: str) -> None:
    data = fetch_board(token)["data"]
    line = data["elimination_line_rank"]

    me = next((e for e in data["entries"] if e.get("is_me")), None)
    if me is None:
        notify("[lb] couldn't find your row (is_me) — are you logged into the "
               "right account for this token?")
        return

    # delayed_* is the authoritative lagged snapshot; fall back to live rank.
    rank = me.get("delayed_rank") or me["rank"]
    eliminated = me.get("is_eliminated", False)
    bucket = bucket_for(rank, line, eliminated)

    margin = line - rank   # +ve = clear of cut
    headline = (f"STATUS: {bucket} — rank #{rank} of {data['ranked_count']}, "
                f"cut at #{line} (clear by {margin}).")

    last = load_last()
    if bucket != last:
        # bucket CHANGED -> ping with the matrix reminder
        notify(f"⚠ BUCKET CHANGE: {last or 'init'} -> {bucket}\n"
               f"{headline}\n{DIALS_REMINDER[bucket]}")
        save_last(bucket)
    else:
        # no change -> quiet console line only, no Telegram spam
        print(f"[lb] no change ({bucket}). {headline}", flush=True)


def main() -> None:
    token = os.environ.get("SYPHONIX_TOKEN")
    if not token:
        raise SystemExit("Set SYPHONIX_TOKEN to a FRESH leaderboard Bearer token.")
    token = token.replace("Bearer ", "").strip()

    watch = "--watch" in sys.argv
    if not watch:
        check_once(token)
        return

    print(f"[lb] watch mode, every {POLL_SECONDS}s. Token will expire — "
          f"expect a 401 in minutes, then re-run with a fresh token.", flush=True)
    while True:
        try:
            check_once(token)
        except SystemExit as e:
            print(str(e), flush=True)
            break
        except Exception as e:
            print(f"[lb] error: {type(e).__name__}: {e}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
