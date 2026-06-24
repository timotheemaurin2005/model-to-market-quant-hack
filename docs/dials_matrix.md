# Dials Matrix — Model to Market

**Cold-execution playbook. Every change is a response to leaderboard data, not a feeling.**
**Margin is the LAST lever, never the first. The stop-out guardrail never moves.**

---

## The levers, ranked by safety (use top-down)

| # | Lever | Variable | Effect | Adds leverage? |
|---|-------|----------|--------|----------------|
| 1 | More shots on goal | `MAX_CONCURRENT_PAIRS` ↑ | more trades, better Sharpe-obs count, more reversion captured | **No** |
| 2 | Trade more often | `ENTRY_Z` ↓ (e.g. 2.0 → 1.8) | more frequent entries | No |
| 3 | Capture bigger move | `EXIT_Z` ↓ toward 0 | larger gross per trade, better cost ratio | No |
| 4 | **More capital per trade** | `MARGIN_PER_PAIR` ↑ | bigger absolute $ per trade | **YES — last resort** |

Reach for 1 → 2 → 3 before ever touching 4. Lever 4 is the only one that shortens
your distance to the 30%-margin-level stop-out (= instant elimination), so it is
nudged, never doubled, and only while watching `cushion_to_stopout` stay large.

---

## ROUND 1 — Launch (Sun 22:00). Conservative. Gather data.

- `MARGIN_PER_PAIR = 20_000` – 25_000
- `MAX_CONCURRENT_PAIRS = 6`
- `ENTRY_Z = 2.0`, `EXIT_Z = 0.5`

**Do nothing for the first several hours.** Confirm fills, exits, and the guardrail
behave on live data. Watch — don't tune. The point of Round 1's early hours is to
*see where you actually stand*, not to react.

---

## After the R1 leaderboard (Monday, WITH data)

Read your standing first, then act — one lever at a time, observe a few hours between changes.

**If surviving AND ranking OK on return** → change **nothing**. The discipline is working.

**If surviving BUT ranking low on return** (the likely case for a market-neutral book):
1. **First — Lever 1**: raise `MAX_CONCURRENT_PAIRS` to 8–10; confirm all 28 pairs feed the screen. More independent trades, no added leverage.
2. **Then — Lever 2**: tighten `ENTRY_Z` to ~1.8. Watch trade quality / win rate.
3. **Then — Lever 3**: lower `EXIT_Z` toward 0.0–0.25 to capture a bigger move per trade (also improves the cost ratio — see spread_check).
4. **Only if 1–3 aren't enough — Lever 4**: nudge `MARGIN_PER_PAIR` up ~25–50% (e.g. 25k → 35k). Never double it. Watch `cushion_to_stopout`.

**If genuinely near the elimination cut**: the answer is still NOT max leverage. First check you're actually *trading* (pairs passing, entries firing) and that costs aren't eating you (run `spread_check.py`). A flat or cost-bleeding book is not fixed by leverage.

---

## NEVER move these (the survival edge)

- `MAX_MARGIN_USAGE = 0.85` margin-usage cap
- Divergence stop (frozen |Z| > 3.5)
- Time stop (3× half-life, capped at one round)
- Never override the screen to force a non-cointegrated pair
- Never flip `DRY_RUN = False` without a deliberate first-fill check

---

## Rules of engagement

1. **One lever at a time.** Observe ≥ a few hours before the next change.
2. **Never pre-commit a margin schedule.** No "Wednesday = $75k" — that marches you into leverage on a timer regardless of the data.
3. **Every change answers a leaderboard observation**, not anxiety.
4. **If you're tempted to change something out of fear, that's the signal to wait.** The disciplined book wins by *not* doing what frightened competitors do.
5. Margin is the last lever. Always.
