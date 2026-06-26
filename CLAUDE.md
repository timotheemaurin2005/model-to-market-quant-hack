# CLAUDE.md — Model to Market trading system

Read this first, every session. It is the standing brief for this project.

## What this is
A statistical-arbitrage pipeline for the **Model to Market: The Quantitative
Hack** competition (Syphonix + AI Engine, 15–27 June 2026). Simulated $1M
account, up to 30x leverage. The edge: mean-reversion on cointegrated FX/metals
pairs, position-sized by an asymmetric conformal-prediction framework, executed
manually (copilot mode) via the Syphonix GUI.

Trading opens **21 Jun 22:00 London**. Elimination rounds 22/23/24 Jun 22:00;
finals 24–26 Jun. The whole live window is ~5 days — this constrains everything.

## Scoring (what we optimise)
Final score = 70% Return Rank + 15% Drawdown Rank + 10% Sharpe Rank +
5% Risk Discipline. All are relative percentile ranks vs. the field.
- Sharpe is on 15-min equity returns; needs ≥8 valid observations or it's capped at 50.
- Forced liquidation = **instant elimination**. Risk management is existential, not cosmetic.

## Pipeline (run in this order)
```
parquet  →  pair_screener.py  →  ranked_pairs.csv
                                      │
                                      ▼
                          kalman_calibration.py  →  kalman_config.json
                                                          │
                                            (live β-filter loads this; no EM live)
```
- `pair_screener.py` — Engle-Granger cointegration + OU half-life filter on a
  60-day window, sliced by timestamp.
- `kalman_calibration.py` — EM on the hedge-ratio state [α, β], constrained and
  held-out validated, → frozen per-pair config. EM is **offline only**.

## Hard rules (do not violate)
- **Never loosen a statistical threshold to force a pass.** If no pair survives,
  report which filter killed each pair and stop. Do not relax p-value or half-life.
- **`BAR_MINUTES` must equal the parquet's real bar size.** A wrong value
  silently corrupts every half-life. Verify it from the data before trusting output.
- Always print `half_life_min` for **all** pairs, pass or fail — the distribution
  is the point, not just the winners.
- Half-life tradeable band: **120–1440 min** (~2h–24h). Below = too fast for
  15-min polling; above = won't round-trip in a 5-day competition (exit needs
  ~2 half-lives: Z 2.0 → 0.5). The 1440 ceiling is PROVISIONAL — set it from the
  observed distribution and the PnL backtest.
- Flag explicitly whether **XAUUSD/XAGUSD** lands in the band. It's the flagship
  pair and classically slow; if it's excluded, say so loudly.
- Trade **only PASS pairs** from calibration. REVIEW = looked good in-sample only.
- Risk guardrail caps (25x leverage / 85% margin / 80% single-instrument) sit
  deliberately below the penalty tiers (28x / 90% / 90%, all sustained ≥30 min).
  **Do not raise them.** The guardrail is a one-way gate: reject or pass, never resize up.
- No credentials, API keys, or `.env` files in this repo. It stays **private**
  until after Round 3 (tech-prize requirement is to make it public then).

## Strategy parameters (current)
- **Core FX Stat-Arb (`live_trader.py`)**:
  - `MAX_CONCURRENT_PAIRS = 8`
  - `ENTRY_Z = 1.75`
  - `EXIT_Z = 0.25`
  - Cointegration: ADF p < 0.05 on the spread.
- **Offensive Gold Sleeve (`directional_sleeve.py`)**:
  - `SYMBOL = XAUUSD` (Short Only)
  - `MAGIC_GOLD = 20260703`
  - `TRANCHE_MARGIN_USD = 50000.0`
  - `MAX_TRANCHES = 3`
  - `STOP_LOSS_PRICE = 4250.0` (Hard Adverse Stop)
  - `KILL_SWITCH_USD = -100000.0` (Cumulative Book PnL stop)
  - Scaling: Only add next tranche if price is lower than previous entry.
  - Data Caution: News blackout block on June 24 (GDP) & June 26 (PCE) if Gold is rallying.

## Files
- `pair_screener.py` — cointegration + half-life screen.
- `kalman_calibration.py` — offline Kalman tuning (needs `pair_screener.py` importable).
- `requirements.txt` — Python deps for the current stage.
- `outputs/` — `ranked_pairs.csv`, `kalman_config.json` land here.
- Parquet (not in this folder): `/Users/timotheemaurin/My Drive/Backtesting Log/prices.parquet`

## Vault (write the log directly — do NOT use the Obsidian MCP)
Obsidian vault: `/Users/timotheemaurin/Documents/Obsidian Vault`
The Obsidian MCP write path hangs; edit the markdown files directly on disk instead.
After each working session, update:
- `Competition/03 - Daily Log.md` — append a dated entry of what changed.
- `Competition/05 - Next Steps.md` — move done items, add new ones.

## Still to build (not yet in this folder)
Live β-filter loop (loads `kalman_config.json`, steps with `filter_update()`),
FastAPI `/signal` server, copilot dashboard, PnL backtest (`hftbacktest`) to tune
Z thresholds + max holding period, Northflank L4 deployment before 21 Jun.
