# Model to Market — Cointegration Stat-Arb + Claude Macro Agent

A market-neutral pairs-trading engine paired with a Claude-powered macro
monitoring layer, built for the **Model to Market: The Quantitative Hack**
competition (Syphonix + AI Engine, 15–27 June 2026). Simulated $1M account,
up to 30x leverage, copilot-mode execution via the Syphonix GUI.

## The story

Most competition entries will lean on directional bets — momentum, breakouts,
a view on where gold or BTC goes next. This system's core book deliberately
doesn't take a view. It trades the **statistical relationship between
correlated instruments** (e.g. GBPUSD/USDJPY, XAUUSD/XAGUSD), entering only
when that relationship is provably cointegrated and has snapped further from
its mean than an asymmetric, calibrated confidence interval allows — then
exits as it reverts. Directional "sleeves" exist alongside it, but they're
satellites: the core engine doesn't need the market to go anywhere, just to
keep being mean-reverting.

The two pieces of novelty worth a judge's attention:

### 1. Conformal-prediction-sized entries, not fixed Z-thresholds
([`trading_engine/conformal.py`](trading_engine/conformal.py))

Most pairs-trading code sizes entries off a single static Z-score (enter at
|Z| > 2, exit at |Z| < 0.5, the same band for every pair). Real cointegrated
spreads are **not symmetric** — a spread that reverts cleanly when it's wide
to the upside can behave completely differently to the downside (regime
shifts, skewed liquidity, asymmetric carry). This repo implements a two-piece
modal split-conformal predictor (Algorithm 1 from the asymmetric conformal
prediction / modal-regression literature) that fits **separate calibrated
confidence bands for each side of the spread**, validated out-of-sample on a
held-out chronological tail before a pair is ever traded live. A pair that
fails out-of-sample coverage is marked `REVIEW`, not `PASS` — it never reaches
the live book. Combined with the Kalman-filtered hedge ratio
([`kalman_calibration.py`](trading_engine/kalman_calibration.py),
[`kalman_live.py`](trading_engine/kalman_live.py)), entries are sized off a
distribution the data actually supports, not an assumption borrowed from a
textbook example.

### 2. A Claude macro chain monitor, narrating without deciding
([`macro_agent/`](macro_agent/))

The execution side ([`trading_engine/`](trading_engine/)) is a fully
deterministic loop — it never calls a language model. Sitting beside it is a
**dual-brain advisory layer**: a Streamlit dashboard
([`macro_agent/dashboard/app.py`](macro_agent/dashboard/app.py)) that
evaluates a 5-signal cross-asset "macro chain" (crypto, FX, metals
correlation regime) in plain Python, then hands the *pre-computed* numbers to
Claude for narrative synthesis — what's it mean, what's worth watching, what
isn't. A second component
([`macro_agent/advisory/claude_analyst.py`](macro_agent/advisory/claude_analyst.py))
reads the live book's `state.json` and produces a dated markdown report the
same way. In both cases, Claude is a pure narrator: it receives structured,
already-computed context and produces commentary — it never sees a path back
into the execution layer, and it makes no sizing or entry/exit decisions. That
separation is deliberate: it's the one place an LLM is genuinely useful in a
high-frequency, latency-sensitive trading loop — explaining risk to a human,
not taking it on.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data-flow diagram.

```
trading_engine/   live_trader.py, mt5_executor.py, pair_screener.py,
                  kalman_calibration.py, kalman_live.py, conformal.py,
                  signal_server.py, and ops/ (monitoring + diagnostics)
sleeves/          directional/momentum satellite strategies
                  (gold short, silver, BTC/ETH, GBP, Donchian breakout)
macro_agent/      dashboard/ (Streamlit) + advisory/ (Claude narrator)
research/         backtests + screeners not part of the live pipeline
docs/             supporting reference docs (parameter tuning notes)
tests/            pytest suite for conformal.py and kalman_live.py
```

## Scoring this was built for

Final score = 70% Return Rank + 15% Drawdown Rank + 10% Sharpe Rank + 5% Risk
Discipline, all relative percentile ranks vs. the field. Forced liquidation is
instant elimination — which is why guardrail caps (25x leverage / 85% margin /
80% single-instrument) sit deliberately below the platform's own penalty
tiers, and why the conformal layer exists at all: a calibrated, validated
entry threshold is a drawdown-control mechanism as much as an alpha source.

## Hard rules this codebase enforces

- Never loosen a statistical threshold to force a pair to pass. Failed pairs
  are reported, not hidden.
- Cointegration screen (`pair_screener.py`) and conformal calibration
  (`conformal.py` / `kalman_calibration.py`) run offline; the live loop only
  ever loads frozen, validated config.
- Only `PASS` pairs from calibration are tradeable. `REVIEW` means in-sample
  only — it never reaches `live_trader.py`.

See [CLAUDE.md](CLAUDE.md) for the complete standing brief, including current
strategy parameters and the half-life tradeable band.

## AI & ML components

This system uses AI/ML at **three distinct levels**, each with a clear
architectural boundary:

### Statistical ML (signal generation & risk gating)

| Component | Technique | Role | File |
|-----------|-----------|------|------|
| **Conformal predictor** | Two-piece modal split-conformal prediction (Rubio & Steel) | Fits asymmetric calibrated confidence bands per spread side; held-out coverage validation gates PASS/REVIEW | [`conformal.py`](trading_engine/conformal.py) |
| **Kalman filter** | EM-calibrated online state-space model (frozen Q/R) | Dynamic hedge-ratio estimation; adapts β to non-stationary spread drift without live re-fitting | [`kalman_calibration.py`](trading_engine/kalman_calibration.py), [`kalman_live.py`](trading_engine/kalman_live.py) |
| **Cointegration screen** | Symmetric Engle-Granger (worse of both orderings) + OU half-life | Only the worst-direction p-value counts; rejects spurious pairs that pass one ordering but fail the other | [`pair_screener.py`](trading_engine/pair_screener.py), [`live_trader.py`](trading_engine/live_trader.py) |

These models are fit **offline** on historical data with chronological
train/calibration/holdout splits (never random). The live loop loads frozen,
validated config only — no online learning in the execution path.

### LLM integration (advisory layer — zero execution authority)

| Component | Model | Role | File |
|-----------|-------|------|------|
| **Portfolio analyst** | Claude Sonnet 4 | Narrates pre-computed portfolio state (reversion progress, timing risk, book concentration) | [`claude_analyst.py`](macro_agent/advisory/claude_analyst.py) |
| **Tail-risk officer** | Claude Sonnet 4 | Identifies macro/geopolitical risks to spread relationships — no directional calls | [`claude_analyst.py`](macro_agent/advisory/claude_analyst.py) |
| **Macro chain narrator** | Claude Sonnet 4 | Assesses whether a 5-signal cross-asset thesis chain is transmitting in live data | [`dashboard/app.py`](macro_agent/dashboard/app.py) |
| **Signal narrative** | Nemotron 120B (via Doubleword) | 2-sentence trade rationale from pre-computed signal parameters; explainability layer for operator trust | [`signal_server.py`](trading_engine/signal_server.py) |

**Design principle: "Narrator, not decider."** Every LLM call receives
pre-computed numbers as input and produces human-readable commentary as output.
No LLM has a write path back into the execution layer. This is enforced
structurally (no import path from advisory → execution) and verified by
inspection of the dependency graph.

### Signal API (structured copilot interface)

The FastAPI signal server ([`signal_server.py`](trading_engine/signal_server.py),
1,100 LOC) provides machine-readable endpoints:

- `GET /signal` — formatted signal box with Nemotron narrative
- `GET /signal/json` — structured JSON (Pydantic-validated) for programmatic consumption
- `GET /watchlist` — all 5 watchlist pairs with Z-scores, tiers, action hints
- `GET /health` — guardrail caps and model info

Tiered conviction sizing modulates position size by conformal interval width
(wider interval → more uncertainty → smaller size), and REVIEW pairs are
capped at LOW tier regardless of Z-score.

## Key system design decisions

| Decision | Rationale |
|----------|-----------|
| **Two-loop separation** (execution vs. advisory) | Execution has a hard latency and correctness budget; an LLM in the hot path adds latency and non-determinism that risks forced liquidation (= instant elimination). Claude sits one layer up with read-only access. |
| **Conformal prediction, not fixed Z-thresholds** | Real spreads are asymmetric. A fixed Z=2 entry band ignores skew, regime shifts, and asymmetric carry. The conformal predictor fits separate calibrated bands per side, validated OOS. |
| **Magic-number separation** | FX core and directional sleeves share a single MT5 account but use distinct magic numbers (`MAGIC_FX=20260631`, `MAGIC_DIR=20260621`). Each book can reconcile its own positions without interfering with the other. |
| **Atomic state writes** | `os.fsync()` + `os.replace()` prevents state corruption if the process crashes between order placement and persistence. A half-written `state.json` could orphan live positions. |
| **Offline calibration → frozen config** | Pair screening, Kalman EM, and conformal fitting run once per session, not live. The live loop loads validated output and never re-fits. This eliminates lookahead bias and keeps the execution path deterministic. |
| **11 numbered safety filters** | Each filter is tagged inline (`[1]`–`[11]`) in `live_trader.py`. They cover: pair cap, symbol overlap, external positions, reentry deferral, hedge-ratio drift, emergency SL, persistence streak, symmetric cointegration, positive beta, OOS holdout, and cost gate. |
| **Per-call `dry_run` override** | `mt5_executor.place_order()` accepts a per-call `dry_run` flag. The FX core runs live while a new sleeve can be dry-run-tested on the same account simultaneously. |
| **Margin-as-last-lever playbook** | [`docs/dials_matrix.md`](docs/dials_matrix.md) ranks every tunable by risk. Margin is lever 4 of 4. The system trades more pairs, trades more often, or captures a bigger move before ever increasing margin. |

## Testing

```bash
pytest tests/ -v --tb=short
```

The test suite covers:
- `test_conformal.py` — two-piece conformal predictor correctness, edge cases, coverage guarantees
- `test_kalman_live.py` — Kalman filter stepping, frozen-config loading, β estimation
- `test_gold_strategy.py` — directional sleeve sizing and guard logic

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real keys — never commit .env
```

Each subsystem also has its own narrower `requirements.txt`
(`macro_agent/dashboard/requirements.txt`, `macro_agent/advisory/requirement.txt`).

## Running

```bash
# Core FX stat-arb loop (on the VPS, live MT5 connection)
python trading_engine/live_trader.py

# Dry-run wrapper (forces DRY_RUN=True regardless of mt5_executor's default)
python trading_engine/ops/dry_loop.py

# Macro dashboard
streamlit run macro_agent/dashboard/app.py

# Tests
pytest
```
