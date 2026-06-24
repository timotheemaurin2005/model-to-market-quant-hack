# Advisory Brain — Dual-Brain FX Statistical-Arbitrage System

A risk-officer briefing tool for a market-neutral FX stat-arb book. It reads the
live portfolio state, computes every analytic in plain Python, and uses an LLM
**only** to narrate the pre-computed numbers — never to calculate or to touch the
execution path.

## The Dual-Brain Architecture

A common failure mode in quant systems is putting a language model inside the live
execution loop. LLMs are non-deterministic, have latency spikes, and introduce API
failure modes into time-sensitive order routing. This system enforces a strict
separation instead.

```
┌──────────────────────────────────────────────────────────────────┐
│  EXECUTION BRAIN   (remote VPS · Python → MetaTrader 5)            │
│  • 100% deterministic stat-arb loop                               │
│  • Cointegration screen, OLS hedge ratio, Z-score entry/exit      │
│  • Hard-coded divergence + time stops                             │
│  • Writes state.json after every position change                  │
│  • NEVER calls a language model                                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │  state.json  (read-only downstream)
┌──────────────────────────────▼───────────────────────────────────┐
│  ADVISORY BRAIN    (local machine · this tool)                    │
│  • Reads state.json                                               │
│  • Computes ALL analytics in plain Python — zero LLM involvement  │
│  • Calls the LLM for narrative commentary only                    │
│  • Writes a dated Markdown report                                 │
│  • Has ZERO influence on the execution path                       │
└──────────────────────────────────────────────────────────────────┘
```

**The LLM is a narrator, never a calculator.** Every number in the report is
produced deterministically before any API call is made. The model receives final,
pre-computed values and is explicitly instructed not to recalculate anything.

## Why this design wins

The LLM is used for what it is actually good at — synthesising unstructured macro
and geopolitical context into a readable risk briefing — without compromising the
speed or safety of the execution engine. The trader gets a daily "risk officer"
briefing without adding a single millisecond of latency, or a single point of
failure, to the trade loop.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."        # never hardcode the key
python claude_analyst.py --state state.json
```

Outputs:
- **stdout** — deterministic tables + the two advisory sections
- **report_YYYY-MM-DD.md** — the dated report file

## What each layer produces

**Deterministic layer (no LLM):** per-pair hold time, reversion progress
(hold ÷ half-life), frozen cointegration parameters (α/β/μ/σ), entry Z, and a
book-level summary (open pairs, distinct currencies, averages, pairs past
half-life).

**Advisory layer (LLM narration only):**
1. *Portfolio read* — interprets the pre-computed metrics; flags positions
   approaching or past their mean-reversion window.
2. *Macro & geopolitical tail-risk briefing* — scoped explicitly to a
   market-neutral book, so it focuses on correlation-/cointegration-break risks
   and scheduled events rather than directional forecasts.

## Roadmap

- **v1.1** — inject the current date and a live-events check into the tail-risk
  prompt so it names specific scheduled releases and breaking political events.
- **v2** — read *closed* trades from `mt5.history_deals_get()` for realised-P&L,
  win-rate, and per-pair attribution analysis.

---

*The Execution Brain has no LLM dependency. This Advisory Brain is read-only and
strictly out-of-band.*
