# Architecture

Two loops that never call into each other: a deterministic execution loop,
and a Claude-narrated advisory loop that only ever reads the execution loop's
output.

```mermaid
flowchart TD
    subgraph Offline["OFFLINE (run once per session, not live)"]
        A1[parquet] -->|cointegration +\nOU half-life filter| A2[pair_screener.py]
        A2 -->|ranked_pairs.csv| A3[kalman_calibration.py + conformal.py]
        A3 -.->|EM hedge-ratio fit, two-piece\nasymmetric conformal calibration,\nheld-out validated| A3
        A3 -->|frozen, PASS-only| A4[kalman_config.json]
    end

    A4 -->|loaded at startup, never re-fit live| B1

    subgraph LiveLoop["LIVE EXECUTION LOOP (deterministic — no LLM calls, ever)"]
        B1[trading_engine/live_trader.py\nFX mean-reversion core,\nkalman_live.py β-filter]
        B2[sleeves/*.py\ndirectional satellites,\nown magic numbers + DRY_RUN]
        
        B1 --> B3[trading_engine/mt5_executor.py\nsizing, guardrails,\nMT5 order placement]
        B2 --> B3
        
        B1 -->|writes| B4[(state.json)]
        B2 -->|writes| B4
    end

    B4 -->|read-only| C1
    B4 -->|read-only| C2

    subgraph Advisory["ADVISORY / MACRO LAYER (Claude is a pure narrator — zero write access back into execution)"]
        C1[macro_agent/advisory/claude_analyst.py\ncomputes analytics in plain Python → Claude narrates] -->|writes| C3[report_YYYY-MM-DD.md]
        
        C4[live crypto/FX/metals feeds] --> C2[macro_agent/dashboard/app.py\nStreamlit\nevaluate_chain in plain Python → Claude narrates]
        C2 --> C5[dashboard UI]
    end

    classDef offline fill:#e2e8f0,stroke:#64748b,stroke-width:1px,color:#0f172a
    classDef live fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef advisory fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef data fill:#fef3c7,stroke:#f59e0b,stroke-width:1px,color:#78350f

    class Offline offline
    class LiveLoop live
    class Advisory advisory
    class A4,B4,C3 data
```

**Auxiliary Processes:**
- `trading_engine/ops/*.py` — out-of-band watchdogs (`pinger.py`, `leaderboard_pinger.py`) that diff `state.json` / poll the leaderboard API and alert via Telegram. They run alongside the loop, never inside it, and never write back to it.
- `research/*.py` — backtests and exploratory screeners (crypto, Donchian, metals via yfinance). Not imported by the live loop; informs parameter choices upstream of the offline calibration step.

## Why the split matters

The execution loop has a hard latency and correctness budget: a wrong call
here risks a forced liquidation, which is instant elimination under the
competition's scoring rules. It is kept fully deterministic — pure Python,
pre-fit statistical models, no network calls to an LLM in the hot path.

Claude sits one layer up, with read-only access to the *output* of that loop
(`state.json`) and to external market context (crypto/FX/metals feeds, news).
Its job is explanation and macro-context synthesis for a human operator
running the system in copilot mode — not decision-making. This is enforced
structurally, not just by convention: `claude_analyst.py` and `app.py` have no
import path back into `live_trader.py` or `mt5_executor.py`, and neither
writes to any file the execution loop reads.
