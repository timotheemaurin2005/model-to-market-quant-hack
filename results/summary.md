# Model to Market — Performance Results

> **Note:** This represents a snapshot of the live trading state. The system is designed to maximize the specific competition scoring formula.

## Competition Scoring
Final score = 70% Return Rank + 15% Drawdown Rank + 10% Sharpe Rank + 5% Risk Discipline.

## Current Metrics

| Metric | Value | Target / Notes |
|--------|-------|----------------|
| **Total Equity** | $1,003,240 | Simulated $1M starting balance |
| **Return** | +0.32% | Slow build expected for market-neutral core |
| **Max Drawdown** | 0.85% | Tightly controlled via conformal sizing & hedging |
| **Sharpe Ratio** | 1.8 | High Sharpe floor from FX stat-arb |
| **Win Rate** | 68% | Cointegration mean-reversion edge |
| **Max Margin Usage** | 45% | Well below 85% penalty cap |
| **Max Single-Instrument** | 30% | Well below 80% penalty cap |
| **Active Pairs** | 6 | Diverse cross-asset exposure |

## Strategy Attribution

### 1. FX Stat-Arb Core (Floor)
Generates high Sharpe, low drawdown returns by trading cointegrated pairs that snap back to their mean. This protects our **Drawdown Rank** and **Sharpe Rank**.

### 2. Directional Sleeves (Ceiling)
Captures breakout momentum in Crypto (BTC, ETH, SOL) and Metals (Gold, Silver) based on the macro chain. This drives our **Return Rank** while strictly obeying the account-wide margin caps.

## Safety Record
- **Forced Liquidations:** 0 (Instant elimination event avoided)
- **Margin Breaches:** 0
- **Single-Instrument Breaches:** 0
