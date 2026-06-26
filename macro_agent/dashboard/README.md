# 📡 Macro Agent — Syphonix Trading Competition

A real-time macro monitoring dashboard powered by Claude claude-sonnet-4-6, built for the **Model to Market: The Quantitative Hack** competition.

## What it does

- **Live price monitoring** across all 15 competition instruments (FX, crypto, metals)
- **Macro chain detection** — tracks the inflation → USD → AUD/USD → crypto thesis in real time
- **Claude-powered analysis** — generates fresh macro reads on demand via Anthropic API
- **Smart alerts** — ranked by urgency, triggered by price thresholds
- **PCE countdown** — tracks the key catalyst with scenario analysis
- **Trade plan** — full tomorrow's execution plan with entry/SL/TP for each instrument

## The thesis

```
Higher inflation fears
    → Stronger USD (DXY breakout)
    → Tighter liquidity
    → Weaker AUD/USD
    → Crypto underperformance (SOL > BTC beta)
    → Volatility rises
```

The agent monitors all 5 chain signals simultaneously and alerts when activation occurs.

## Tech stack

- **Claude claude-sonnet-4-6** — macro interpretation engine
- **Streamlit** — dashboard UI
- **yfinance** — FX and metals prices
- **CCXT + Binance** — crypto prices
- **NewsAPI** — sentiment feed (optional)
- **Plotly** — charts

## Setup

```bash
git clone https://github.com/yourusername/macro-agent
cd macro-agent
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
streamlit run app.py
```

## Environment variables

```
ANTHROPIC_API_KEY=your_key_here
NEWS_API_KEY=your_newsapi_key_here  # optional
```

## Features

### Chain monitor
Tracks 5 confirmation signals in real time:
1. DXY breaking out (>100)
2. AUD/USD breaking down (<0.700)
3. Treasury yields rising (Fed hawkish proxy)
4. BTC lagging equities
5. SOL underperforming BTC

### Smart alerts
- PCE countdown with urgency scaling
- SOL momentum alerts
- AUD/USD breakout detection
- DXY strength confirmation

### Claude analysis
One-click macro analysis that:
- Reads all live market data
- Assesses chain activation status
- Gives specific trade recommendations
- Identifies key risks

### PCE scenarios
Pre-computed P&L for hot/in-line/cool scenarios across all positions.

## Competition context

Built for Syphonix "Model to Market: The Quantitative Hack" — $1M virtual capital, real market quotes.

**Author:** Timothée Maurin  
**Strategy:** Macro chain monitoring with PCE as primary catalyst  
**Tools:** Claude API, yfinance, CCXT, Streamlit
