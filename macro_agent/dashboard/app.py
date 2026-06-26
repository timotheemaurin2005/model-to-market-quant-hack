import streamlit as st
import anthropic
import yfinance as yf
import ccxt
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Macro Agent — Syphonix",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stMetric { background: #1a1d24; border: 1px solid #2a2e37; border-radius: 8px; padding: 0.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #e5e7eb; }
    div[data-testid="stMetricLabel"] { color: #9ca3af; }
    .chain-active { color: #22c55e; font-weight: 600; }
    .chain-watch { color: #f59e0b; font-weight: 600; }
    .chain-inactive { color: #6b7280; }
    .alert-high { background: rgba(239,68,68,0.12); border-left: 3px solid #ef4444; padding: 8px 12px; border-radius: 4px; margin: 4px 0; font-size: 13px; color: #fca5a5; }
    .alert-med { background: rgba(245,158,11,0.12); border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 4px; margin: 4px 0; font-size: 13px; color: #fcd34d; }
    .alert-low { background: rgba(59,130,246,0.12); border-left: 3px solid #3b82f6; padding: 8px 12px; border-radius: 4px; margin: 4px 0; font-size: 13px; color: #93c5fd; }
    .pnl-pos { color: #22c55e; font-weight: 600; }
    .pnl-neg { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ─────────────────────────────────────────────

import math

def _is_bad(x):
    """True if a value is missing/NaN/non-finite — i.e. NOT a usable price."""
    try:
        return x is None or math.isnan(float(x)) or not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


@st.cache_data(ttl=60)
def fetch_crypto_prices():
    try:
        exchange = ccxt.binance()
        symbols = {
            "BTC": "BTC/USDT",
            "ETH": "ETH/USDT",
            "SOL": "SOL/USDT",
            "XRP": "XRP/USDT",
        }
        prices = {}
        for name, sym in symbols.items():
            ticker = exchange.fetch_ticker(sym)
            last = ticker["last"]
            prices[name] = {
                "price": last,
                "change_pct": ticker["percentage"],
                "high": ticker["high"],
                "low": ticker["low"],
                "volume": ticker["quoteVolume"],
                "ok": not _is_bad(last),
            }
        return prices
    except Exception:
        # Clearly-flagged fallback (ok=False) so the UI never shows these as live.
        return {
            "BTC": {"price": None, "change_pct": None, "high": None, "low": None, "volume": 0, "ok": False},
            "ETH": {"price": None, "change_pct": None, "high": None, "low": None, "volume": 0, "ok": False},
            "SOL": {"price": None, "change_pct": None, "high": None, "low": None, "volume": 0, "ok": False},
            "XRP": {"price": None, "change_pct": None, "high": None, "low": None, "volume": 0, "ok": False},
        }


@st.cache_data(ttl=60)
def fetch_fx_prices():
    try:
        tickers = yf.download(
            ["AUDUSD=X", "USDJPY=X", "USDCAD=X", "DX-Y.NYB"],
            period="2d", interval="1h", progress=False
        )
        close = tickers["Close"].iloc[-1]
        prev = tickers["Close"].iloc[-2]
        result = {}
        for sym, key in [("AUDUSD=X", "AUDUSD"), ("USDJPY=X", "USDJPY"),
                          ("USDCAD=X", "USDCAD"), ("DX-Y.NYB", "DXY")]:
            p = float(close[sym]) if sym in close.index else None
            pv = float(prev[sym]) if sym in prev.index else None
            # A partial yfinance failure returns NaN cells (not an exception),
            # so guard every value explicitly and flag bad ones as ok=False.
            if _is_bad(p) or _is_bad(pv) or pv == 0:
                result[key] = {"price": None, "change_pct": None, "ok": False}
            else:
                result[key] = {"price": p, "change_pct": (p - pv) / pv * 100, "ok": True}
        return result
    except Exception:
        # Total failure -> clearly-flagged fallback so the UI shows "no data",
        # not stale numbers masquerading as live.
        return {
            "AUDUSD": {"price": None, "change_pct": None, "ok": False},
            "USDJPY": {"price": None, "change_pct": None, "ok": False},
            "USDCAD": {"price": None, "change_pct": None, "ok": False},
            "DXY": {"price": None, "change_pct": None, "ok": False},
        }


@st.cache_data(ttl=60)
def fetch_metals_prices():
    try:
        tickers = yf.download(
            ["GC=F", "SI=F"],
            period="2d", interval="1h", progress=False
        )
        close = tickers["Close"].iloc[-1]
        prev = tickers["Close"].iloc[-2]
        result = {}
        for sym, key in [("GC=F", "XAU"), ("SI=F", "XAG")]:
            p = float(close[sym]) if sym in close.index else None
            pv = float(prev[sym]) if sym in prev.index else None
            if _is_bad(p) or _is_bad(pv) or pv == 0:
                result[key] = {"price": None, "change_pct": None, "ok": False}
            else:
                result[key] = {"price": p, "change_pct": (p - pv) / pv * 100, "ok": True}
        return result
    except Exception:
        return {
            "XAU": {"price": None, "change_pct": None, "ok": False},
            "XAG": {"price": None, "change_pct": None, "ok": False},
        }


@st.cache_data(ttl=300)
def fetch_news_sentiment():
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return [
            {"title": "Dollar surges to 1-year high as Fed signals rate hike", "sentiment": "bearish_risk", "source": "Reuters"},
            {"title": "Gold falls third session as hawkish Fed bets lift dollar", "sentiment": "bearish_gold", "source": "MT Newswires"},
            {"title": "SOL underperforms BTC as crypto risk-off continues", "sentiment": "bearish_crypto", "source": "CoinDesk"},
            {"title": "BofA, Deutsche Bank forecast September Fed rate hike", "sentiment": "hawkish_fed", "source": "Bloomberg"},
            {"title": "PCE inflation data due Wednesday — consensus +0.5% MoM", "sentiment": "key_event", "source": "BEA"},
        ]
    try:
        url = f"https://newsapi.org/v2/everything?q=Federal+Reserve+inflation+gold+dollar&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
        r = requests.get(url, timeout=5)
        articles = r.json().get("articles", [])
        return [{"title": a["title"], "sentiment": "neutral", "source": a["source"]["name"]} for a in articles[:5]]
    except Exception:
        return []


# ── Chain signal evaluation ───────────────────────────────────

def evaluate_chain(crypto, fx, metals):
    """Returns (signals, active_count, nodata_count).
    A signal is only 'active'/'inactive' if its inputs are real. Missing/NaN
    inputs -> 'no_data' and are NEVER counted toward active_count. This is the
    fix for the bug where a failed fetch showed up as a live chain signal."""
    signals = {}

    dxy_ok = fx.get("DXY", {}).get("ok", False)
    dxy = fx.get("DXY", {}).get("price")
    if not dxy_ok:
        signals["DXY breaking out"] = ("no_data", "no data — fetch failed")
    else:
        signals["DXY breaking out"] = ("active", f"{dxy:.2f} — above 100") if dxy > 100 else ("watch", f"{dxy:.2f}")

    aud_ok = fx.get("AUDUSD", {}).get("ok", False)
    audusd = fx.get("AUDUSD", {}).get("price")
    if not aud_ok:
        signals["AUD/USD breaking down"] = ("no_data", "no data — fetch failed")
    else:
        signals["AUD/USD breaking down"] = ("active", f"{audusd:.4f} — below 0.700") if audusd < 0.70 else \
                                            ("watch", f"{audusd:.4f} — approaching 0.700") if audusd < 0.705 else \
                                            ("inactive", f"{audusd:.4f}")

    # Treasury-yield leg has no live data source wired in — be honest about that
    # rather than proxying it off DXY (which double-counts the dollar signal).
    signals["Treasury yields rising"] = ("no_data", "no 10Y feed wired — verify manually")

    btc_ok = crypto.get("BTC", {}).get("ok", False)
    sol_ok = crypto.get("SOL", {}).get("ok", False)
    btc_chg = crypto.get("BTC", {}).get("change_pct")
    sol_chg = crypto.get("SOL", {}).get("change_pct")

    if not btc_ok:
        signals["BTC lagging equities"] = ("no_data", "no data — fetch failed")
    else:
        signals["BTC lagging equities"] = ("active", f"BTC {btc_chg:+.1f}%") if btc_chg < -0.5 else ("inactive", f"BTC {btc_chg:+.1f}%")

    if not (btc_ok and sol_ok):
        signals["SOL underperforming BTC"] = ("no_data", "no data — fetch failed")
    else:
        spread = sol_chg - btc_chg
        signals["SOL underperforming BTC"] = ("active", f"SOL {spread:+.1f}% vs BTC") if spread < -1 else \
                                              ("watch", f"Spread: {spread:+.1f}%") if spread < 0 else \
                                              ("inactive", f"Spread: {spread:+.1f}%")

    active_count = sum(1 for s, _ in signals.values() if s == "active")
    nodata_count = sum(1 for s, _ in signals.values() if s == "no_data")
    return signals, active_count, nodata_count


# ── Claude macro analysis ─────────────────────────────────────

def get_claude_analysis(crypto, fx, metals, chain_signals, active_count, positions):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    def line(label, d, key, fmt):
        if not d.get("ok", False) or _is_bad(d.get(key)):
            return f"- {label}: NO DATA"
        return f"- {label}: {fmt.format(d[key])}" + (
            f" ({d['change_pct']:+.2f}%)" if not _is_bad(d.get("change_pct")) else "")

    market_context = "Current market data (NO DATA = fetch failed, do not infer a value):\n" + "\n".join([
        line("BTC", crypto['BTC'], 'price', "${:,.0f}"),
        line("SOL", crypto['SOL'], 'price', "${:.2f}"),
        line("ETH", crypto['ETH'], 'price', "${:,.0f}"),
        line("AUD/USD", fx['AUDUSD'], 'price', "{:.4f}"),
        line("DXY", fx['DXY'], 'price', "{:.2f}"),
        line("XAU", metals['XAU'], 'price', "${:,.0f}"),
        line("XAG", metals['XAG'], 'price', "${:.2f}"),
    ])
    chain_lines = "\n".join([f"- {sig}: {status.upper()} ({detail})"
                             for sig, (status, detail) in chain_signals.items()])

    system = (
        "You are a macro analyst for a simulated trading competition (not real money). "
        "You assess whether a stated macro thesis chain is ACTUALLY transmitting in the live data — "
        "you do not assume it is, and you do not give buy/sell/size instructions. "
        "If a signal is NO DATA, never treat it as confirming or denying the thesis. "
        "If signals diverge (e.g. dollar up but crypto not following), say the chain is BREAKING and explain. "
        "Be direct and specific; no hedging filler."
    )
    prompt = f"""{market_context}

Macro chain signals (computed):
{chain_lines}
Active signals: {active_count}/5

Thesis chain: inflation fears -> stronger USD -> weaker AUD/USD -> crypto underperformance (SOL>BTC beta) -> vol rises

In 3-4 sentences:
1. Is the chain transmitting, partially transmitting, or breaking right now? Cite which signals confirm vs diverge.
2. What single thing would most change this read in the next 8 hours?
3. The biggest risk to the thesis as stated.
Do not recommend specific trades, sizes, entries, or stops."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Analysis unavailable — check API key. Error: {str(e)}"


# ── PCE countdown ─────────────────────────────────────────────

def get_pce_countdown():
    # PCE releases 08:30 ET = 12:30 UTC. (NOTE: confirm the exact date/time against
    # a live calendar — release schedules move.) In French time (CEST, UTC+2) that
    # is 14:30, not 13:30. Compute from a UTC anchor to avoid the timezone slip.
    from datetime import timezone
    now = datetime.now(timezone.utc)
    pce_time_utc = datetime(2026, 6, 25, 12, 30, tzinfo=timezone.utc)  # 08:30 ET
    delta = pce_time_utc - now
    if delta.total_seconds() < 0:
        return "PCE RELEASED", 0
    hours = int(delta.total_seconds() // 3600)
    mins = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {mins}m", delta.total_seconds()


# ── Session state ─────────────────────────────────────────────

if "positions" not in st.session_state:
    st.session_state.positions = [
        {"instrument": "BTC/USD", "direction": "Short", "size": 2.88, "entry": 62531, "sl": 71910, "tp": None, "pnl": "+$333"},
        {"instrument": "ETH/USD", "direction": "Long", "size": 18, "entry": 1659.97, "sl": 1410.98, "tp": None, "pnl": "+$29"},
        {"instrument": "AUD/USD", "direction": "Short", "size": 10, "entry": 0.6917, "sl": 0.6960, "tp": 0.6820, "pnl": "-$98"},
        {"instrument": "SOL/USD", "direction": "Short", "size": 50, "entry": 68.80, "sl": 72.91, "tp": 65.00, "pnl": "-$80"},
    ]

if "analysis" not in st.session_state:
    st.session_state.analysis = "Click 'Get fresh analysis' to generate Claude's macro read."

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()


# ── Main layout ───────────────────────────────────────────────

col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.markdown("## 📡 Macro Agent — Syphonix Competition")
    st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')} French time")
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

st.divider()

# Load data
with st.spinner("Fetching live market data..."):
    crypto = fetch_crypto_prices()
    fx = fetch_fx_prices()
    metals = fetch_metals_prices()
    news = fetch_news_sentiment()
    chain_signals, active_count, nodata_count = evaluate_chain(crypto, fx, metals)
    countdown, seconds_left = get_pce_countdown()

# ── PCE banner ────────────────────────────────────────────────
pce_color = "#ef4444" if seconds_left < 3600 * 3 else "#f59e0b" if seconds_left < 3600 * 12 else "#3b82f6"
st.markdown(f"""
<div style="background: {pce_color}15; border: 1px solid {pce_color}40; border-radius: 8px;
     padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
    <div>
        <span style="font-weight: 600; color: {pce_color};">🔥 PCE INFLATION PRINT</span>
        <span style="color: #6b7280; font-size: 13px; margin-left: 12px;">Wed 25 Jun · 12:30 UTC (14:30 French) · consensus +0.5% MoM · verify on live calendar</span>
    </div>
    <div style="font-size: 22px; font-weight: 700; color: {pce_color};">{countdown}</div>
</div>
""", unsafe_allow_html=True)

# ── Price grid ────────────────────────────────────────────────
st.markdown("### Market snapshot")
cols = st.columns(8)

def _fmt(d, key, kind):
    """Format a price/change cell, or '—' if the value is flagged bad."""
    if not d.get("ok", False) or _is_bad(d.get(key)):
        return None
    v = d[key]
    if kind == "usd0":   return f"${v:,.0f}"
    if kind == "usd2":   return f"${v:.2f}"
    if kind == "usd4":   return f"${v:.4f}"
    if kind == "fx4":    return f"{v:.4f}"
    if kind == "fx2":    return f"{v:.2f}"
    if kind == "pct":    return f"{v:+.2f}%"
    return str(v)

assets = [
    ("BTC", _fmt(crypto['BTC'], 'price', 'usd0'), _fmt(crypto['BTC'], 'change_pct', 'pct')),
    ("SOL", _fmt(crypto['SOL'], 'price', 'usd2'), _fmt(crypto['SOL'], 'change_pct', 'pct')),
    ("ETH", _fmt(crypto['ETH'], 'price', 'usd0'), _fmt(crypto['ETH'], 'change_pct', 'pct')),
    ("XRP", _fmt(crypto['XRP'], 'price', 'usd4'), _fmt(crypto['XRP'], 'change_pct', 'pct')),
    ("AUD/USD", _fmt(fx['AUDUSD'], 'price', 'fx4'), _fmt(fx['AUDUSD'], 'change_pct', 'pct')),
    ("DXY", _fmt(fx['DXY'], 'price', 'fx2'), _fmt(fx['DXY'], 'change_pct', 'pct')),
    ("XAU", _fmt(metals['XAU'], 'price', 'usd0'), _fmt(metals['XAU'], 'change_pct', 'pct')),
    ("XAG", _fmt(metals['XAG'], 'price', 'usd2'), _fmt(metals['XAG'], 'change_pct', 'pct')),
]

for col, (name, price, chg) in zip(cols, assets):
    if price is None:
        col.metric(name, "—", "no data")
    else:
        col.metric(name, price, chg if chg is not None else "—",
                   delta_color="inverse" if name == "DXY" else "normal")

st.divider()

# ── Chain + Alerts ────────────────────────────────────────────
col_chain, col_alerts = st.columns([1, 1])

with col_chain:
    st.markdown("### Macro chain")
    status_map = {"active": "🟢 Active", "watch": "🟡 Watch", "inactive": "⚪ Inactive"}
    color_map = {"active": "chain-active", "watch": "chain-watch", "inactive": "chain-inactive"}

    chain_color = "#22c55e" if active_count == 5 else "#f59e0b" if active_count >= 3 else "#6b7280"
    nodata_note = f" · {nodata_count} signal(s) NO DATA" if nodata_count else ""
    st.markdown(f"""
    <div style="background: {chain_color}15; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
        <span style="font-size: 28px; font-weight: 700; color: {chain_color};">{active_count}/5</span>
        <span style="color: #6b7280; font-size: 13px; margin-left: 8px;">signals active{nodata_note} — {'Chain fully activated' if active_count == 5 else 'Chain partially active' if active_count >= 3 else 'Chain not activated'}</span>
    </div>
    """, unsafe_allow_html=True)

    for signal, (status, detail) in chain_signals.items():
        icon = {"active": "🟢", "watch": "🟡", "inactive": "⚪", "no_data": "⛔"}.get(status, "⚪")
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 0.5px solid #2a2e37; font-size: 13px;">
            <span>{icon} {signal}</span>
            <span style="color: #6b7280;">{detail}</span>
        </div>
        """, unsafe_allow_html=True)

with col_alerts:
    st.markdown("### Alerts")

    sol = crypto['SOL']
    audd = fx['AUDUSD']
    dxyd = fx['DXY']

    alerts = []
    if seconds_left and seconds_left < 3600 * 14:
        alerts.append(("high", f"⚡ PCE in {countdown} — elevated volatility window. Confirm release time on a live calendar."))
    # Observational alerts only — no sizing/fire instructions. The dashboard reports
    # what the data shows; trade decisions stay deliberate and human.
    if sol.get("ok") and not _is_bad(sol.get("change_pct")) and sol["change_pct"] < -3:
        alerts.append(("high", f"📉 SOL {sol['change_pct']:+.1f}% — moving hard; thesis leg (SOL underperformance) consistent. Watch, don't chase."))
    if audd.get("ok") and not _is_bad(audd.get("price")):
        if audd["price"] < 0.695:
            alerts.append(("high", f"💱 AUD/USD {audd['price']:.4f} — below 0.695, thesis-consistent. Note: move may be largely priced."))
        elif audd["price"] < 0.700:
            alerts.append(("med", f"💱 AUD/USD {audd['price']:.4f} — below 0.700, approaching thesis level."))
    if dxyd.get("ok") and not _is_bad(dxyd.get("price")) and dxyd["price"] > 101:
        alerts.append(("med", f"💵 DXY {dxyd['price']:.2f} — dollar strength consistent with USD↑ leg."))
    if nodata_count:
        alerts.append(("med", f"⚠️ {nodata_count} chain signal(s) have NO DATA — fix the feed before trusting the chain status."))
    if not alerts:
        alerts.append(("low", "No threshold alerts. Chain is being monitored; nothing flashing."))

    for level, msg in alerts[:5]:
        st.markdown(f'<div class="alert-{level}">{msg}</div>', unsafe_allow_html=True)

st.divider()

# ── Positions + Calendar ──────────────────────────────────────
col_pos, col_cal = st.columns([1, 1])

with col_pos:
    st.markdown("### Current positions")
    total_pnl = 0
    for pos in st.session_state.positions:
        pnl_val = float(pos["pnl"].replace("$", "").replace("+", "").replace(",", ""))
        total_pnl += pnl_val
        color = "#22c55e" if pnl_val >= 0 else "#ef4444"
        dir_color = "#ef4444" if pos["direction"] == "Short" else "#22c55e"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center;
             padding: 8px 0; border-bottom: 0.5px solid #2a2e37; font-size: 13px;">
            <span style="font-weight: 500; min-width: 90px;">{pos['instrument']}</span>
            <span style="background: {dir_color}20; color: {dir_color}; padding: 2px 8px;
                  border-radius: 4px; font-size: 11px; font-weight: 600;">{pos['direction']}</span>
            <span style="color: #6b7280;">{pos['size']} lots</span>
            <span style="color: #6b7280;">@ {pos['entry']}</span>
            <span style="color: {color}; font-weight: 600;">{pos['pnl']}</span>
        </div>
        """, unsafe_allow_html=True)

    # NOTE: this is a manually-set display baseline, not a live account read.
    # Update EQUITY_BASE to your actual equity, or wire in a live account feed.
    EQUITY_BASE = 1_000_000
    pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; padding: 10px 0; margin-top: 4px;">
        <span style="font-weight: 600;">Display equity (manual baseline)</span>
        <span style="font-size: 16px; font-weight: 700; color: {pnl_color};">
            ${EQUITY_BASE + total_pnl:,.0f} ({(EQUITY_BASE + total_pnl - 1000000) / 10000:+.2f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_cal:
    st.markdown("### Economic calendar")
    events = [
        ("Asian", "🟡 MED", "Asian session — metals physical-bid window (verify China not on holiday)"),
        ("07:00", "🟡 MED", "European open — liquidity returns"),
        ("12:30", "🔴 HIGH", "PCE inflation print (UTC) — main catalyst; confirm time on live calendar"),
        ("intraday", "🟡 MED", "Watch DXY/crypto for chain transmission vs divergence"),
        ("21:00", "🔴 HIGH", "Round snapshot (UTC) — equity measured; decide flat vs deliberate hold"),
    ]
    for time_str, impact, desc in events:
        weight = "700" if "HIGH" in impact else "400"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 7px 0;
             border-bottom: 0.5px solid #2a2e37; font-size: 13px;">
            <span style="min-width: 45px; color: #6b7280; font-size: 11px;">{time_str}</span>
            <span style="font-size: 11px; font-weight: {weight};">{impact}</span>
            <span style="color: #cbd5e1;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Claude analysis ───────────────────────────────────────────
st.markdown("### Claude macro analysis")
col_analysis, col_btn = st.columns([5, 1])

with col_analysis:
    st.info(st.session_state.analysis)

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧠 Get analysis", use_container_width=True):
        with st.spinner("Asking Claude..."):
            st.session_state.analysis = get_claude_analysis(
                crypto, fx, metals, chain_signals, active_count,
                st.session_state.positions
            )
        st.rerun()

st.divider()

# ── News sentiment ────────────────────────────────────────────
st.markdown("### News feed")
cols = st.columns(len(news) if news else 1)
sentiment_colors = {
    "bearish_risk": "#ef4444",
    "bearish_gold": "#f59e0b",
    "bearish_crypto": "#ef4444",
    "hawkish_fed": "#ef4444",
    "key_event": "#3b82f6",
    "neutral": "#6b7280"
}
for col, item in zip(cols, news):
    color = sentiment_colors.get(item.get("sentiment", "neutral"), "#6b7280")
    col.markdown(f"""
    <div style="background: {color}10; border: 0.5px solid {color}30; border-radius: 8px;
         padding: 10px; font-size: 12px; height: 80px; overflow: hidden;">
        <div style="color: {color}; font-weight: 600; font-size: 10px; margin-bottom: 4px;">{item['source']}</div>
        <div style="color: #cbd5e1; line-height: 1.4;">{item['title'][:80]}...</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Watch list (informational — not a trade plan) ─────────────
st.markdown("### Catalyst watch list")
st.caption("Events to watch and what each would *signal* — not a pre-committed trade plan. "
           "Sizing and entries stay deliberate and human, decided live with full risk context.")

watch_data = {
    "Time (UTC)": ["12:30", "12:30", "ongoing", "ongoing", "21:00"],
    "Event": ["PCE inflation print", "PCE prices detail", "DXY direction", "Crypto vs DXY", "Round snapshot"],
    "What a HOT read would signal": [
        "dollar-up pressure, thesis-supportive", "more hike odds → metals/crypto headwind",
        "USD↑ leg confirming", "if crypto follows DXY down, chain transmitting", "—"],
    "What a SOFT read would signal": [
        "dollar-down, thesis breaking", "hike odds fade → metals/crypto relief",
        "USD↑ leg failing", "if crypto rallies on strong DXY, chain BROKEN", "—"],
    "Note": ["confirm exact time on live calendar", "watch core MoM", "—",
             "the divergence to watch for", "be flat or sized deliberately into it"],
}
st.dataframe(pd.DataFrame(watch_data), use_container_width=True, hide_index=True)

st.caption("Reminder: scoring rewards return rank + low drawdown + Sharpe. A single oversized "
           "directional bet into a print spikes drawdown/vol — the metrics that sink rank. "
           "This dashboard is a thinking aid; it does not size or place trades.")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Macro Agent v1.0 — Built for Syphonix 'Model to Market' competition · Powered by Claude claude-sonnet-4-6")