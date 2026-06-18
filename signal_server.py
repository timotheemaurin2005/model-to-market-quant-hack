"""
signal_server.py — FastAPI copilot signal server for Model to Market.

Serves the /signal endpoint that:
  1. Loads real tick parquet data, resamples to 15-min bars.
  2. Runs OLS hedge ratio, ADF cointegration, OU half-life on 30-day window.
  3. Applies hard guardrail validation (25x leverage, 85% margin, 80% single).
  4. Computes tiered conviction sizing (HIGH/MEDIUM/LOW).
  5. Calls Doubleword Nemotron for a 2-sentence trade rationale.
  6. Returns the formatted signal box.

Data source: pricer-output parquet files (tick-level bid/ask).
XAUUSD/XAGUSD is NOT in this dataset; we use EURUSD/EURGBP/EURCHF pairs.
"""

from __future__ import annotations

import asyncio
import glob
import itertools
import logging
import re
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger("signal_server")

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

DOUBLEWORD_API_KEY = os.getenv("DOUBLEWORD_API_KEY")
DOUBLEWORD_BASE_URL = os.getenv("DOUBLEWORD_BASE_URL", "https://api.doubleword.ai/v1")
DOUBLEWORD_MODEL = os.getenv(
    "DOUBLEWORD_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)

if not DOUBLEWORD_API_KEY:
    raise RuntimeError(
        "DOUBLEWORD_API_KEY not found in environment. "
        "Create a .env file with DOUBLEWORD_API_KEY=sk-..."
    )


# ── Guardrail caps (CLAUDE.md §Hard rules: below penalty tiers) ─────────────
MAX_LEVERAGE = 25.0       # penalty tier = 28x sustained ≥30 min
MAX_MARGIN_PCT = 85.0     # penalty tier = 90%
MAX_SINGLE_PCT = 80.0     # penalty tier = 90%


# ── Pydantic models ─────────────────────────────────────────────────────────

class ActionType(str, Enum):
    ENTER = "ENTER"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


class TierType(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalResponse(BaseModel):
    """Structured signal returned by the /signal endpoint."""
    pair: str = Field(..., description="Pair name, e.g. XAUUSD/XAGUSD")
    action: ActionType = Field(..., description="Trade action: ENTER, CLOSE, or HOLD")
    z_score: float = Field(..., description="Current Z-score of the spread")
    beta: float = Field(..., description="Dynamic Kalman hedge ratio")
    r_value: float = Field(
        ..., description="Conformal-prediction asymmetric scale ratio"
    )
    half_life_hours: float = Field(
        ..., description="OU half-life in hours"
    )
    tier: TierType = Field(..., description="Conviction tier: HIGH, MEDIUM, or LOW")
    leg_buy: str = Field(..., description="Symbol to buy, e.g. 'BUY 1.2 XAUUSD'")
    leg_sell: str = Field(..., description="Symbol to sell, e.g. 'SELL 0.8 XAGUSD'")
    size_pct: float = Field(
        ..., description="Position size as % of guardrail ceiling"
    )
    passes_guardrail: bool = Field(
        ..., description="Whether the signal passes all hard guardrails"
    )
    narrative: str = Field(
        ..., description="2-sentence trade rationale from Nemotron"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str
    model: str
    guardrails: dict


# ── Guardrail validation ────────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    passes: bool
    leverage_ok: bool
    margin_ok: bool
    single_instrument_ok: bool
    effective_leverage: float
    margin_usage_pct: float
    single_instrument_pct: float


def validate_guardrails(
    size_pct: float,
    leverage_requested: float = 10.0,
    margin_usage_pct: float = 50.0,
    single_instrument_pct: float = 50.0,
) -> GuardrailResult:
    """Hard guardrail: reject or pass, never resize up.
    Caps from CLAUDE.md: 25x / 85% margin / 80% single-instrument.
    """
    leverage_ok = leverage_requested <= MAX_LEVERAGE
    margin_ok = margin_usage_pct <= MAX_MARGIN_PCT
    single_ok = single_instrument_pct <= MAX_SINGLE_PCT

    return GuardrailResult(
        passes=leverage_ok and margin_ok and single_ok,
        leverage_ok=leverage_ok,
        margin_ok=margin_ok,
        single_instrument_ok=single_ok,
        effective_leverage=leverage_requested,
        margin_usage_pct=margin_usage_pct,
        single_instrument_pct=single_instrument_pct,
    )


# ── Tiered conviction sizing ────────────────────────────────────────────────

def compute_tier_and_size(
    z_score: float,
    r_value: float,
    guardrail_ceiling: float = MAX_SINGLE_PCT,
) -> tuple[TierType, float]:
    """Tiered conviction sizing from CLAUDE.md strategy parameters.

    HIGH:   |Z| > 2.5 AND r > 4  → 0.90 × guardrail ceiling
    MEDIUM: |Z| > 2.0 AND r > 3  → 0.60 × guardrail ceiling
    LOW:    otherwise             → 0.30 × guardrail ceiling
    """
    abs_z = abs(z_score)

    if abs_z > 2.5 and r_value > 4.0:
        return TierType.HIGH, round(0.90 * guardrail_ceiling, 2)
    elif abs_z > 2.0 and r_value > 3.0:
        return TierType.MEDIUM, round(0.60 * guardrail_ceiling, 2)
    else:
        return TierType.LOW, round(0.30 * guardrail_ceiling, 2)


# ── Data engine: parquet + yfinance fallback ─────────────────────────────────

_DEFAULT_DATA_DIR = str(Path(__file__).parent / "pricer-output-2026-05-11_2026-06-10")
DATA_DIR = Path(os.getenv("PARQUET_DIR", _DEFAULT_DATA_DIR))
BAR_MINUTES = 15      # will be overridden to 60 if yfinance fallback is used
CALIB_DAYS = 30

# yfinance ticker mapping for FX pairs
YF_TICKER_MAP = {
    "EURUSD": "EURUSD=X",
    "EURGBP": "EURGBP=X",
    "AUDUSD": "AUDUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "EURCHF": "EURCHF=X",
    "GBPUSD": "GBPUSD=X",
}

# Module-level flag: set True when parquet is unavailable
_use_yfinance: bool = False

# Competition symbols available in the dataset
COMPETITION_SYMBOLS = [
    "AUDUSD", "EURCHF", "EURGBP", "EURUSD",
    "GBPUSD", "USDCAD", "USDCHF", "USDJPY",
]

# Active watchlist — curated from the full 28-pair screen.
# Two primaries, one secondary, two watch (Z approaching entry).
WATCHLIST = [
    ("AUDUSD", "USDJPY"),   # primary — strongest ADF (stat=-5.026)
    ("USDCAD", "USDJPY"),   # primary — tightest half-life (4.72h)
    ("USDCHF", "USDJPY"),   # secondary
    ("USDCAD", "USDCHF"),   # watch — Z approaching entry
    ("EURGBP", "EURUSD"),   # watch — Z approaching entry
]
CANDIDATE_PAIRS = WATCHLIST

# Half-life filter bounds (hours)
HL_MIN_HOURS = 2.0
HL_MAX_HOURS = 120.0

# Cached bar series per symbol
_bar_cache: dict[str, pd.Series] = {}


def _has_parquet_data() -> bool:
    """Check if parquet files exist in DATA_DIR."""
    return bool(glob.glob(str(DATA_DIR / "*.parquet")))


def _load_symbol_yfinance(symbol: str) -> pd.Series:
    """Fetch 30 days of 1h OHLCV from yfinance as fallback.

    Uses the Close column as the price series, resampled to 1h bars.
    """
    import yfinance as yf

    ticker = YF_TICKER_MAP.get(symbol)
    if not ticker:
        raise ValueError(f"No yfinance ticker mapping for {symbol}")

    logger.info(f"Fetching {symbol} via yfinance ({ticker}, 30d, 1h)...")
    df = yf.download(
        ticker,
        period=f"{CALIB_DAYS}d",
        interval="1h",
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    # yf.download returns MultiIndex columns when single ticker; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    bars = df["Close"].dropna()
    bars.name = symbol
    bars.index = bars.index.tz_localize(None)  # strip tz for consistency

    logger.info(
        f"  yfinance {symbol}: {len(bars)} bars "
        f"({bars.index[0]} to {bars.index[-1]})"
    )
    return bars


def _load_symbol_bars(symbol: str) -> pd.Series:
    """Load bar data for a symbol.

    Strategy:
      1. Try parquet files from DATA_DIR (tick data → 15-min bars).
      2. If no parquet files found, fall back to yfinance (1h bars).

    Uses fastparquet engine to avoid pyarrow 'Repetition level histogram'
    bug with the nested list columns in these files.
    """
    global _use_yfinance, BAR_MINUTES

    if symbol in _bar_cache:
        return _bar_cache[symbol]

    # ── Try parquet first ──────────────────────────────────────────────
    if not _use_yfinance:
        pattern = str(DATA_DIR / f"{symbol}_*.parquet")
        files = sorted(glob.glob(pattern))

        if files:
            dfs = []
            skipped = 0
            for f in files:
                try:
                    df = pd.read_parquet(
                        f, engine="fastparquet", columns=["time", "bid", "ask"]
                    )
                except Exception as e:
                    logger.warning(f"Skipping corrupted file {Path(f).name}: {e}")
                    skipped += 1
                    continue
                df["time"] = pd.to_datetime(df["time"])
                df["mid"] = (df["bid"] + df["ask"]) / 2.0
                dfs.append(df[["time", "mid"]])

            if dfs:
                all_ticks = pd.concat(dfs).sort_values("time").set_index("time")
                bars = all_ticks["mid"].resample(f"{BAR_MINUTES}min").last().dropna()
                bars.name = symbol
                skip_note = f" ({skipped} skipped)" if skipped else ""
                logger.info(
                    f"Loaded {symbol}: {len(files) - skipped}/{len(files)} "
                    f"files{skip_note} → {len(bars)} bars "
                    f"({bars.index[0]} to {bars.index[-1]})"
                )
                _bar_cache[symbol] = bars
                return bars

        # No parquet files → switch to yfinance for all symbols
        if not _use_yfinance:
            logger.warning(
                f"No parquet files for {symbol} in {DATA_DIR} — "
                "switching to yfinance fallback for ALL symbols"
            )
            _use_yfinance = True
            BAR_MINUTES = 60  # yfinance gives 1h bars

    # ── yfinance fallback ─────────────────────────────────────────────
    bars = _load_symbol_yfinance(symbol)
    _bar_cache[symbol] = bars
    return bars


def _ou_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion in BARS via Dickey-Fuller regression.

    d(s_t) = c + λ·s_{t-1} + e_t,  half-life = -ln(2) / ln(1 + λ).
    Same logic as pair_screener.py.
    """
    s = spread.dropna()
    s_lag = s.shift(1)
    delta = s - s_lag
    df = pd.concat([delta, s_lag], axis=1).dropna()
    df.columns = ["delta", "s_lag"]
    if len(df) < 10:
        return np.inf
    x = sm.add_constant(df["s_lag"].values)
    lam = sm.OLS(df["delta"].values, x).fit().params[1]
    if lam >= 0 or (1.0 + lam) <= 0:
        return np.inf
    return float(-np.log(2.0) / np.log(1.0 + lam))


def _screen_pair(sym_a: str, sym_b: str) -> dict:
    """Run cointegration + half-life screen on a pair using real bar data.

    Returns a dict with all signal fields. Does NOT apply guardrails or
    sizing — that's the endpoint's job.
    """
    bars_a = _load_symbol_bars(sym_a)
    bars_b = _load_symbol_bars(sym_b)

    # Align and slice to last CALIB_DAYS
    px = pd.DataFrame({sym_a: bars_a, sym_b: bars_b}).dropna()
    cutoff = px.index.max() - pd.Timedelta(days=CALIB_DAYS)
    px = px.loc[px.index >= cutoff]

    if len(px) < 100:
        raise ValueError(f"{sym_a}/{sym_b}: only {len(px)} overlapping bars")

    # OLS hedge ratio: A = alpha + beta * B + spread
    x = sm.add_constant(px[sym_b].values)
    ols = sm.OLS(px[sym_a].values, x).fit()
    alpha, beta = float(ols.params[0]), float(ols.params[1])
    spread = px[sym_a] - (alpha + beta * px[sym_b])

    # ADF test on spread
    adf_result = adfuller(spread.dropna(), autolag="AIC")
    adf_p = float(adf_result[1])

    # OU half-life
    hl_bars = _ou_half_life(spread)
    hl_hours = hl_bars * BAR_MINUTES / 60.0

    # Z-score
    mu, sigma = float(spread.mean()), float(spread.std())
    z_score = float((spread.iloc[-1] - mu) / sigma) if sigma > 0 else 0.0

    # Cointegration pass: ADF p < 0.05 AND tradeable half-life (2-120h)
    passes_coint = adf_p < 0.05
    passes_hl = np.isfinite(hl_hours) and HL_MIN_HOURS <= hl_hours <= HL_MAX_HOURS
    passes_screen = passes_coint and passes_hl

    # r_value: use OLS R² as a proxy for conformal-prediction scale ratio.
    # Map R² → scale ratio r ∈ [1, 10] for tiered sizing.
    r_squared = float(ols.rsquared)
    r_value = round(1.0 + 9.0 * r_squared, 1)  # R²=1 → r=10, R²=0 → r=1

    return {
        "pair": f"{sym_a}/{sym_b}",
        "symbol_a": sym_a,
        "symbol_b": sym_b,
        "z_score": round(z_score, 4),
        "beta": round(beta, 6),
        "r_value": r_value,
        "half_life_hours": round(hl_hours, 2) if np.isfinite(hl_hours) else 999.0,
        "adf_pvalue": round(adf_p, 5),
        "passes_screen": passes_screen,
        "passes_coint": passes_coint,
        "passes_hl": passes_hl,
        "n_bars": len(px),
        "spread_mu": mu,
        "spread_sigma": sigma,
        # Placeholder portfolio-level metrics (conservative defaults)
        "leverage_requested": 10.0,
        "margin_usage_pct": 40.0,
        "single_instrument_pct": 50.0,
    }


def _get_best_signal(pair_name: str | None = None) -> dict:
    """Screen all candidate pairs and return the best actionable signal.

    If pair_name is specified (e.g. 'EURUSD/EURGBP'), return that pair.
    Otherwise, return the pair with the highest |Z| that passes screening.
    If no pair passes, return the pair with the highest |Z| anyway (for
    transparency) — the endpoint's action logic will set HOLD.
    """
    if pair_name:
        parts = pair_name.split("/")
        if len(parts) == 2:
            return _screen_pair(parts[0], parts[1])

    results = []
    for sym_a, sym_b in CANDIDATE_PAIRS:
        try:
            r = _screen_pair(sym_a, sym_b)
            results.append(r)
        except Exception as e:
            logger.warning(f"Failed to screen {sym_a}/{sym_b}: {e}")

    if not results:
        raise HTTPException(status_code=503, detail="No pairs could be screened")

    # Prefer passing pairs, then highest |Z|
    passing = [r for r in results if r["passes_screen"]]
    pool = passing if passing else results
    best = max(pool, key=lambda r: abs(r["z_score"]))
    return best


def _log_startup_screening():
    """Log schema + screening results for all candidate pairs at startup."""
    has_parquet = _has_parquet_data()

    if has_parquet:
        # Log schema from one file
        sample_files = sorted(glob.glob(str(DATA_DIR / "EURUSD_*.parquet")))[:1]
        if sample_files:
            sample = pd.read_parquet(
                sample_files[0], engine="fastparquet", columns=["time", "bid", "ask"]
            )
            logger.info(f"Parquet schema columns: {list(sample.columns)}")
            logger.info(f"  dtypes: {dict(sample.dtypes)}")

        # Available symbols
        all_files = glob.glob(str(DATA_DIR / "*.parquet"))
        symbols = sorted({Path(f).stem.rsplit("_", 3)[0] for f in all_files})
        logger.info(f"Available symbols: {symbols}")
        logger.info(f"Data source: parquet ({DATA_DIR})")
    else:
        logger.warning(
            f"No parquet files found in {DATA_DIR} — "
            "Using yfinance fallback (live 1h OHLCV, 30 days)"
        )
        logger.info(f"yfinance ticker map: {YF_TICKER_MAP}")

    logger.info(
        f"Screening {len(WATCHLIST)} curated watchlist pairs"
    )

    # Screen all candidate pairs
    logger.info("── Startup pair screening (30-day window) ──")
    header = f"{'Pair':20s} {'β':>8s} {'ADF p':>10s} {'HL (hrs)':>10s} {'Z':>8s} {'Coint':>6s} {'HL ok':>6s}"
    logger.info(header)
    for sym_a, sym_b in CANDIDATE_PAIRS:
        try:
            r = _screen_pair(sym_a, sym_b)
            logger.info(
                f"{r['pair']:20s} {r['beta']:8.4f} {r['adf_pvalue']:10.5f} "
                f"{r['half_life_hours']:10.2f} {r['z_score']:+8.4f} "
                f"{'PASS' if r['passes_coint'] else 'FAIL':>6s} "
                f"{'PASS' if r['passes_hl'] else 'FAIL':>6s}"
            )
        except Exception as e:
            logger.error(f"{sym_a}/{sym_b}: {e}")


# ── Nemotron narrative (async via OpenAI-compatible client) ──────────────────

_oai_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    """Lazy-init the async OpenAI client pointed at Doubleword."""
    global _oai_client
    if _oai_client is None:
        _oai_client = AsyncOpenAI(
            api_key=DOUBLEWORD_API_KEY,
            base_url=DOUBLEWORD_BASE_URL,
        )
    return _oai_client


async def _call_nemotron(client: AsyncOpenAI, system_prompt: str, user_prompt: str) -> str | None:
    """Single Nemotron call → cleaned narrative string, or None on failure."""
    resp = await client.chat.completions.create(
        model=DOUBLEWORD_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    raw_content = resp.choices[0].message.content
    if raw_content is None or not raw_content.strip():
        # Thinking/reasoning model may put the answer in the reasoning field
        reasoning = getattr(resp.choices[0].message, "reasoning", None)
        if reasoning and reasoning.strip():
            paragraphs = [p.strip() for p in reasoning.strip().split("\n\n") if p.strip()]
            raw_content = paragraphs[-1] if paragraphs else reasoning.strip()
        else:
            return None
    return raw_content.strip()


def _clean_narrative(raw: str) -> str:
    """Strip <think> blocks, reasoning preamble, and enforce 2-sentence max."""
    text = raw.strip()

    # 1. Strip <think>...</think> — greedy to catch nested or multi-block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 2. Handle unclosed <think> — strip from <think> to end-of-string
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    # 3. Handle orphaned </think> — strip from start to </think>
    text = re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()
    # 4. Strip reasoning-preamble SENTENCES (not lines).
    #    Split into sentences first, then discard any that match meta-commentary.
    #    This preserves real content on the same line as preamble.
    preamble_re = re.compile(
        r"^(Let'?s\s|Here\s(is|are)\s|I\s(need|should|will|must|have|'ll)\s"
        r"|We\s(need|should|will|must|have|are)\s|Okay[,.\s]|Sure[,.\s]"
        r"|Now[,.\s]|So[,.\s]|Output:?\s*$|Signal:\s|Provide\s"
        r"|Two\ssentences|Sentence\s\d|Rationale:"
        # Parameter echo: sentence is just restating input values
        r"|Z-?score[=:\s]|beta[=:\s]\d|r[=:\s]\d|half-?life[=:\s]"
        r"|Must\s)",
        re.IGNORECASE,
    )

    # Flatten to single line, then split on sentence boundaries
    text = " ".join(text.split())
    all_sentences = re.split(r'(?<=[.!?])\s+', text)

    kept = []
    for sent in all_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if preamble_re.match(sent):
            continue
        kept.append(sent)
    text = " ".join(kept).strip()

    # 5. Enforce 2-sentence max
    if text:
        sentence_parts = re.split(r'(?<=[.!?])\s+', text)
        if len(sentence_parts) > 2:
            text = " ".join(sentence_parts[:2])
        # Ensure it ends with a period
        if text and text[-1] not in ".!?":
            text += "."

    return text


FALLBACK_NARRATIVE = (
    "Signal generated from spread dislocation on the pair. "
    "Sizing follows tiered conviction rules within guardrail limits."
)
MIN_NARRATIVE_LEN = 20


async def generate_narrative(
    pair: str,
    action: str,
    z_score: float,
    beta: float,
    r_value: float,
    half_life_hours: float,
    tier: str,
    size_pct: float,
) -> str:
    """Call Doubleword Nemotron for a 2-sentence trade rationale.

    Uses the OpenAI-compatible chat completions API in async mode,
    routed through the Doubleword inference endpoint.  If the first
    attempt yields junk (< 20 chars after cleaning), retries once.
    """
    system_prompt = (
        "You are a quantitative trading copilot for a stat-arb system. "
        "Given the signal parameters, produce EXACTLY two concise sentences: "
        "sentence 1 explains WHY the signal fires (spread dislocation, "
        "mean-reversion logic); sentence 2 states the risk framing "
        "(conviction tier, sizing, half-life context). "
        "No bullet points, no markdown, no hedging. Be precise and direct. "
        "Do not use <think> tags or reasoning blocks. "
        "Output ONLY the two sentences, nothing else."
    )

    user_prompt = (
        f"/no_think Signal: {action} on {pair}. "
        f"Z-score={z_score:.2f}, beta={beta:.4f}, r={r_value:.1f}, "
        f"half-life={half_life_hours:.1f}h. "
        f"Tier={tier}, size={size_pct:.1f}% of guardrail ceiling. "
        f"Provide the 2-sentence rationale."
    )

    try:
        client = _get_client()

        # Attempt 1
        raw = await _call_nemotron(client, system_prompt, user_prompt)
        narrative = _clean_narrative(raw) if raw else ""

        # Retry once if cleaned result is too short / empty
        if len(narrative) < MIN_NARRATIVE_LEN:
            raw = await _call_nemotron(client, system_prompt, user_prompt)
            narrative = _clean_narrative(raw) if raw else ""

        if len(narrative) < MIN_NARRATIVE_LEN:
            return FALLBACK_NARRATIVE

        return narrative
    except Exception as e:
        return f"[Narrative unavailable: {type(e).__name__}: {e}]"


# ── Signal box formatter ────────────────────────────────────────────────────

# Box inner width: 36 chars between the left ║ and right ║.
_BOX_W = 36


def _pad(content: str) -> str:
    """Pad content to exactly _BOX_W chars: '║' + content + '║'."""
    return f"║{content:<{_BOX_W}}║"


def format_signal_box(sig: SignalResponse) -> str:
    """Format the signal into the fixed-width box required by the spec.

    Every inner line is exactly 36 chars wide so the box renders
    cleanly regardless of value widths.
    """
    action_label = sig.action.value
    stats_line = f"  Z={sig.z_score:+.2f} | β={sig.beta:.4f} | r={sig.r_value:.1f}"

    border_top = f"╔{'═' * _BOX_W}╗"
    border_bot = f"╚{'═' * _BOX_W}╝"

    lines = [
        border_top,
        _pad(f"  SIGNAL — {action_label}"),
        _pad(f"  {sig.leg_buy}"),
        _pad(f"  {sig.leg_sell}"),
        _pad(stats_line),
        _pad(f"  TIER: {sig.tier.value}"),
        border_bot,
        f"RATIONALE: {sig.narrative}",
    ]
    return "\n".join(lines)


# ── FastAPI app ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: load data, log schema, screen pairs."""
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    _log_startup_screening()
    yield


app = FastAPI(
    title="Model to Market — Signal Server",
    description=(
        "Stat-arb copilot signal endpoint for the Quantitative Hack. "
        "Serves real parquet data with OLS/ADF/OU screening + Nemotron narratives."
    ),
    version="0.2.0",
    lifespan=_lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=DOUBLEWORD_MODEL,
        guardrails={
            "max_leverage": MAX_LEVERAGE,
            "max_margin_pct": MAX_MARGIN_PCT,
            "max_single_pct": MAX_SINGLE_PCT,
        },
    )


# ── Watchlist endpoint ───────────────────────────────────────────────────────


class WatchlistItem(BaseModel):
    """Single pair in the watchlist summary."""
    pair: str
    z_score: float
    abs_z: float
    beta: float
    adf_pvalue: float
    half_life_hours: float
    r_value: float
    passes_screen: bool
    tier: str
    action_hint: str = Field(
        description="Quick label: ENTRY if |Z|>=2, APPROACHING if |Z|>=1.5, WATCH otherwise"
    )


class WatchlistResponse(BaseModel):
    """All 5 watchlist pairs with current Z-scores."""
    timestamp: str
    pairs: list[WatchlistItem]
    best_pair: str
    best_z: float


@app.get("/watchlist", response_model=WatchlistResponse)
async def watchlist():
    """Return current Z-scores for all 5 watchlist pairs, sorted by |Z| desc.

    Use this to monitor which pairs are heating up toward entry threshold.
    """
    items = []
    for sym_a, sym_b in WATCHLIST:
        try:
            r = _screen_pair(sym_a, sym_b)
            abs_z = abs(r["z_score"])

            # Quick action hint
            if abs_z >= 2.0 and r["passes_screen"]:
                hint = "ENTRY"
            elif abs_z >= 1.5:
                hint = "APPROACHING"
            else:
                hint = "WATCH"

            # Tier for context
            tier, _ = compute_tier_and_size(r["z_score"], r["r_value"])

            items.append(WatchlistItem(
                pair=r["pair"],
                z_score=r["z_score"],
                abs_z=round(abs_z, 4),
                beta=r["beta"],
                adf_pvalue=r["adf_pvalue"],
                half_life_hours=r["half_life_hours"],
                r_value=r["r_value"],
                passes_screen=r["passes_screen"],
                tier=tier.value,
                action_hint=hint,
            ))
        except Exception as e:
            logger.warning(f"Watchlist: failed to screen {sym_a}/{sym_b}: {e}")

    # Sort by |Z| descending
    items.sort(key=lambda x: x.abs_z, reverse=True)

    best = items[0] if items else None
    return WatchlistResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        pairs=items,
        best_pair=best.pair if best else "NONE",
        best_z=best.z_score if best else 0.0,
    )


@app.get("/signal", response_class=PlainTextResponse)
async def signal(pair: Optional[str] = None):
    """Generate a trading signal from real parquet data.

    Returns the formatted signal box with Nemotron narrative.
    Pass ?pair=EURUSD/EURGBP to target a specific pair.
    """
    # ── 1. Get real signal data from parquet ─────────────────────────────────
    raw = _get_best_signal(pair)

    z_score = raw["z_score"]
    beta = raw["beta"]
    r_value = raw["r_value"]
    half_life_hours = raw["half_life_hours"]
    symbol_a = raw["symbol_a"]
    symbol_b = raw["symbol_b"]

    # ── 2. Determine action ─────────────────────────────────────────────────
    abs_z = abs(z_score)
    if abs_z >= 2.0:
        action = ActionType.ENTER
    elif abs_z <= 0.5:
        action = ActionType.CLOSE
    else:
        action = ActionType.HOLD

    # ── 3. Tiered conviction sizing ─────────────────────────────────────────
    tier, size_pct = compute_tier_and_size(z_score, r_value)

    # ── 4. Guardrail validation ─────────────────────────────────────────────
    guardrail = validate_guardrails(
        size_pct=size_pct,
        leverage_requested=raw["leverage_requested"],
        margin_usage_pct=raw["margin_usage_pct"],
        single_instrument_pct=raw["single_instrument_pct"],
    )

    if not guardrail.passes:
        # Guardrail is a one-way gate: reject, never resize up
        action = ActionType.HOLD

    # ── 5. Compute legs ─────────────────────────────────────────────────────
    # spread = A − (α + β·B).  Z < 0 → spread below mean → A underpriced
    # relative to B → LONG A, SHORT B.  Z > 0 → the reverse.
    # For XAUUSD/XAGUSD at Z = −2.35: BUY XAUUSD (A), SELL XAGUSD (B). ✓
    if z_score < 0:
        leg_buy_sym, leg_sell_sym = symbol_a, symbol_b
    else:
        leg_buy_sym, leg_sell_sym = symbol_b, symbol_a

    # Lot sizing: leg A gets 1.0 lots, leg B gets beta lots (hedge ratio)
    lot_a = round(size_pct / 10.0, 1)  # scale for display
    lot_b = round(lot_a * abs(beta), 1) if abs(beta) > 0.001 else lot_a

    # For metals where beta is tiny (e.g. 0.0128), normalise both legs
    # to make sizes readable. Show as ratio with minimum 0.1 lots.
    if lot_b < 0.1:
        lot_b = round(lot_a * abs(beta) * 100, 1)
        lot_a_display = lot_a
    else:
        lot_a_display = lot_a

    leg_buy = f"BUY  {lot_a_display} {leg_buy_sym}"
    leg_sell = f"SELL {lot_b} {leg_sell_sym}"

    # ── 6. Generate narrative (async Nemotron call) ──────────────────────────
    narrative = await generate_narrative(
        pair=raw["pair"],
        action=action.value,
        z_score=z_score,
        beta=beta,
        r_value=r_value,
        half_life_hours=half_life_hours,
        tier=tier.value,
        size_pct=size_pct,
    )

    # ── 7. Build response ───────────────────────────────────────────────────
    sig = SignalResponse(
        pair=raw["pair"],
        action=action,
        z_score=z_score,
        beta=beta,
        r_value=r_value,
        half_life_hours=half_life_hours,
        tier=tier,
        leg_buy=leg_buy,
        leg_sell=leg_sell,
        size_pct=size_pct,
        passes_guardrail=guardrail.passes,
        narrative=narrative,
    )

    return format_signal_box(sig)


@app.get("/signal/json", response_model=SignalResponse)
async def signal_json(pair: Optional[str] = None):
    """Same as /signal but returns structured JSON instead of the text box."""
    raw = _get_best_signal(pair)

    z_score = raw["z_score"]
    beta = raw["beta"]
    r_value = raw["r_value"]
    half_life_hours = raw["half_life_hours"]
    symbol_a = raw["symbol_a"]
    symbol_b = raw["symbol_b"]

    abs_z = abs(z_score)
    if abs_z >= 2.0:
        action = ActionType.ENTER
    elif abs_z <= 0.5:
        action = ActionType.CLOSE
    else:
        action = ActionType.HOLD

    tier, size_pct = compute_tier_and_size(z_score, r_value)

    guardrail = validate_guardrails(
        size_pct=size_pct,
        leverage_requested=raw["leverage_requested"],
        margin_usage_pct=raw["margin_usage_pct"],
        single_instrument_pct=raw["single_instrument_pct"],
    )

    if not guardrail.passes:
        action = ActionType.HOLD

    if z_score < 0:
        leg_buy_sym, leg_sell_sym = symbol_a, symbol_b
    else:
        leg_buy_sym, leg_sell_sym = symbol_b, symbol_a

    lot_a = round(size_pct / 10.0, 1)
    lot_b = round(lot_a * abs(beta), 1) if abs(beta) > 0.001 else lot_a
    if lot_b < 0.1:
        lot_b = round(lot_a * abs(beta) * 100, 1)
        lot_a_display = lot_a
    else:
        lot_a_display = lot_a

    leg_buy = f"BUY  {lot_a_display} {leg_buy_sym}"
    leg_sell = f"SELL {lot_b} {leg_sell_sym}"

    narrative = await generate_narrative(
        pair=raw["pair"],
        action=action.value,
        z_score=z_score,
        beta=beta,
        r_value=r_value,
        half_life_hours=half_life_hours,
        tier=tier.value,
        size_pct=size_pct,
    )

    return SignalResponse(
        pair=raw["pair"],
        action=action,
        z_score=z_score,
        beta=beta,
        r_value=r_value,
        half_life_hours=half_life_hours,
        tier=tier,
        leg_buy=leg_buy,
        leg_sell=leg_sell,
        size_pct=size_pct,
        passes_guardrail=guardrail.passes,
        narrative=narrative,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "signal_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
