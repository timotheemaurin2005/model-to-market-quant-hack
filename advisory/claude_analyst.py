#!/usr/bin/env python3
"""
claude_analyst.py
=================

Advisory Brain — Dual-Brain FX Statistical-Arbitrage System
------------------------------------------------------------

ARCHITECTURE
    This system enforces a strict separation between decision-making and narration:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  EXECUTION BRAIN  (remote VPS · Python → MetaTrader 5)             │
    │  • Runs a fully deterministic stat-arb loop                        │
    │  • Manages cointegrated FX pair positions                          │
    │  • Writes state.json after every position change                   │
    │  • NEVER calls any language model                                  │
    └───────────────────────────────┬──────────────────────────────────────┘
                                    │  state.json  (read-only to this script)
    ┌───────────────────────────────▼──────────────────────────────────────┐
    │  ADVISORY BRAIN  (local Mac · this script)                         │
    │  • Reads state.json                                                │
    │  • Computes all analytics in plain Python — zero LLM involvement   │
    │  • Calls Claude API for narrative commentary only                  │
    │  • Writes a dated Markdown report                                  │
    │  • Has ZERO influence on the execution path                        │
    └──────────────────────────────────────────────────────────────────────┘

    The LLM is a pure narrator.  It receives pre-computed numbers and structured
    context prose; it performs no arithmetic, makes no trade decisions, and cannot
    interact with the execution layer in any way.

USAGE
    export ANTHROPIC_API_KEY="sk-ant-..."
    python claude_analyst.py [--state path/to/state.json]

DEPENDENCIES
    pip install anthropic

OUTPUT
    • stdout — deterministic tables + Claude advisory sections
    • file   — report_YYYY-MM-DD.md (same directory as this script)
"""

__version__ = "1.0.0"
__author__ = "Advisory Brain"

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Graceful import guard — give the user a clear fix instead of a traceback
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:
    print(
        "ERROR: The 'anthropic' package is not installed.\n"
        "       Fix: pip install anthropic",
        file=sys.stderr,
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MODEL_ID: str = "claude-sonnet-4-6"
MAX_TOKENS: int = 1024
DEFAULT_STATE_PATH: str = "state.json"

# Required fields validated inside every pair block of state.json
_REQUIRED_PAIR_FIELDS: tuple[str, ...] = (
    "alpha", "beta", "mu", "sigma",
    "entry_z", "entry_time", "half_life_min", "symbols",
)


# ═══════════════════════════════════════════════════════════════════════════
# Formatting utilities
# ═══════════════════════════════════════════════════════════════════════════

def _hr(char: str = "─", width: int = 72) -> str:
    """Return a horizontal rule of the requested character and width."""
    return char * width


def _banner(title: str, char: str = "═", width: int = 72) -> str:
    """Return a visually distinct section banner."""
    bar = _hr(char, width)
    return f"\n{bar}\n  {title}\n{bar}"


def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Compute column widths wide enough to fit every header and cell value."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    return widths


def _render_table(headers: list[str], rows: list[list[str]], pad: int = 2) -> str:
    """
    Render a fixed-width plain-text table.

    Parameters
    ----------
    headers : column header strings
    rows    : list of string rows (must have same length as headers)
    pad     : number of spaces between columns
    """
    widths = _col_widths(headers, rows)
    sep = " " * pad

    def _fmt_row(cells: list[str]) -> str:
        return sep.join(c.ljust(w) for c, w in zip(cells, widths))

    divider = sep.join("─" * w for w in widths)
    output_lines = [_fmt_row(headers), divider]
    for row in rows:
        output_lines.append(_fmt_row(row))
    return "\n".join(output_lines)


def _indent(text: str, spaces: int = 2) -> str:
    """Indent every line of *text* by *spaces* space characters."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic analytics layer
# (NO LLM — every number here must be independently verifiable)
# ═══════════════════════════════════════════════════════════════════════════

def parse_currencies(symbols: list[str]) -> set[str]:
    """
    Split each 6-character FX symbol into its two 3-character currency codes.

    Examples
    --------
    >>> sorted(parse_currencies(["GBPUSD", "USDJPY"]))
    ['GBP', 'JPY', 'USD']
    """
    currencies: set[str] = set()
    for sym in symbols:
        sym = sym.strip().upper()
        if len(sym) == 6:
            currencies.add(sym[:3])
            currencies.add(sym[3:])
        else:
            # Non-standard length — store whole and flag for review
            currencies.add(sym)
    return currencies


def validate_pair_data(pair_tag: str, raw: dict[str, Any]) -> None:
    """
    Raise ValueError if any required field is absent from a pair block.

    Keeps the deterministic layer honest about its inputs.
    """
    missing = [f for f in _REQUIRED_PAIR_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            f"Pair '{pair_tag}' is missing required fields: {missing}"
        )
    if not isinstance(raw.get("symbols"), list) or len(raw["symbols"]) != 2:
        raise ValueError(
            f"Pair '{pair_tag}': 'symbols' must be a 2-element list."
        )


def compute_pair_metrics(
    pair_tag: str,
    raw: dict[str, Any],
    now_utc: datetime,
) -> dict[str, Any]:
    """
    Derive all per-pair analytics from one raw state block.

    Parameters
    ----------
    pair_tag : the dict key from state.json, e.g. "GBPUSD/USDJPY"
    raw      : the value block for that pair
    now_utc  : current UTC timestamp (timezone-aware)

    Returns
    -------
    Flat dict containing all original fields plus:
        hold_min, reversion_progress, currencies, past_half_life
    """
    validate_pair_data(pair_tag, raw)

    # Parse entry time — defend against naive timestamps
    entry_time: datetime = datetime.fromisoformat(raw["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    half_life_min: float = float(raw["half_life_min"])
    hold_min: float = (now_utc - entry_time).total_seconds() / 60.0
    reversion_progress: float = (
        hold_min / half_life_min if half_life_min > 0.0 else float("inf")
    )

    return {
        # ── Original fields (preserved exactly as stored) ─────────────
        "pair_tag":       pair_tag,
        "symbols":        raw["symbols"],
        "alpha":          float(raw["alpha"]),
        "beta":           float(raw["beta"]),
        "mu":             float(raw["mu"]),
        "sigma":          float(raw["sigma"]),
        "entry_z":        float(raw["entry_z"]),
        "entry_time_str": raw["entry_time"],
        "half_life_min":  half_life_min,
        # ── Derived ───────────────────────────────────────────────────
        "hold_min":           hold_min,
        "reversion_progress": reversion_progress,
        "currencies":         parse_currencies(raw["symbols"]),
        "past_half_life":     reversion_progress >= 1.0,
    }


def compute_book_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-pair metrics into book-level statistics.

    Parameters
    ----------
    metrics : list of dicts returned by compute_pair_metrics()

    Returns
    -------
    dict with book-level aggregates.
    """
    n: int = len(metrics)
    if n == 0:
        return {
            "n_pairs": 0,
            "currencies": [],
            "avg_hold_min": 0.0,
            "avg_entry_z": 0.0,
            "avg_rev_progress": 0.0,
            "n_past_half_life": 0,
        }

    all_currencies: set[str] = set()
    for pm in metrics:
        all_currencies.update(pm["currencies"])

    return {
        "n_pairs":           n,
        "currencies":        sorted(all_currencies),
        "avg_hold_min":      sum(pm["hold_min"]           for pm in metrics) / n,
        "avg_entry_z":       sum(pm["entry_z"]            for pm in metrics) / n,
        "avg_rev_progress":  sum(pm["reversion_progress"] for pm in metrics) / n,
        "n_past_half_life":  sum(1 for pm in metrics if pm["past_half_life"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic report builder
# ═══════════════════════════════════════════════════════════════════════════

def build_deterministic_section(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    now_utc: datetime,
) -> str:
    """
    Render the fully deterministic portion of the report as a string.

    No external calls — purely derived from state.json arithmetic.
    """
    lines: list[str] = []

    # ── Section header ────────────────────────────────────────────────────
    lines.append(_hr("═"))
    lines.append("  DETERMINISTIC PORTFOLIO SNAPSHOT")
    lines.append(f"  Generated : {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Source    : state.json  ({summary['n_pairs']} open position(s))")
    lines.append(_hr("═"))

    if summary["n_pairs"] == 0:
        lines.append("\n  [INFO] No open positions found in state.json.")
        return "\n".join(lines)

    lines.append("")

    # ── Per-pair summary table ────────────────────────────────────────────
    lines.append("  PER-PAIR OVERVIEW")
    lines.append("  " + _hr())

    table_headers = [
        "Pair",
        "Entry Z",
        "α (alpha)",
        "β (beta)",
        "μ (mu)",
        "σ (sigma)",
        "Half-life(min)",
        "Hold(min)",
        "Rev.Progress",
        "Status",
    ]
    table_rows: list[list[str]] = []
    for pm in metrics:
        status = "⚠ PAST HL" if pm["past_half_life"] else "✓ in window"
        table_rows.append([
            pm["pair_tag"],
            f"{pm['entry_z']:+.4f}",
            f"{pm['alpha']:.6f}",
            f"{pm['beta']:.6f}",
            f"{pm['mu']:.4e}",
            f"{pm['sigma']:.6f}",
            f"{pm['half_life_min']:.1f}",
            f"{pm['hold_min']:.1f}",
            f"{pm['reversion_progress'] * 100:.1f}%",
            status,
        ])

    table_str = _render_table(table_headers, table_rows)
    for tl in table_str.splitlines():
        lines.append("  " + tl)
    lines.append("")

    # ── Per-pair detail blocks ────────────────────────────────────────────
    lines.append("  PER-PAIR DETAIL")
    lines.append("  " + _hr())

    for pm in metrics:
        hold_hr   = pm["hold_min"]   / 60.0
        hl_hr     = pm["half_life_min"] / 60.0
        z_sign    = "spread above mean (short spread)" if pm["entry_z"] > 0 else "spread below mean (long spread)"
        rev_flag  = "  ← ⚠ PAST HALF-LIFE" if pm["past_half_life"] else ""

        lines += [
            f"  [{pm['pair_tag']}]",
            f"    Symbols          : {' + '.join(pm['symbols'])}",
            f"    Currencies       : {', '.join(sorted(pm['currencies']))}",
            f"    Entry Time (UTC) : {pm['entry_time_str']}",
            f"    Entry Z-score    : {pm['entry_z']:+.6f}  ({z_sign})",
            f"    α (intercept)    : {pm['alpha']:.8f}",
            f"    β (hedge ratio)  : {pm['beta']:.8f}",
            f"    μ (spread mean)  : {pm['mu']:.6e}",
            f"    σ (spread std)   : {pm['sigma']:.8f}",
            f"    Half-life        : {pm['half_life_min']:.2f} min  ({hl_hr:.2f} hr)",
            f"    Hold time        : {pm['hold_min']:.2f} min  ({hold_hr:.2f} hr)",
            f"    Rev. progress    : {pm['reversion_progress'] * 100:.1f}%{rev_flag}",
            "",
        ]

    # ── Book-level summary table ──────────────────────────────────────────
    lines.append("  BOOK-LEVEL SUMMARY")
    lines.append("  " + _hr())

    summary_items: list[tuple[str, str]] = [
        ("Open pairs",           str(summary["n_pairs"])),
        ("Currencies held",      "  ".join(summary["currencies"])),
        ("Avg hold time",        f"{summary['avg_hold_min']:.1f} min  ({summary['avg_hold_min'] / 60:.2f} hr)"),
        ("Avg entry |Z|",        f"{abs(summary['avg_entry_z']):.4f}"),
        ("Avg rev. progress",    f"{summary['avg_rev_progress'] * 100:.1f}%"),
        ("Pairs past half-life", f"{summary['n_past_half_life']} / {summary['n_pairs']}"),
    ]
    label_w = max(len(k) for k, _ in summary_items)
    for label, value in summary_items:
        lines.append(f"  {label.ljust(label_w)}  :  {value}")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# LLM context builders
# (Serialize pre-computed state into structured prose for the model)
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio_context(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """
    Produce a terse, unambiguous text block representing the portfolio state.

    The LLM reads this as input.  It is not expected to derive any numbers —
    all values here are final and pre-computed.
    """
    lines: list[str] = [
        "=== PORTFOLIO STATE (all values pre-computed — treat as authoritative) ===",
        "",
        f"Open pairs           : {summary['n_pairs']}",
        f"Avg hold time        : {summary['avg_hold_min']:.1f} min "
        f"({summary['avg_hold_min'] / 60:.2f} hr)",
        f"Avg entry Z-score    : {summary['avg_entry_z']:+.4f}",
        f"Avg rev. progress    : {summary['avg_rev_progress'] * 100:.1f}%",
        f"Pairs past half-life : {summary['n_past_half_life']} of {summary['n_pairs']}",
        "",
        "--- Per-Pair Breakdown ---",
    ]

    for pm in metrics:
        status = "PAST HALF-LIFE (hold time exceeds 1× half-life)" if pm["past_half_life"] \
            else "within expected reversion window"
        lines += [
            "",
            f"Pair: {pm['pair_tag']}  (legs: {' + '.join(pm['symbols'])})",
            f"  Entry Z-score    : {pm['entry_z']:+.4f}",
            f"  Half-life        : {pm['half_life_min']:.1f} min ({pm['half_life_min'] / 60:.2f} hr)",
            f"  Hold time        : {pm['hold_min']:.1f} min ({pm['hold_min'] / 60:.2f} hr)",
            f"  Rev. progress    : {pm['reversion_progress'] * 100:.1f}%  — {status}",
            f"  Spread std (σ)   : {pm['sigma']:.6f}",
            f"  Hedge ratio (β)  : {pm['beta']:.6f}",
        ]

    return "\n".join(lines)


def build_currency_context(
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> str:
    """
    Produce a structured description of the book's currency exposure for the
    tail-risk prompt.  Emphasises market-neutral nature and spread relationships.
    """
    pair_lines = "\n".join(
        f"  • {pm['pair_tag']}  (legs: {' and '.join(pm['symbols'])})"
        for pm in metrics
    )

    return dedent(f"""\
        === BOOK CURRENCY EXPOSURE ===

        Strategy type : Market-neutral statistical arbitrage (stat-arb)
        Directional   : NO net directional exposure to any single currency
        Risk model    : Cointegrated spread positions — each leg hedges the other

        Distinct currencies held : {', '.join(summary['currencies'])}
        Number of spread pairs   : {summary['n_pairs']}

        Active spread pairs:
        {pair_lines}

        Primary risk dimensions for this book (NOT directional):
          1. Cointegration / correlation breakdown between spread legs
          2. Macro regime changes that invalidate historical spread relationships
          3. Scheduled central-bank or data events that could spike spread volatility
          4. Liquidity or microstructure dislocations preventing orderly pair execution
          5. Simultaneous moves across correlated pairs (e.g. USD appears in multiple legs)
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Claude API calls
# ═══════════════════════════════════════════════════════════════════════════

def call_portfolio_analyst(
    client: anthropic.Anthropic,
    portfolio_context: str,
) -> str:
    """
    Call (a) — Portfolio read.

    Ask Claude to narrate the pre-computed portfolio state: whether positions
    are within or past their mean-reversion window and overall book health.
    The model performs NO arithmetic.
    """
    system_prompt = dedent("""\
        You are a quantitative analyst providing a morning brief for a market-neutral
        FX statistical-arbitrage desk.  You will receive a structured snapshot of the
        current portfolio with all metrics already computed.

        Your role is to narrate — not calculate.  Every number you receive is final.

        Focus on:
          • Whether each position is within or past its expected mean-reversion window
          • Any timing or concentration concerns visible in the book as a whole
          • An overall health assessment of the portfolio

        Write in a clear, professional tone.  150–200 words.  Flowing prose, no bullet
        points.  Present tense.  Do not repeat raw numbers back verbatim; synthesise them
        into a coherent picture.
    """)

    user_message = dedent(f"""\
        Please provide the portfolio read for today's morning risk brief.

        {portfolio_context}

        Summarise what this book is currently doing and whether the open positions
        are on track relative to their mean-reversion half-lives.
    """)

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def call_tail_risk_officer(
    client: anthropic.Anthropic,
    currency_context: str,
) -> str:
    """
    Call (b) — Macro and geopolitical tail-risk briefing.

    Ask Claude to act as a risk officer identifying risks specific to a
    market-neutral book holding the given currency relationships today.
    Emphasis is on correlation/cointegration-break risks and scheduled events,
    not directional forecasts.
    """
    system_prompt = dedent("""\
        You are a senior FX risk officer delivering a daily tail-risk briefing to a
        market-neutral statistical-arbitrage desk.

        Critical constraint: this book carries NO net directional currency exposure.
        Do NOT make directional calls (e.g. "USD will weaken").  Your mandate is
        to identify risks that could BREAK the spread relationships themselves, not
        predict which direction prices move.

        Focus exclusively on:
          1. Scheduled macro / central-bank events that could spike cross-currency
             volatility and disrupt cointegrated spread relationships
          2. Geopolitical or macro surprises that could cause correlation breakdowns
             between the pairs held
          3. Regime-change risks: structural shifts that might invalidate the
             historical cointegration assumptions underlying each pair
          4. Liquidity or market-microstructure risks relevant to simultaneous
             dual-leg FX execution

        Structure your response as:
          (i)   Top 3 Event Risks — specific scheduled or foreseeable triggers
          (ii)  Correlation-Break Watch List — pairs or currency blocs most at risk
          (iii) Overall Tail-Risk Temperature — one sentence summary

        Tone: professional risk-officer.  200–250 words total.
    """)

    user_message = dedent(f"""\
        Please produce today's tail-risk briefing for the desk.

        {currency_context}

        Identify the most pressing macro and geopolitical tail risks relevant to a
        market-neutral book holding these currency spread relationships.  Emphasise
        correlation-break and event risks.  No directional calls.
    """)

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Report assembly
# ═══════════════════════════════════════════════════════════════════════════

_ADVISORY_UNAVAILABLE = (
    "*Advisory section unavailable — "
    "Claude API call failed or ANTHROPIC_API_KEY is not set.*"
)


def assemble_markdown_report(
    det_section: str,
    advisory_a: str,
    advisory_b: str,
    now_utc: datetime,
) -> str:
    """
    Combine all sections into the final Markdown report.

    The deterministic section is wrapped in a fenced code block so the
    fixed-width table formatting is preserved in any Markdown renderer.
    """
    date_str = now_utc.strftime("%Y-%m-%d")
    timestamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    return "\n".join([
        f"# FX Stat-Arb Advisory Report — {date_str}",
        "",
        "> **Dual-Brain Architecture Note**  ",
        "> The deterministic tables in §1 are computed in plain Python with zero LLM",
        "> involvement and are the authoritative record of portfolio state.",
        "> §2 and §3 are Claude's narrative commentary — they inform the risk officer",
        "> but have no influence on the execution path.",
        "",
        "---",
        "",
        "## 1 · Deterministic Portfolio Snapshot",
        "",
        "```text",
        det_section.strip(),
        "```",
        "",
        "---",
        "",
        "## 2 · Claude Advisory — Portfolio Read",
        "",
        advisory_a,
        "",
        "---",
        "",
        "## 3 · Claude Advisory — Macro & Geopolitical Tail Risk",
        "",
        advisory_b,
        "",
        "---",
        "",
        f"*Report generated {timestamp} by `claude_analyst.py` v{__version__}*",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_state(path: str) -> dict[str, Any]:
    """
    Load and perform basic validation on state.json.

    Exits with a descriptive error rather than an opaque traceback.
    """
    state_path = Path(path)
    if not state_path.exists():
        print(
            f"ERROR: state file not found: {state_path.resolve()}\n"
            f"       Check that the Execution Brain has written state.json "
            f"and that --state points to the correct location.",
            file=sys.stderr,
        )
        sys.exit(1)

    with state_path.open(encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: state.json contains invalid JSON: {exc}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, dict):
        print(
            "ERROR: state.json must be a JSON object (top-level dict).",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def write_report(content: str, now_utc: datetime) -> Path:
    """Write the assembled report to a dated Markdown file."""
    filename = f"report_{now_utc.strftime('%Y-%m-%d')}.md"
    report_path = Path(__file__).parent / filename
    report_path.write_text(content, encoding="utf-8")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="claude_analyst.py",
        description="Advisory Brain — FX Stat-Arb daily risk briefing via Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Environment variables:
              ANTHROPIC_API_KEY   Required for Claude advisory sections.
                                  If absent, deterministic report still runs.

            Examples:
              python claude_analyst.py
              python claude_analyst.py --state /data/live/state.json
        """),
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE_PATH,
        metavar="PATH",
        help=f"Path to state.json from the Execution Brain (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()

    # ── 0. Bootstrap ──────────────────────────────────────────────────────
    now_utc: datetime = datetime.now(tz=timezone.utc)

    api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")

    print(_banner("ADVISORY BRAIN — FX Stat-Arb Daily Briefing"))
    print(f"  Version   : {__version__}")
    print(f"  Timestamp : {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Model     : {MODEL_ID}")
    print(f"  State     : {Path(args.state).resolve()}")
    api_status = "✓ found" if api_key else "✗ NOT SET — advisory sections will be skipped"
    print(f"  API key   : {api_status}")

    if not api_key:
        print(
            "\n  WARNING: ANTHROPIC_API_KEY is not set.\n"
            "           Set it with: export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "           The deterministic report will still be produced.",
            file=sys.stderr,
        )

    # ── 1. Load and validate state ────────────────────────────────────────
    print(f"\n  Loading state.json …\n")
    raw_state: dict[str, Any] = load_state(args.state)

    # ── 2. Deterministic analytics (no LLM) ──────────────────────────────
    pair_metrics: list[dict[str, Any]] = []
    for pair_tag, pair_data in raw_state.items():
        try:
            pm = compute_pair_metrics(pair_tag, pair_data, now_utc)
            pair_metrics.append(pm)
        except (ValueError, KeyError, TypeError) as exc:
            print(
                f"  WARNING: Skipping pair '{pair_tag}' due to data error: {exc}",
                file=sys.stderr,
            )

    summary: dict[str, Any] = compute_book_summary(pair_metrics)
    det_section: str = build_deterministic_section(pair_metrics, summary, now_utc)

    # Always print the deterministic section — it is unconditionally available
    print(det_section)

    # ── 3. Advisory layer (Claude) ────────────────────────────────────────
    advisory_a: str = _ADVISORY_UNAVAILABLE
    advisory_b: str = _ADVISORY_UNAVAILABLE

    if summary["n_pairs"] == 0:
        print("  [INFO] No open positions — advisory sections skipped.\n")

    elif not api_key:
        # Already warned above; advisory vars hold the unavailable message
        pass

    else:
        client = anthropic.Anthropic(api_key=api_key)
        portfolio_ctx = build_portfolio_context(pair_metrics, summary)
        currency_ctx  = build_currency_context(summary, pair_metrics)

        # ── Call (a): Portfolio analyst ───────────────────────────────
        print(_banner("CLAUDE ADVISORY (a) — PORTFOLIO READ"))
        print()
        try:
            advisory_a = call_portfolio_analyst(client, portfolio_ctx)
            print(advisory_a)
        except anthropic.APIConnectionError as exc:
            advisory_a = f"*Advisory unavailable — connection error: {exc}*"
            print(f"  [ERROR] Connection to Anthropic API failed: {exc}", file=sys.stderr)
            print(advisory_a)
        except anthropic.RateLimitError as exc:
            advisory_a = f"*Advisory unavailable — rate limit exceeded: {exc}*"
            print(f"  [ERROR] Rate limit hit: {exc}", file=sys.stderr)
            print(advisory_a)
        except anthropic.APIStatusError as exc:
            advisory_a = f"*Advisory unavailable — API error {exc.status_code}: {exc.message}*"
            print(
                f"  [ERROR] API returned status {exc.status_code}: {exc.message}",
                file=sys.stderr,
            )
            print(advisory_a)
        except Exception as exc:  # noqa: BLE001  — catch-all so report always completes
            advisory_a = f"*Advisory unavailable — unexpected error: {type(exc).__name__}*"
            print(f"  [ERROR] Unexpected error in portfolio analyst call: {exc}", file=sys.stderr)
            print(advisory_a)

        print()

        # ── Call (b): Tail-risk officer ───────────────────────────────
        print(_banner("CLAUDE ADVISORY (b) — MACRO & GEOPOLITICAL TAIL RISK"))
        print()
        try:
            advisory_b = call_tail_risk_officer(client, currency_ctx)
            print(advisory_b)
        except anthropic.APIConnectionError as exc:
            advisory_b = f"*Advisory unavailable — connection error: {exc}*"
            print(f"  [ERROR] Connection to Anthropic API failed: {exc}", file=sys.stderr)
            print(advisory_b)
        except anthropic.RateLimitError as exc:
            advisory_b = f"*Advisory unavailable — rate limit exceeded: {exc}*"
            print(f"  [ERROR] Rate limit hit: {exc}", file=sys.stderr)
            print(advisory_b)
        except anthropic.APIStatusError as exc:
            advisory_b = f"*Advisory unavailable — API error {exc.status_code}: {exc.message}*"
            print(
                f"  [ERROR] API returned status {exc.status_code}: {exc.message}",
                file=sys.stderr,
            )
            print(advisory_b)
        except Exception as exc:  # noqa: BLE001
            advisory_b = f"*Advisory unavailable — unexpected error: {type(exc).__name__}*"
            print(f"  [ERROR] Unexpected error in tail-risk call: {exc}", file=sys.stderr)
            print(advisory_b)

        print()

    # ── 4. Assemble and write report ──────────────────────────────────────
    report_md  = assemble_markdown_report(det_section, advisory_a, advisory_b, now_utc)
    report_path = write_report(report_md, now_utc)

    print(_hr("═"))
    print(f"  Report written → {report_path}")
    print(_hr("═"))
    print()


if __name__ == "__main__":
    main()