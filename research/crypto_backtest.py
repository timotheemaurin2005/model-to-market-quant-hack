"""
crypto_backtest.py — standalone validation harness for the 5/20 EMA crossover
on BTC-USD, ETH-USD, SOL-USD. VALIDATION ONLY — does not touch signal_server.py,
does not size or allocate. Decides whether the signal deserves any capital.

Signal (fixed, not tuned to fit):
  ema_fast = 5-bar EMA, ema_slow = 20-bar EMA
  entry filter: abs((ema_fast - ema_slow) / ema_slow) > 0.002
  position: +1 if ema_fast > ema_slow else -1, entered ONLY on a crossover bar
  exit: reverse crossover OR 2.5% max adverse excursion from entry (intrabar
        high/low touch, not just close)

Costs are a round-trip charge per trade (entry + exit), modeling
spread+slippage+impact. Reported at 5/10/20 bps per side.

Split is chronological — last 30% of bars is the untouched holdout. EMAs are
computed over the full series (causal, no leakage) so the holdout doesn't
suffer an EMA cold-start; trade COUNTS/win-rate/holding-period are restricted
to trades whose ENTRY falls inside the holdout, so no train-period decision
is credited to holdout performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = {"BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "SOL-USD": "SOL-USD"}
INTERVAL = "15m"
PERIOD = "60d"
HOLDOUT_FRAC = 0.30

EMA_FAST = 5
EMA_SLOW = 20
ENTRY_FILTER = 0.002
MAE_STOP = 0.025

COST_BPS_PER_SIDE_LIST = [5, 10, 20]
PRIMARY_COST_BPS = 10

BARS_PER_YEAR = 365.25 * 24 * 4  # 15-min bars, 24/7 crypto


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is not None else idx
    return df.sort_index()


# ── Signal ───────────────────────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    out["filter_ok"] = (out["ema_fast"] - out["ema_slow"]).abs() / out["ema_slow"] > ENTRY_FILTER
    fast_above = out["ema_fast"] > out["ema_slow"]
    out["cross_up"] = fast_above & ~fast_above.shift(1, fill_value=fast_above.iloc[0])
    out["cross_down"] = (~fast_above) & fast_above.shift(1, fill_value=fast_above.iloc[0])
    return out


# ── Trade simulation (causal state machine, bar by bar) ────────────────────

def simulate(df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Returns (position_series, trades). position_series[t] = position HELD
    ENTERING bar t (i.e. exposure during bar t's price move)."""
    n = len(df)
    position_at_open = np.zeros(n)  # exposure during bar t
    trades = []

    pos = 0
    entry_price = None
    entry_idx = None

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    cross_up = df["cross_up"].values
    cross_down = df["cross_down"].values
    filter_ok = df["filter_ok"].values
    idx = df.index

    for t in range(EMA_SLOW, n):
        position_at_open[t] = pos  # exposure carried INTO this bar from prior decision

        if pos == 0:
            if filter_ok[t]:
                if cross_up[t]:
                    pos = 1
                    entry_price = close[t]
                    entry_idx = t
                elif cross_down[t]:
                    pos = -1
                    entry_price = close[t]
                    entry_idx = t
        else:
            # Check stop first (intrabar touch), then reverse crossover.
            if pos == 1:
                mae = (entry_price - low[t]) / entry_price
                stopped = mae >= MAE_STOP
            else:
                mae = (high[t] - entry_price) / entry_price
                stopped = mae >= MAE_STOP

            exit_now = stopped or (pos == 1 and cross_down[t]) or (pos == -1 and cross_up[t])
            if exit_now:
                exit_price = close[t]
                gross_ret = pos * (exit_price / entry_price - 1.0)
                trades.append({
                    "direction": pos,
                    "entry_time": idx[entry_idx],
                    "exit_time": idx[t],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_idx": entry_idx,
                    "exit_idx": t,
                    "holding_bars": t - entry_idx,
                    "gross_return": gross_ret,
                    "stopped_out": stopped,
                })
                pos = 0
                entry_price = None
                entry_idx = None

    position_series = pd.Series(position_at_open, index=df.index)
    return position_series, trades


# ── Bar-level net-of-cost returns ───────────────────────────────────────────

def bar_returns_net_of_cost(df: pd.DataFrame, position: pd.Series, trades: list[dict],
                             cost_bps_per_side: float) -> pd.Series:
    """Per-bar strategy return = position[t-1]*(close[t]/close[t-1]-1), minus
    a cost charge dropped into the bar where each entry/exit occurs."""
    close = df["Close"]
    gross = position.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)

    cost = cost_bps_per_side / 10000.0
    net = gross.copy()
    for tr in trades:
        # entry cost hits the bar AFTER entry (first bar with the new exposure)
        entry_cost_idx = tr["entry_idx"] + 1
        if entry_cost_idx < len(net):
            net.iloc[entry_cost_idx] -= cost
        # exit cost hits the bar after exit (return to flat priced in next bar's pct_change baseline)
        exit_cost_idx = tr["exit_idx"] + 1
        if exit_cost_idx < len(net):
            net.iloc[exit_cost_idx] -= cost
    return net


# ── Metrics ──────────────────────────────────────────────────────────────────

def sharpe_15min(returns: pd.Series) -> tuple[float, int]:
    """Sharpe on 15-min equity returns, per CLAUDE.md's definition: mean/std
    of the per-bar return series. Returns (sharpe, n_obs); CLAUDE.md caps the
    competition score at 50 if n_obs < 8 -- we just flag that here."""
    r = returns.dropna()
    n = len(r)
    if n < 2 or r.std() == 0:
        return 0.0, n
    return float(r.mean() / r.std()), n


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    return float(dd.min())


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


# ── Per-instrument evaluation ───────────────────────────────────────────────

def evaluate_instrument(ticker: str) -> dict:
    df = fetch_ohlc(ticker)
    sig = compute_signal(df)

    n = len(sig)
    split_idx = int(n * (1 - HOLDOUT_FRAC))
    holdout_start_time = sig.index[split_idx]

    position, trades = simulate(sig)

    holdout_mask = sig.index >= holdout_start_time
    holdout_trades = [t for t in trades if t["entry_time"] >= holdout_start_time]

    bench_returns = sig["Close"].pct_change().fillna(0.0)
    bench_holdout = bench_returns.loc[holdout_mask]
    bench_total_return = total_return(bench_holdout)
    bench_sharpe, _ = sharpe_15min(bench_holdout)

    cost_results = {}
    for cost_bps in COST_BPS_PER_SIDE_LIST:
        net = bar_returns_net_of_cost(sig, position, trades, cost_bps)
        net_holdout = net.loc[holdout_mask]

        n_trades = len(holdout_trades)
        wins = sum(1 for t in holdout_trades if t["gross_return"] - 2 * cost_bps / 10000.0 > 0)
        win_rate = wins / n_trades if n_trades else float("nan")
        avg_hold_bars = np.mean([t["holding_bars"] for t in holdout_trades]) if n_trades else float("nan")

        sharpe, n_obs = sharpe_15min(net_holdout)
        ret = total_return(net_holdout)
        mdd = max_drawdown(net_holdout)

        passed = (ret > bench_total_return) and (sharpe > 0)

        cost_results[cost_bps] = {
            "net_return": ret,
            "sharpe": sharpe,
            "sharpe_n_obs": n_obs,
            "max_drawdown": mdd,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_holding_bars": avg_hold_bars,
            "avg_holding_minutes": avg_hold_bars * 15 if n_trades else float("nan"),
            "passed": passed,
            "net_returns_series": net_holdout,
        }

    return {
        "ticker": ticker,
        "n_bars_total": n,
        "holdout_start": holdout_start_time,
        "holdout_end": sig.index[-1],
        "n_bars_holdout": int(holdout_mask.sum()),
        "bench_total_return": bench_total_return,
        "bench_sharpe": bench_sharpe,
        "bench_returns_holdout": bench_holdout,
        "costs": cost_results,
        "holdout_mask": holdout_mask,
        "position": position,
        "trades": trades,
    }


# ── Pooled (equal-weight, 3-instrument) evaluation ──────────────────────────

def evaluate_pooled(results: list[dict], cost_bps: int) -> dict:
    # Align on the intersection of holdout timestamps (data should already match).
    series = [r["costs"][cost_bps]["net_returns_series"] for r in results]
    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)
    aligned = [s.loc[common_idx] for s in series]
    pooled_returns = sum(aligned) / len(aligned)  # equal-weight average

    bench_series = [r["bench_returns_holdout"].loc[common_idx] for r in results]
    pooled_bench = sum(bench_series) / len(bench_series)

    sharpe, n_obs = sharpe_15min(pooled_returns)
    ret = total_return(pooled_returns)
    mdd = max_drawdown(pooled_returns)
    bench_ret = total_return(pooled_bench)
    bench_sharpe, _ = sharpe_15min(pooled_bench)

    n_trades = sum(r["costs"][cost_bps]["n_trades"] for r in results)
    passed = (ret > bench_ret) and (sharpe > 0)

    return {
        "net_return": ret,
        "sharpe": sharpe,
        "sharpe_n_obs": n_obs,
        "max_drawdown": mdd,
        "n_trades": n_trades,
        "bench_total_return": bench_ret,
        "bench_sharpe": bench_sharpe,
        "passed": passed,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%" if x == x else "  n/a "


def main():
    print(f"5/20 EMA crossover backtest — VALIDATION ONLY, not wired to live.")
    print(f"Data: {list(TICKERS)} | {INTERVAL} bars | period={PERIOD} | "
          f"holdout = last {int(HOLDOUT_FRAC*100)}% (chronological, never random)\n")

    results = []
    for ticker in TICKERS:
        r = evaluate_instrument(ticker)
        results.append(r)
        print(f"{ticker}: {r['n_bars_total']} total bars -> holdout "
              f"{r['holdout_start']} .. {r['holdout_end']} "
              f"({r['n_bars_holdout']} bars)")

    print("\n" + "=" * 100)
    print(f"PER-INSTRUMENT RESULTS (holdout only) — primary cost = {PRIMARY_COST_BPS} bps/side")
    print("=" * 100)
    header = (f"{'Ticker':10s} {'NetRet':>9s} {'B&H Ret':>9s} {'Sharpe':>8s} "
              f"{'B&H Sharpe':>10s} {'MaxDD':>8s} {'#Trades':>8s} {'WinRate':>8s} "
              f"{'AvgHold(min)':>13s} {'PASS?':>6s}")
    print(header)
    for r in results:
        c = r["costs"][PRIMARY_COST_BPS]
        print(f"{r['ticker']:10s} {fmt_pct(c['net_return']):>9s} {fmt_pct(r['bench_total_return']):>9s} "
              f"{c['sharpe']:8.3f} {r['bench_sharpe']:10.3f} {fmt_pct(c['max_drawdown']):>8s} "
              f"{c['n_trades']:8d} {c['win_rate']*100 if c['win_rate']==c['win_rate'] else float('nan'):7.1f}% "
              f"{c['avg_holding_minutes']:13.1f} {'PASS' if c['passed'] else 'FAIL':>6s}")

    print("\n" + "=" * 100)
    print("COST SENSITIVITY (net return / Sharpe per instrument, by cost per side)")
    print("=" * 100)
    header2 = f"{'Ticker':10s}" + "".join(f"{f'{b}bps Ret':>12s}{f'{b}bps Shrp':>12s}" for b in COST_BPS_PER_SIDE_LIST)
    print(header2)
    for r in results:
        row = f"{r['ticker']:10s}"
        for b in COST_BPS_PER_SIDE_LIST:
            c = r["costs"][b]
            row += f"{fmt_pct(c['net_return']):>12s}{c['sharpe']:12.3f}"
        print(row)

    print("\n" + "=" * 100)
    print(f"POOLED (equal-weight, 3 instruments) — by cost per side")
    print("=" * 100)
    for b in COST_BPS_PER_SIDE_LIST:
        p = evaluate_pooled(results, b)
        tag = "PASS" if p["passed"] else "FAIL"
        print(f"  {b:>3d} bps/side: net_ret={fmt_pct(p['net_return'])}  "
              f"sharpe={p['sharpe']:.3f} (n={p['sharpe_n_obs']})  "
              f"B&H_ret={fmt_pct(p['bench_total_return'])}  B&H_sharpe={p['bench_sharpe']:.3f}  "
              f"maxDD={fmt_pct(p['max_drawdown'])}  n_trades={p['n_trades']}  -> {tag}")

    print("\n" + "=" * 100)
    print(f"VERDICT — acceptance bar: PASS only if net return (at {PRIMARY_COST_BPS} bps/side) > "
          f"buy-and-hold AND Sharpe > 0, on the HOLDOUT.")
    print("=" * 100)
    for r in results:
        c = r["costs"][PRIMARY_COST_BPS]
        beats = c["passed"]
        print(f"  {r['ticker']}: does 5/20 EMA momentum beat buy-and-hold net of costs? "
              f"{'YES' if beats else 'NO'}  "
              f"(net_ret={fmt_pct(c['net_return'])} vs B&H={fmt_pct(r['bench_total_return'])}, "
              f"sharpe={c['sharpe']:.3f}, n_trades={c['n_trades']})")
    pooled_primary = evaluate_pooled(results, PRIMARY_COST_BPS)
    print(f"  POOLED: {'YES' if pooled_primary['passed'] else 'NO'} "
          f"(net_ret={fmt_pct(pooled_primary['net_return'])} vs "
          f"B&H={fmt_pct(pooled_primary['bench_total_return'])}, sharpe={pooled_primary['sharpe']:.3f})")

    print(f"\nFLAG: the holdout window above spans {results[0]['n_bars_holdout']} bars "
          f"(~{results[0]['n_bars_holdout']*15/60/24:.1f} days). The competition's live window "
          f"is ~5 days. Even a PASS here is calibrated on a regime ~{results[0]['n_bars_holdout']*15/60/24/5:.0f}x "
          f"longer than live — a PASS does NOT guarantee the same edge holds in the live window; "
          f"crypto is excluded from the live watchlist per CLAUDE.md regardless of this result "
          f"(\"decouples faster than the 15-min loop can act\").")


if __name__ == "__main__":
    main()
