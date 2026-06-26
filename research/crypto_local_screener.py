import os
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# These are the exact crypto assets allowed in the rules (ignoring BAR due to liquidity)
VALID_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

def load_local_data(symbol):
    """Loads the M15 Parquet data fetched from Binance."""
    filename = f"crypto_data/{symbol}.parquet"
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_parquet(filename)
        return df['close']
    except Exception as e:
        print(f"⚠️ Error loading {filename}: {e}")
        return None

def calculate_half_life(spread):
    """Computes the Ornstein-Uhlenbeck mean-reversion half-life."""
    s = spread.dropna()
    lag = s.shift(1)
    delta = (s - lag).dropna()
    lag = lag.loc[delta.index]
    
    # Run OLS: d_y = const + lambda * y_{t-1}
    try:
        res = sm.OLS(delta.values, sm.add_constant(lag.values)).fit()
        lam = res.params[1]
        if lam >= 0 or (1 + lam) <= 0:
            return np.inf
        return -np.log(2) / np.log(1 + lam)
    except:
        return np.inf

def analyze_pair(series_y, series_x, name_y, name_x):
    """Runs OLS, ADF, and Half-Life on two aligned price series."""
    # Align data by exact timestamps
    df = pd.concat([series_y, series_x], axis=1).dropna()
    df.columns = ["y", "x"]
    
    if len(df) < 500: # Need sufficient sample size
        return None
        
    res = sm.OLS(df["y"].values, sm.add_constant(df["x"].values)).fit()
    alpha = res.params[0]
    beta = res.params[1]
    
    # Calculate Spread
    spread = df["y"] - (alpha + beta * df["x"])
    mu = spread.mean()
    sigma = spread.std()
    
    try:
        adf_pvalue = adfuller(spread.values, autolag="AIC")[1]
    except:
        adf_pvalue = 1.0
        
    hl_bars = calculate_half_life(spread)
    hl_hours = hl_bars * 15 / 60.0 # Convert M15 bars to hours
    
    current_z = (spread.iloc[-1] - mu) / sigma
    
    return {
        "Pair": f"{name_y} / {name_x}",
        "Beta": beta,
        "ADF_P": adf_pvalue,
        "Half_Life_Hrs": hl_hours,
        "Current_Z": current_z
    }

def main():
    print("==================================================")
    print("🔬 LOCAL CRYPTO STAT-ARB SCREENER (M15)")
    print("==================================================")
    
    data_dict = {}
    for sym in VALID_CRYPTO:
        series = load_local_data(sym)
        if series is not None:
            data_dict[sym] = series
            print(f"✅ Loaded {sym}: {len(series)} M15 candles")
        else:
            print(f"❌ Missing data for {sym}. Run binance_fetcher.py first.")
            
    if len(data_dict) < 2:
        print("\n⚠️ Not enough data to compare pairs. Exiting.")
        return

    print("\n⏳ Running Cointegration Math on all combinations...\n")
    
    results = []
    # Generate all unique combinations (e.g., BTC/ETH, BTC/SOL)
    pairs = list(itertools.combinations(data_dict.keys(), 2))
    
    for y_sym, x_sym in pairs:
        metrics = analyze_pair(data_dict[y_sym], data_dict[x_sym], y_sym, x_sym)
        if metrics:
            results.append(metrics)
            
    if not results:
        print("No valid data for analysis.")
        return
        
    summary = pd.DataFrame(results)
    summary = summary.sort_values(by="ADF_P")
    
    print(f"{'PAIR':<18} | {'BETA':<8} | {'ADF P-VAL':<10} | {'HALF-LIFE':<10} | {'LIVE Z-SCORE'}")
    print("-" * 70)
    
    for _, row in summary.iterrows():
        p_str = f"{row['ADF_P']:.4f}"
        if row['ADF_P'] < 0.05:
            p_str += " ✅"
        else:
            p_str += " ❌"
            
        hl_str = f"{row['Half_Life_Hrs']:.1f}h" if np.isfinite(row['Half_Life_Hrs']) else "Inf"
        
        print(f"{row['Pair']:<18} | {row['Beta']:<8.4f} | {p_str:<10} | {hl_str:<10} | {row['Current_Z']:+.2f}")
        
    print("-" * 70)
    print("\n[Analysis Guide]")
    print("1. ADF P-VAL: Must be < 0.05 to prove the spread is mean-reverting.")
    print("2. HALF-LIFE: Look for pairs that revert in 4 to 24 hours.")
    print("3. BETA: Your hedge ratio. If Beta is 15, you short 15 X for every 1 Y.")

if __name__ == "__main__":
    main()