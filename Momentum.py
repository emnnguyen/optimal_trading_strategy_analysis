import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# === Step 1: Load S&P 500 data to define market regime ===
spx = pd.read_csv("/Users/ayy/Desktop/sta160/Stock/^GSPC.csv")
spx['Date'] = pd.to_datetime(spx['Date'])
spx = spx.sort_values('Date')

# Calculate 200-day SMA and label Bull / Bear regimes
spx['SMA200'] = spx['Close'].rolling(window=200).mean()
spx['Market_Regime'] = np.where(spx['Close'] > spx['SMA200'], 'Bull', 'Bear')
spx = spx[['Date', 'Market_Regime']]

# === Step 2: Parameters ===
path = "/Users/ayy/Desktop/sta160/Stock"
files = glob.glob(path + "/*.csv")

# Split full history into 5 multi-year periods
periods = [
    ('2000-01-01', '2005-12-31'),
    ('2006-01-01', '2010-12-31'),
    ('2011-01-01', '2015-12-31'),
    ('2016-01-01', '2020-12-31'),
    ('2021-01-01', '2025-12-31')
]

# Define different momentum lookback windows (trading days)
lookbacks = {
    "1W": 5,
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252
}

results = []

# === Step 3: Loop through all stocks ===
for file in files:
    ticker = os.path.basename(file).replace(".csv", "")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Daily returns
    df['Return'] = df['Close'].pct_change()
    # Merge market regime info from S&P 500
    df = pd.merge(df, spx, on='Date', how='left')

    # === Step 4: Momentum-based trading strategy ===
    for label, lb in lookbacks.items():
        # Calculate momentum (price relative to lb days ago)
        df[f'Momentum_{label}'] = df['Close'] / df['Close'].shift(lb) - 1
        # Trading signal: 1 = long, -1 = short
        df[f'Signal_{label}'] = np.where(df[f'Momentum_{label}'] > 0, 1, -1)
        # Strategy daily return
        df[f'Strategy_{label}'] = df[f'Signal_{label}'].shift(1) * df['Return']
        # Cumulative performance
        df[f'Cumulative_{label}'] = (1 + df[f'Strategy_{label}']).cumprod()

        # === Step 5: Evaluate by period & market regime ===
        for (start, end) in periods:
            sub = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
            if len(sub) < lb:
                continue

            for regime in ['Bull', 'Bear']:
                sub_regime = sub[sub['Market_Regime'] == regime]
                if len(sub_regime) < lb:
                    continue

                # Calculate performance metrics
                cum_return = (1 + sub_regime[f'Strategy_{label}']).prod() - 1
                avg_daily = sub_regime[f'Strategy_{label}'].mean()
                std_daily = sub_regime[f'Strategy_{label}'].std()
                sharpe = avg_daily / std_daily * np.sqrt(252) if std_daily != 0 else np.nan

                results.append({
                    "Ticker": ticker,
                    "Period": f"{start[:4]}–{end[:4]}",
                    "Market_Regime": regime,
                    "Lookback": label,
                    "Cumulative Return": cum_return,
                    "Sharpe Ratio": sharpe
                })

# === Step 6: Combine all results ===
results_df = pd.DataFrame(results)
pd.set_option('display.max_rows', None)
print("== Multi-Period Momentum Strategy Results with Bull/Bear Regimes ==")
print(results_df.head(30))

# === Step 7: Find best strategy per company (highest Sharpe) ===
best_strategies = results_df.loc[
    results_df.groupby("Ticker")["Sharpe Ratio"].idxmax()
].reset_index(drop=True)

print("\n== Best Strategy per Company (Including Market Regime) ==")
print(best_strategies.head(30))

# === Step 8: Visualization — Average Sharpe by Lookback & Market Regime ===
summary = results_df.groupby(["Market_Regime", "Lookback"]).agg({
    "Cumulative Return": "mean",
    "Sharpe Ratio": "mean"
}).reset_index()

plt.figure(figsize=(10,6))
for regime, color in zip(['Bull', 'Bear'], ['orange', 'steelblue']):
    subset = summary[summary["Market_Regime"] == regime]
    plt.plot(subset["Lookback"], subset["Sharpe Ratio"], marker='o', label=regime, color=color)
    for _, row in subset.iterrows():
        plt.text(row["Lookback"], row["Sharpe Ratio"]+0.02,
                 f'{row["Sharpe Ratio"]:.2f}', ha='center', fontsize=9)

plt.title("Average Sharpe Ratio by Lookback in Bull vs Bear Markets (2000–2025)")
plt.xlabel("Momentum Lookback Window")
plt.ylabel("Average Sharpe Ratio")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.close()

# === Step 9: Summary statistics for robustness ===
print("\n=== Mean Sharpe by Market Regime ===")
print(summary.groupby("Market_Regime")["Sharpe Ratio"].mean())

print("\n=== Std of Sharpe by Market Regime ===")
print(summary.groupby("Market_Regime")["Sharpe Ratio"].std())

print("\n=== Proportion of Positive Sharpe Ratios (All Strategies) ===")
print((results_df["Sharpe Ratio"] > 0).mean())

# Keep terminal window open (useful when running from VS Code)
input("\nPress Enter to exit...")

