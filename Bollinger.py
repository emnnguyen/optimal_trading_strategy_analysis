import pandas as pd
from pathlib import Path

# === CONFIG ===
DATA_DIR = Path.home() / "Downloads" / "stockdata" / "data"
START, END = "2000-01-03", "2024-12-30"
WINDOW, K = 20, 2

TICKERS = [
    '^GSPC', 'AAPL','MSFT','NVDA','GOOGL','META','AVGO','CSCO','ADBE','CRM','INTC',
    'JPM','BAC','WFC','GS','MS','BLK','AXP','SCHW',
    'UNH','JNJ','LLY','ABBV','MRK','PFE','TMO',
    'AMZN','TSLA','HD','MCD','NKE','SBUX',
    'WMT','PG','KO','PEP',
    'BA','CAT','UPS','RTX','HON',
    'XOM','CVX','COP',
    'DIS','NFLX','CMCSA',
    'NEE','DUK','AMT','PLD'
]

def bollinger_actions(df, window=20, k=2.0):
    close = df["Close"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = ma + k*std, ma - k*std
    sig_buy, sig_sell = (close < lower), (close > upper)
    act_buy, act_sell = sig_buy.shift(1).fillna(False), sig_sell.shift(1).fillna(False)
    buy_dates = df.index[act_buy].strftime("%Y-%m-%d").tolist()
    sell_dates = df.index[act_sell].strftime("%Y-%m-%d").tolist()
    return buy_dates, sell_dates

actions = []

for tk in TICKERS:
    file = DATA_DIR / f"{tk}.csv"
    if not file.exists():
        print(f"[skip] {tk} (file missing)")
        continue

    df = pd.read_csv(file, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df = df.loc[START:END, ["Open","High","Low","Close","Volume"]].dropna()
    if df.empty: continue

    buys, sells = bollinger_actions(df, WINDOW, K)
    actions += [{"Ticker": tk, "Date": d, "Action": "BUY"} for d in buys]
    actions += [{"Ticker": tk, "Date": d, "Action": "SELL"} for d in sells]

actions_df = pd.DataFrame(actions).sort_values(["Ticker","Date"]).reset_index(drop=True)

# === OUTPUT ===
out_csv = DATA_DIR / "bollinger_actions_all.csv"
actions_df.to_csv(out_csv, index=False, date_format="%Y-%m-%d")

print(f"\nSaved CSV: {out_csv}")
print(f"Total rows: {len(actions_df)}   |   Tickers: {actions_df['Ticker'].nunique()}")
