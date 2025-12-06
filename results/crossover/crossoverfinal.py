import os
import glob
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


DATA_DIR = "/Users/vanessaliu/Desktop/STA160/featurecompany"
OUT_SUMMARY = "results_feature_summary.csv"
OUT_YEARLY  = "feature_yearly_breakdown.csv"
OUT_GRID    = "feature_grid_results.csv"
OUT_SECTOR  = "feature_sector_summary.csv"
OUT_PARAM   = "feature_param_summary.csv"
OUT_BEST_SHARPE = "feature_best_by_sharpe.csv"
OUT_BEST_RETURN = "feature_best_by_return.csv"
OUT_YEARLY_SHARPE = "feature_yearly_best_by_sharpe.csv"
OUT_YEARLY_RETURN = "feature_yearly_best_by_return.csv"
OUT_TEXT_SUMMARY = "feature_analysis_summary.txt"

USE_WEEKLY  = False
TRAIN_RATIO = 0.7
CASH        = 10_000
COMMISSION  = 0.001
TEST_MODE   = False                   


PERIODS = [5, 10, 20, 30, 50, 100, 150]


SECTOR_MAP = {
    # Technology (10)
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","GOOGL":"Technology",
    "META":"Technology","AVGO":"Technology","CSCO":"Technology","ADBE":"Technology",
    "CRM":"Technology","INTC":"Technology",
    # Financials (8)
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","BLK":"Financials","AXP":"Financials","SCHW":"Financials",
    # Healthcare (7)
    "UNH":"Healthcare","JNJ":"Healthcare","LLY":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","PFE":"Healthcare","TMO":"Healthcare",
    # Consumer Discretionary (6)
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary","HD":"Consumer Discretionary",
    "MCD":"Consumer Discretionary","NKE":"Consumer Discretionary","SBUX":"Consumer Discretionary",
    # Consumer Staples (4)
    "WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    # Industrials (5)
    "BA":"Industrials","CAT":"Industrials","UPS":"Industrials","RTX":"Industrials","HON":"Industrials",
    # Energy (3)
    "XOM":"Energy","CVX":"Energy","COP":"Energy",
    # Communication Services (3)
    "DIS":"Communication Services","NFLX":"Communication Services","CMCSA":"Communication Services",
    # Utilities (2)
    "NEE":"Utilities","DUK":"Utilities",
    # Real Estate (2)
    "AMT":"Real Estate","PLD":"Real Estate",
}


def SMA(series, n):
    # Backtesting data is ndarray-like and lacks a rolling method; implement SMA via convolution
    import numpy as np
    if n <= 1:
        return series
    kernel = np.ones(int(n)) / float(n)
    return np.convolve(series, kernel, mode="same")

class SmaCross(Strategy):
    n_fast = 10
    n_slow = 20
    
    def init(self):
        close = self.data.Close
        self.ma_fast = self.I(SMA, close, self.n_fast)
        self.ma_slow = self.I(SMA, close, self.n_slow)
    
    def next(self):
        if crossover(self.ma_fast, self.ma_slow):
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy()
        elif crossover(self.ma_slow, self.ma_fast):
            if self.position.is_long:
                self.position.close()

class BuyHold(Strategy):
    def init(self): pass
    def next(self):
        if not self.position:
            self.buy()


# ========== Data loading & cleaning ==========
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize mixed-case column names: open/Open/OPEN -> Open, etc.
    """
    rename_map = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=rename_map)
    return df

def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Force specified columns to numeric: strip thousands separators, percent signs, and other common 'dirty' chars.
    Invalid values are converted to NaN (dropped later).
    """
    for c in cols:
        if c not in df.columns:
            continue
        # Convert to string first, then clean common symbols
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)   # thousands separator
            .str.replace("%", "", regex=False)   # percent sign (if any)
            .str.strip()
            .replace({"": None, "-": None, "null": None, "None": None, "NaN": None, "nan": None})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_one_csv(path: str) -> pd.DataFrame:
    """Read pre-featured CSVs from 'featurecompany': keep OHLCV and feature columns; no additional feature engineering."""
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "Date" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} is missing column: Date")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")

    need = ["Open", "High", "Low", "Close", "Volume"]
    df = _coerce_numeric(df, need)
    df = df.dropna(subset=need)
    for c in need:
        if c in df.columns:
            df = df[df[c] > 0]

    if USE_WEEKLY:
        df = (
            df.resample("W")
              .agg({"Open":"first", "High":"max", "Low":"min", "Close":"last", "Volume":"sum"})
              .dropna()
        )

    return df


def split_train_test(data: pd.DataFrame, ratio: float):
    idx = max(1, int(len(data) * ratio))
    return data.iloc[:idx], data.iloc[idx:]


# ========== Backtesting wrapper ==========
def optimize_on_train(train_df: pd.DataFrame):
    """Keep the interface; this script uses grid search handled by grid_evaluate."""
    return 10, 20, None

def run_on_test(test_df: pd.DataFrame, n_fast: int, n_slow: int):
    # Create a temporary strategy class
    class TempStrategy(SmaCross):
        pass
    
    # Set parameters
    TempStrategy.n_fast = n_fast
    TempStrategy.n_slow = n_slow
    
    bt = Backtest(test_df, TempStrategy, cash=CASH, commission=COMMISSION, exclusive_orders=True)
    return bt.run()

def run_buyhold(test_df: pd.DataFrame):
    bt = Backtest(test_df, BuyHold, cash=CASH, commission=COMMISSION, exclusive_orders=True)
    return bt.run()

def yearly_eval(test_df: pd.DataFrame, n_fast: int, n_slow: int):
    class TempStrategy(SmaCross):
        pass
    
    # Set parameters
    TempStrategy.n_fast = n_fast
    TempStrategy.n_slow = n_slow
    
    rows = []
    for y in sorted({d.year for d in test_df.index}):
        seg = test_df[test_df.index.year == y]
        if len(seg) < 100:
            continue
        bt = Backtest(seg, TempStrategy, cash=CASH, commission=COMMISSION, exclusive_orders=True)
        s = bt.run()
        rows.append({
            "year": y,
            "Return[%]": s["Return [%]"],
            "Sharpe": s["Sharpe Ratio"],
            "MaxDD[%]": s["Max. Drawdown [%]"],
            "Trades": s["# Trades"]
        })
    return pd.DataFrame(rows)


def grid_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """Train/test all fast < slow combinations in PERIODS and return the best and the full grid records."""
    best_fast, best_slow = 10, 20
    best_sharpe = -1e9
    grid_rows = []

    for i, fast in enumerate(PERIODS[:-1]):
        for slow in PERIODS[i+1:]:
            class TempStrategy(SmaCross):
                pass
            TempStrategy.n_fast = fast
            TempStrategy.n_slow = slow

            bt_tr = Backtest(train_df, TempStrategy, cash=CASH, commission=COMMISSION, exclusive_orders=True)
            st_tr = bt_tr.run()

            bt_te = Backtest(test_df, TempStrategy, cash=CASH, commission=COMMISSION, exclusive_orders=True)
            st_te = bt_te.run()

            grid_rows.append({
                "Ticker": ticker,
                "n_fast": fast,
                "n_slow": slow,
                "Train_Return[%]": st_tr.get("Return [%]", float("nan")),
                "Train_Sharpe": st_tr.get("Sharpe Ratio", float("nan")),
                "Train_MaxDD[%]": st_tr.get("Max. Drawdown [%]", float("nan")),
                "Train_Trades": st_tr.get("# Trades", float("nan")),
                "Test_Return[%]": st_te.get("Return [%]", float("nan")),
                "Test_Sharpe": st_te.get("Sharpe Ratio", float("nan")),
                "Test_MaxDD[%]": st_te.get("Max. Drawdown [%]", float("nan")),
                "Test_Trades": st_te.get("# Trades", float("nan")),
            })

            if st_tr.get("Sharpe Ratio", -1e9) > best_sharpe:
                best_sharpe = st_tr.get("Sharpe Ratio")
                best_fast, best_slow = fast, slow

    return best_fast, best_slow, grid_rows


# ========== Main workflow ==========
def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}/.")

    # Test mode: only process the first 3 files
    if TEST_MODE:
        files = files[:3]
        print(f"[Test Mode] Only processing the first {len(files)} files")

    summary_rows, yearly_rows = [], []
    all_grid_rows = []
    best_by_sharpe_rows, best_by_return_rows = [], []
    yearly_sharpe_rows, yearly_return_rows = [], []

    for f in files:
        # Clean ticker from filename: take substring before first underscore and uppercase
        name = os.path.splitext(os.path.basename(f))[0]
        ticker = name.split("_", 1)[0].upper()

        # Skip common index file names (e.g., ^GSPC)
        if ticker.startswith("^"):
            print(f"[SKIP] {ticker}: Skipping index files")
            continue

        try:
            data = load_one_csv(f)

            # After cleaning, check length again
            if len(data) < 200:
                print(f"[SKIP] {ticker}: Too few rows after cleaning ({len(data)})")
                continue

            train, test = split_train_test(data, TRAIN_RATIO)

            # 1) Parameter grid: get best and full combinations
            best_fast, best_slow, grid_rows = grid_evaluate(train, test, ticker)
            all_grid_rows.extend(grid_rows)

            # Derive best by Sharpe and by Return for this ticker from its grid rows
            import math
            import numpy as np
            grid_this = [r for r in grid_rows if r["Ticker"] == ticker]
            if grid_this:
                # Best by Sharpe
                row_sharpe = max(grid_this, key=lambda r: (r.get("Test_Sharpe", -math.inf)))
                best_by_sharpe_rows.append({
                    "Ticker": ticker,
                    "n_fast": row_sharpe["n_fast"],
                    "n_slow": row_sharpe["n_slow"],
                    "Test_Sharpe": row_sharpe.get("Test_Sharpe"),
                    "Test_Return[%]": row_sharpe.get("Test_Return[%]"),
                    "Test_MaxDD[%]": row_sharpe.get("Test_MaxDD[%]"),
                })
                # Best by Return
                row_return = max(grid_this, key=lambda r: (r.get("Test_Return[%]", -math.inf)))
                best_by_return_rows.append({
                    "Ticker": ticker,
                    "n_fast": row_return["n_fast"],
                    "n_slow": row_return["n_slow"],
                    "Test_Sharpe": row_return.get("Test_Sharpe"),
                    "Test_Return[%]": row_return.get("Test_Return[%]"),
                    "Test_MaxDD[%]": row_return.get("Test_MaxDD[%]"),
                })

                # Yearly on full data using both best params
                yf_s = yearly_eval(data, row_sharpe["n_fast"], row_sharpe["n_slow"]).assign(Ticker=ticker, SelMetric="Sharpe")
                yf_r = yearly_eval(data, row_return["n_fast"], row_return["n_slow"]).assign(Ticker=ticker, SelMetric="Return")
                # Ensure full year coverage 2000-2024 (fill missing years with NaNs)
                for y in range(2000, 2025):
                    if y not in set(yf_s["year"]) if not yf_s.empty else set():
                        yearly_sharpe_rows.append({"Ticker": ticker, "year": y, "Return[%]": float("nan"), "Sharpe": float("nan"), "MaxDD[%]": float("nan"), "Trades": float("nan"), "SelMetric": "Sharpe"})
                    if y not in set(yf_r["year"]) if not yf_r.empty else set():
                        yearly_return_rows.append({"Ticker": ticker, "year": y, "Return[%]": float("nan"), "Sharpe": float("nan"), "MaxDD[%]": float("nan"), "Trades": float("nan"), "SelMetric": "Return"})
                if not yf_s.empty:
                    yearly_sharpe_rows.extend(yf_s.to_dict("records"))
                if not yf_r.empty:
                    yearly_return_rows.extend(yf_r.to_dict("records"))

            # 2) Test-set evaluation (with best params)
            test_stats = run_on_test(test, best_fast, best_slow)
            bh_stats   = run_buyhold(test)

            # Summary output
            summary_rows.append({
                "Ticker": ticker,
                "Freq": "W" if USE_WEEKLY else "D",
                "TrainStart": train.index[0].date(),
                "TrainEnd": train.index[-1].date(),
                "TestStart": test.index[0].date(),
                "TestEnd": test.index[-1].date(),
                "BestFast": best_fast,
                "BestSlow": best_slow,
                "Train_Sharpe": float("nan"),
                "Train_Return[%]": float("nan"),
                "Test_Return[%]": test_stats["Return [%]"],
                "Test_Sharpe": test_stats["Sharpe Ratio"],
                "Test_MaxDD[%]": test_stats["Max. Drawdown [%]"],
                "Test_Trades": test_stats["# Trades"],
                "BH_Return[%]": bh_stats["Return [%]"],
                "BH_Sharpe": bh_stats["Sharpe Ratio"],
                "BH_MaxDD[%]": bh_stats["Max. Drawdown [%]"],
                "Alpha_vs_BH[%]": test_stats["Return [%]"] - bh_stats["Return [%]"]  # Excess return
            })

            # 3) Optional: yearly performance on the test set
            ydf = yearly_eval(test, best_fast, best_slow)
            if not ydf.empty:
                ydf.insert(0, "Ticker", ticker)
                yearly_rows.append(ydf)

            print(f"[OK] {ticker}: best=({best_fast},{best_slow})  Test Sharpe={test_stats['Sharpe Ratio']:.2f}")

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    # Write CSVs (saved to current working directory)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(["Ticker"]).to_csv(OUT_SUMMARY, index=False)
        print(f"\nSummary written: {OUT_SUMMARY}")

    if yearly_rows:
        pd.concat(yearly_rows, ignore_index=True).to_csv(OUT_YEARLY, index=False)
        print(f"Yearly performance written: {OUT_YEARLY}")

    # Write full grid results and do sector/parameter aggregations
    if all_grid_rows:
        grid_df = pd.DataFrame(all_grid_rows)
        # Clean Ticker column defensively as requested
        grid_df["Ticker"] = grid_df["Ticker"].astype(str).str.split("_", n=1, expand=True)[0].str.upper()
        grid_df.to_csv(OUT_GRID, index=False)

        # Sector aggregation: map Ticker -> Sector
        grid_df["Sector"] = grid_df["Ticker"].map(SECTOR_MAP).fillna("Other")
        sector_agg = (
            grid_df.groupby("Sector")[
                ["Test_Return[%]", "Test_Sharpe", "Test_MaxDD[%]", "Test_Trades"]
            ]
            .mean()
            .reset_index()
            .sort_values("Test_Sharpe", ascending=False)
        )
        sector_agg.to_csv(OUT_SECTOR, index=False)
        print(f"Sector summary written: {OUT_SECTOR}")

        # Parameter aggregation: average effects for different fast/slow time frames
        param_agg = (
            grid_df.groupby(["n_fast", "n_slow"])[
                ["Test_Return[%]", "Test_Sharpe", "Test_MaxDD[%]", "Test_Trades"]
            ]
            .mean()
            .reset_index()
            .sort_values("Test_Sharpe", ascending=False)
        )
        param_agg.to_csv(OUT_PARAM, index=False)
        print(f"Parameter summary written: {OUT_PARAM}")

    # Write best-by-Sharpe and best-by-Return tables
    if best_by_sharpe_rows:
        pd.DataFrame(best_by_sharpe_rows).sort_values(["Ticker"]).to_csv(OUT_BEST_SHARPE, index=False)
        print(f"Best-by-Sharpe results written: {OUT_BEST_SHARPE}")
    if best_by_return_rows:
        pd.DataFrame(best_by_return_rows).sort_values(["Ticker"]).to_csv(OUT_BEST_RETURN, index=False)
        print(f"Best-by-Return results written: {OUT_BEST_RETURN}")

    # Write yearly tables for both selection metrics
    if yearly_sharpe_rows:
        pd.DataFrame(yearly_sharpe_rows).sort_values(["Ticker","year"]).to_csv(OUT_YEARLY_SHARPE, index=False)
        print(f"Yearly (Sharpe selection) performance written: {OUT_YEARLY_SHARPE}")
    if yearly_return_rows:
        pd.DataFrame(yearly_return_rows).sort_values(["Ticker","year"]).to_csv(OUT_YEARLY_RETURN, index=False)
        print(f"Yearly (Return selection) performance written: {OUT_YEARLY_RETURN}")

    # Produce a simple narrative text summary comparing best metrics
    try:
        lines = []
        if best_by_sharpe_rows:
            lines.append("Best by Sharpe per Ticker:\n")
            for r in sorted(best_by_sharpe_rows, key=lambda x: x["Ticker"]):
                lines.append(f"- {r['Ticker']}: fast={r['n_fast']}, slow={r['n_slow']}, Sharpe={r['Test_Sharpe']:.2f}, Return={r['Test_Return[%]']:.2f}%")
            lines.append("")
        if best_by_return_rows:
            lines.append("Best by Return per Ticker:\n")
            for r in sorted(best_by_return_rows, key=lambda x: x["Ticker"]):
                lines.append(f"- {r['Ticker']}: fast={r['n_fast']}, slow={r['n_slow']}, Return={r['Test_Return[%]']:.2f}%, Sharpe={r['Test_Sharpe']:.2f}")
            lines.append("")
        if yearly_sharpe_rows:
            lines.append("Top yearly performers (Sharpe selection):\n")
            ydf = pd.DataFrame(yearly_sharpe_rows).dropna(subset=["Sharpe"])
            if not ydf.empty:
                topy = ydf.sort_values(["Sharpe"], ascending=False).head(10)
                for _, r in topy.iterrows():
                    lines.append(f"- {r['Ticker']} {int(r['year'])}: Sharpe={r['Sharpe']:.2f}, Return={r['Return[%]']:.2f}%")
                lines.append("")
        if yearly_return_rows:
            lines.append("Top yearly performers (Return selection):\n")
            ydf2 = pd.DataFrame(yearly_return_rows).dropna(subset=["Return[%]"])
            if not ydf2.empty:
                topy2 = ydf2.sort_values(["Return[%]"] , ascending=False).head(10)
                for _, r in topy2.iterrows():
                    lines.append(f"- {r['Ticker']} {int(r['year'])}: Return={r['Return[%]']:.2f}%, Sharpe={r['Sharpe']:.2f}")
        if lines:
            with open(OUT_TEXT_SUMMARY, "w") as fh:
                fh.write("\n".join(lines))
            print(f"Analysis text written: {OUT_TEXT_SUMMARY}")
    except Exception as _:
        pass

if __name__ == "__main__":
    main()
