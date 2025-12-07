import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import os
import warnings
import glob
import traceback
import sys
import importlib

warnings.filterwarnings('ignore')

# =========================================================
# GLOBAL COMPATIBILITY PATCH (The "Nuclear" Option)
# =========================================================
def force_numpy_patch():
    try:
        # If current environment is numpy 1.x (no _core) and has core
        if not hasattr(np, '_core') and hasattr(np, 'core'):
            sys.modules['numpy._core'] = np.core
            for submodule in ['numeric', 'multiarray', 'umath', 'overrides', 'defchararray', 'records', 'memmap', 'varargs']:
                if hasattr(np.core, submodule):
                    real_mod = getattr(np.core, submodule)
                    sys.modules[f'numpy._core.{submodule}'] = real_mod
            print("【System】NumPy Compatibility Patch Applied Successfully.")
    except Exception as e:
        print(f"【System】Patch warning: {e}")

force_numpy_patch()

# =========================================================
# 1. Configuration & Sector Mapping
# =========================================================
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AVGO": "Technology",
    "CSCO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "INTC": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "AXP": "Financials", "SCHW": "Financials",
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "BA": "Industrials", "CAT": "Industrials", "UPS": "Industrials",
    "RTX": "Industrials", "HON": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "DIS": "Communication Services", "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    "NEE": "Utilities", "DUK": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate",
}

# =========================================================
# INTELLIGENT PATH FINDER
# =========================================================
def find_resource_path(filename, search_paths=None):
    """
    Tries to find a file in multiple likely locations.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of the script
    cwd = os.getcwd() # Current working directory
    
    # 2. Possible relative subdirectories
    subdirs = ['', 'models', 'clean_out', 'data']
    
    # 3. Possible parent directories
    parents = [base_dir, cwd, os.path.join(base_dir, '..'), os.path.join(cwd, '..')]
    
    checked_paths = []
    
    # Generate all combinations
    for parent in parents:
        for subdir in subdirs:
            full_path = os.path.normpath(os.path.join(parent, subdir, filename))
            if full_path not in checked_paths:
                checked_paths.append(full_path)
                if os.path.exists(full_path):
                    return full_path, []
    
    # If not found, return None and the list of checked paths
    return None, checked_paths

# Define Model Filenames (Just filenames, not full paths)
MODEL_FILES = {
    'ml1': 'ppo_fast_multi_stock.zip',
    'ml2_bull': 'ppo_regime_bull.zip',
    'ml2_bear': 'ppo_regime_bear.zip',
    'ml2_sideways': 'ppo_regime_sideways.zip',
}

SECTOR_MODEL_FILES = {
    'Technology': 'ppo_sector_Technology.zip',
    'Financials': 'ppo_sector_Financials.zip',
    'Healthcare': 'ppo_sector_Healthcare.zip',
    'Consumer Discretionary': 'ppo_sector_Consumer_Discretionary.zip',
    'Consumer Staples': 'ppo_sector_Consumer_Staples.zip',
    'Energy': 'ppo_sector_Energy.zip',
    'Utilities': 'ppo_sector_Utilities.zip',
    'Industrials': 'ppo_sector_Industrials.zip',
    'Communication Services': 'ppo_sector_Communication_Services.zip',
    'Real Estate': 'ppo_sector_Real_Estate.zip'
}

# =========================================================
# RL FEATURE ENGINEERING
# =========================================================
RL_FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d",
    "price_vs_ma5", "price_vs_ma20",
    "trend_5_20", "trend_20_50",
    "volatility",
    "bollinger_position",
    "strength_rsi_proxy",
]

def add_trading_signals_for_rl(df):
    """
    Adds 10 RL features to the original DataFrame (must have 'Close').
    Returns a new DataFrame.
    """
    df = df.copy()

    close = df["Close"]

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    df["price_vs_ma5"] = close / ma_5 - 1
    df["price_vs_ma20"] = close / ma_20 - 1

    df["trend_5_20"] = (ma_5 - ma_20) / (ma_20 + 1e-8)
    df["trend_20_50"] = (ma_20 - ma_50) / (ma_50 + 1e-8)

    df["volatility"] = close.pct_change().rolling(20).std()

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    df["bollinger_position"] = (close - rolling_mean) / (rolling_std + 1e-8)

    df["strength_rsi_proxy"] = close.rolling(14).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8),
        raw=False
    )

    return df.fillna(0)

# =========================================================
# 2. Helper Functions
# =========================================================
def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

def RSI(values, n=14):
    delta = pd.Series(values).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

def BBANDS(values, n=20, k=2):
    ma = pd.Series(values).rolling(n).mean()
    std = pd.Series(values).rolling(n).std()
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower

def MOMENTUM(values, n):
    return (pd.Series(values) / pd.Series(values).shift(n)) - 1

def calculate_var(equity_curve):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0: return 0
    return returns.quantile(0.05)

def calculate_benchmark_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0: return 0, 0, 0, 0, 0
    
    var_95 = returns.quantile(0.05)
    annual_return = returns.mean() * 252
    annual_std = returns.std() * np.sqrt(252)
    sharpe = (annual_return / annual_std) if annual_std != 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return / downside_std) if downside_std != 0 else 0
    
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = drawdown.min() * 100
    
    win_rate = (len(returns[returns > 0]) / len(returns)) * 100
    
    return sharpe, sortino, max_dd, win_rate, var_95

# =========================================================
# 3. Strategy Classes
# =========================================================

class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    def next(self):
        if crossover(self.sma1, self.sma2): 
            self.buy(size=0.95)
        elif crossover(self.sma2, self.sma1): 
            self.position.close()

class BollingerBandStrategy(Strategy):
    n = 20
    k = 2.0
    def init(self):
        self.mid, self.upper, self.lower = self.I(BBANDS, self.data.Close, self.n, self.k)
    def next(self):
        price = self.data.Close[-1]
        if price < self.lower[-1] and not self.position: 
            self.buy(size=0.95)
        elif price > self.mid[-1] and self.position: 
            self.position.close()

class MomentumStrategy(Strategy):
    lookback = 20
    threshold = 0.0 
    def init(self):
        self.mom = self.I(MOMENTUM, self.data.Close, self.lookback)
    def next(self):
        if self.mom[-1] > self.threshold and not self.position: 
            self.buy(size=0.95)
        elif self.mom[-1] < 0 and self.position: 
            self.position.close()

#
# UPDATED PPOStrategy
#
class PPOStrategy(Strategy):
    """
    Universal PPO Strategy:
    - Pass model_instance via class attributes (Backtest.run(model_instance=...))
    - Use class attribute 'action_mode' to determine action interpretation:
        * "5actions": ML1 (0=Hold, 1=Buy20%, 2=Buy40%, 3=Sell, 4=Close)
        * "3actions": Regime / Sector models (0=Hold, 1=BuyAll, 2=SellAll)
    """
    model_instance = None
    action_mode = "5actions"   # Default for ML1

    def init(self):
        if self.model_instance is None:
            raise ValueError(
                "PPOStrategy.model_instance must be set via Backtest.run(model_instance=...)"
            )

        self.model = self.model_instance
        self.window = 10

        # Read PPO model observation dimension
        self.obs_dim = int(self.model.observation_space.shape[0])

        # Get all columns from underlying DataFrame
        df = self.data.df if hasattr(self.data, "df") else self.data._data

        # ✅ Prioritize using defined RL_FEATURE_COLS (same as training)
        available = [c for c in RL_FEATURE_COLS if c in df.columns]

        if len(available) == 0:
            # Fallback: Use all columns except Date/Close/market_regime
            self.feature_cols = [
                c for c in df.columns
                if c not in ["Date", "Close", "market_regime"]
            ]
            print(f"[PPOStrategy] WARNING: RL_FEATURE_COLS not found, fallback to all columns ({len(self.feature_cols)})")
        else:
            self.feature_cols = available

        # Variables used in Regime/Sector environments
        self.days_since_trade = 0     # For ML1 idle logic
        self.hold_days = 0            # For ML2/ML3 hold_days logic
        self.last_action = 0          # For ML2/ML3 last_action logic
        self.initial_equity = float(self.equity) 

    # ----------------- Trading Logic -----------------
    def next(self):
        current_idx = len(self.data.Close) - 1
        if current_idx < self.window:
            return

        features = self._extract_features(current_idx)
        if features is None:
            self.days_since_trade += 1
            self.hold_days += 1
            return

        action = self._get_action(features)

        price = self.data.Close[-1]
        commission_rate = 0.001
        position_value = self.position.size * price if self.position else 0
        cash = self.equity - position_value

        mode = getattr(self.__class__, "action_mode", "5actions")

        # ========= Mode 1: 3 Actions (Regime / Sector Env) =========
        if mode == "3actions":
            # 0: hold
            if action == 0:
                if self.position:
                    self.hold_days += 1

            # 1: all-in buy
            elif action == 1:
                if not self.position:
                    self.buy(size=0.99)
                    self.days_since_trade = 0
                    self.hold_days = 0

            # 2: sell all
            elif action == 2:
                if self.position:
                    self.position.close()
                    self.days_since_trade = 0
                    self.hold_days = 0

        # ========= Mode 2: 5 Actions (Original ML1 PPO) =========
        else:
            # 0: hold
            if action == 0:
                if self.position:
                    self.hold_days += 1

            # 1: Buy 20%
            elif action == 1:
                target_value = self.equity * 0.20
                max_shares = int(cash / (price * (1 + commission_rate)))
                target_shares = int(target_value / price)
                shares = min(max_shares, target_shares)
                if shares > 0:
                    self.buy(size=shares)
                    self.days_since_trade = 0
                    self.hold_days = 0

            # 2: Buy 40%
            elif action == 2:
                target_value = self.equity * 0.40
                max_shares = int(cash / (price * (1 + commission_rate)))
                target_shares = int(target_value / price)
                shares = min(max_shares, target_shares)
                if shares > 0:
                    self.buy(size=shares)
                    self.days_since_trade = 0
                    self.hold_days = 0

            # 3 / 4: Sell / Close
            elif action in (3, 4):
                if self.position:
                    self.position.close()
                    self.days_since_trade = 0
                    self.hold_days = 0

        self.days_since_trade += 1
        self.last_action = int(action)

    # ----------------- Feature Extraction -----------------
    def _extract_features(self, idx):
        try:
            df = self.data.df if hasattr(self.data, "df") else self.data._data
            start = max(0, idx - self.window + 1)

            # Get features within window
            hist = df.iloc[start:idx+1][self.feature_cols].values

            # Pad if window is short
            if len(hist) < self.window:
                pad = np.tile(hist[0], (self.window - len(hist), 1))
                hist = np.vstack([pad, hist])
            else:
                hist = hist[-self.window:]

            feat = hist.flatten()

            price = self.data.Close[-1]
            position_value = self.position.size * price if self.position else 0
            cash = self.equity - position_value
            total_equity = max(self.equity, 1e-8)
            initial_eq = max(self.initial_equity, 1e-8)

            mode = getattr(self.__class__, "action_mode", "5actions")

            # Regime / Sector Models: Closer to RegimeSpecificEnv
            if mode == "3actions":
                cash_ratio0 = cash / initial_eq
                pos_ratio0 = position_value / initial_eq
                last_act = float(self.last_action)
                hold_norm = min(self.hold_days, 100) / 100.0

                extra = np.array(
                    [cash_ratio0, pos_ratio0, last_act, hold_norm],
                    dtype=np.float32
                )

            # ML1: Original 3 state variables
            else:
                cash_ratio = cash / total_equity
                pos_ratio = position_value / total_equity
                idle = min(self.days_since_trade / 10.0, 1.0)

                extra = np.array(
                    [cash_ratio, pos_ratio, idle],
                    dtype=np.float32
                )

            feat = np.concatenate([feat, extra])

            # Align obs_dim
            if len(feat) < self.obs_dim:
                feat = np.pad(feat, (0, self.obs_dim - len(feat)))
            elif len(feat) > self.obs_dim:
                feat = feat[:self.obs_dim]

            return feat.astype(np.float32)

        except Exception as e:
            return None

    # ----------------- Prediction -----------------
    def _get_action(self, features):
        try:
            obs = features.reshape(1, -1)
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        except Exception as e:
            return 0


# =========================================================
# 4. Data Loading (Auto-Scanning)
# =========================================================

def get_stock_data(ticker, start, end):
    """
    Returns a DataFrame:
    - Contains Open / High / Low / Close / Volume
    - If local *_features.csv exists, loads it (for ML strategies)
    - Applies 'add_trading_signals_for_rl' to ensure features exist
    """
    data = pd.DataFrame()

    # 1. Prioritize finding training script style *_features.csv
    for fname in [f"{ticker}_features.csv", f"{ticker}_features_deep_clean.csv"]:
        file_path, _ = find_resource_path(fname)
        if file_path and os.path.exists(file_path):
            try:
                # print(f"Loading local feature file: {file_path}")
                data = pd.read_csv(file_path)
                break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                data = pd.DataFrame()

    # 2. If local not found, fallback to yfinance
    if data.empty:
        try:
            print("Falling back to yfinance...")
            yf_data = yf.download(ticker, start=start, end=end, progress=False)
            if not yf_data.empty:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_data.columns = yf_data.columns.get_level_values(0)
                data = yf_data
        except Exception as e:
            print("yfinance error:", e)
            return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # 3. Process dates & slice time period
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
        data.set_index("Date", inplace=True)
    elif "timestamp" in data.columns:
        data["Date"] = pd.to_datetime(data["timestamp"])
        data.set_index("Date", inplace=True)

    # Filter by date
    try:
        mask = (data.index >= pd.Timestamp(start)) & (data.index <= pd.Timestamp(end))
        data = data.loc[mask]
    except Exception as e:
        print("Date filter error:", e)

    if data.empty:
        return pd.DataFrame()

    # 4. Ensure OHLCV columns exist
    req_cols = ["Open", "High", "Low", "Close", "Volume"]
    col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    data.rename(columns=col_map, inplace=True)

    for col in req_cols:
        if col not in data.columns and "Close" in data.columns:
            data[col] = data["Close"]

    try:
        # Only drop rows where Close is NaN
        data = data.dropna(subset=["Close"])
    except KeyError:
        return pd.DataFrame()

    # ⭐ Apply RL Feature Engineering to all data (Crucial for ML) ⭐
    data = add_trading_signals_for_rl(data)

    return data

def load_ppo_model(strategy_name, ticker):
    filename = ""
    # UPDATED: Matches the new 1-7 Strategy Indexing
    if strategy_name == "5. ML: PPO Complex (ML1)":
        filename = MODEL_FILES['ml1']
    elif strategy_name == "6. ML: PPO Regime Aware (ML2)":
        filename = MODEL_FILES['ml2_bull']
    elif strategy_name == "7. ML: PPO Sector (ML3)":
        sector = SECTOR_MAP.get(ticker)
        filename = SECTOR_MODEL_FILES.get(sector, MODEL_FILES['ml1'])
    else:
        return None, "Not an ML strategy"

    # Use Intelligent Finder
    result, checked_paths = find_resource_path(filename)
    
    if result is None:
        # Format the checked paths nicely for display
        debug_msg = "\n".join([f"- {p}" for p in checked_paths])
        return None, f"FILE NOT FOUND: {filename}\nSearched in:\n{debug_msg}"
    else:
        print(f"Loading model from: {result}")
        try:
            return PPO.load(result), None
        except Exception as e:
            # Final line of defense
            err_str = str(e)
            if "numpy._core" in err_str:
                print(f"Caught pickle error: {err_str}. Retrying with dynamic injection...")
                try:
                    force_numpy_patch()
                    return PPO.load(result), None
                except Exception as e2:
                    return None, f"LOAD ERROR (Retry Failed):\n{str(e2)}"
            
            return None, f"LOAD ERROR ({type(e).__name__}):\n{err_str}"

# Init Variables (Scan for tickers)
sample_file, _ = find_resource_path("AAPL_features_deep_clean.csv")
data_dir_to_scan = ""

if sample_file:
    data_dir_to_scan = os.path.dirname(sample_file)
else:
    # Try just standard locations
    base = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(base, "clean_out")):
        data_dir_to_scan = os.path.join(base, "clean_out")

if data_dir_to_scan and os.path.exists(data_dir_to_scan):
    available_files = glob.glob(os.path.join(data_dir_to_scan, "*_features_deep_clean.csv"))
    scanned_tickers = [os.path.basename(f).split('_')[0] for f in available_files]
    all_tickers = sorted(list(set(scanned_tickers)))
else:
    all_tickers = sorted([k for k in SECTOR_MAP.keys()])

min_date = '2000-01-01'
max_date = pd.Timestamp.today().strftime('%Y-%m-%d')

# =========================================================
# RE-INDEXED STRATEGIES (1-7)
# =========================================================
STRATEGIES = {
    "1. Benchmark: Buy & Hold": None,
    "2. Traditional: SMA Crossover": SmaCross,
    "3. Traditional: Bollinger Bands": BollingerBandStrategy,
    "4. Traditional: Absolute Momentum": MomentumStrategy,
    "5. ML: PPO Complex (ML1)": PPOStrategy,
    "6. ML: PPO Regime Aware (ML2)": PPOStrategy,
    "7. ML: PPO Sector (ML3)": PPOStrategy,
}

# =========================================================
# 5. Dash App Setup
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.SLATE, 
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
], suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Team 8 - Trading Analytics</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #000; }
            .card { border: 1px solid #333; background-color: #222; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .card-header { background-color: #2c2c2c; border-bottom: 1px solid #444; font-weight: 600; color: #e0e0e0; letter-spacing: 0.5px; }
            .metric-card { background: linear-gradient(145deg, #2a2a2a, #222); border-radius: 8px; border: 1px solid #333; }
            .metric-val { font-family: 'Roboto Mono', monospace; font-weight: 700; }
            .nav-tabs .nav-link { color: #aaa; border: none; }
            .nav-tabs .nav-link.active { background-color: #222; border-color: #333; border-bottom-color: #222; color: #00bc8c; font-weight: bold; }
            .range-btn { font-size: 0.8rem; font-weight: 600; margin-right: 5px; border-radius: 20px; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #1a1a1a; }
            ::-webkit-scrollbar-thumb { background: #444; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #555; }
            .text-bright-green { color: #00ff00 !important; }
            .text-light-purple { color: #E0B0FF !important; }
            
            /* Introduction Page Styles */
            .intro-hero { padding: 4rem 2rem; background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,188,140,0.05) 100%); border-bottom: 1px solid #333; }
            .intro-section { padding: 3rem 0; }
            .feature-icon { font-size: 2.5rem; color: #00bc8c; margin-bottom: 1rem; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def InfoIcon(id_name, text):
    return html.Span([
        html.I(className="fas fa-info-circle", id=id_name, 
               style={"cursor": "pointer", "color": "#17a2b8", "marginLeft": "8px", "fontSize": "0.9rem"}),
        dbc.Tooltip(text, target=id_name, placement="top", style={"maxWidth": "300px"})
    ])

def MetricCard(title, id_name, color_class="text-white"):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted text-uppercase", 
                        style={"fontSize": "0.75rem", "letterSpacing": "1px"}),
                html.H3("0.00", id=id_name, className=f"metric-val {color_class} mb-0")
            ], className="p-3")
        ], className="metric-card h-100"), 
        width=6, lg=2, className="mb-2"
    )

# =========================================================
# 6. Layouts (Intro & Dashboard)
# =========================================================

# --- Navigation Bar ---
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard", active="exact")),
    ],
    brand="QuantPro Analytics",
    brand_href="/",
    color="#1a1a1a",
    dark=True,
    className="border-bottom border-secondary"
)

# --- Introduction Page Layout ---
layout_intro = html.Div([
    # Hero Section
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Can AI Outsmart Wall Street?", className="display-4 fw-bold text-white mb-4"),
                    html.P([
                        "Retail trading has exploded in popularity, with millions of individual investors now competing against institutional and corporate algorithms. ",
                        "Many are drawn to AI powered trading bots that promise easy returns, but overfitting causes these tools to fail in the real world. ",
                    ], className="lead text-light"),
                    html.P([
                        "Our project addresses a critical question for modern investors: ",
                        html.Span("Can machine learning models provide a better risk-adjusted return than traditional trading methods?", className="text-success fw-bold")
                    ], className="lead text-light"),
                    
                    html.P(
                        "We invite you to explore the Interactive Dashboard to test these strategies yourself.",
                        className="text-light fst-italic mt-4 mb-3"
                    ),
                    
                    dbc.Button("Launch Dashboard", href="/dashboard", color="success", size="lg", className="rounded-pill px-5")
                ], width=12, lg=10, className="mx-auto text-center")
            ])
        ])
    ], className="intro-hero"),

    # Main Content Section
    dbc.Container([
        # Project Overview
        dbc.Row([
            dbc.Col([
                html.H3("Project Overview", className="text-info border-bottom border-secondary pb-2 mb-4"),
                html.P([
                    "Our STA 160 Capstone Project explores this question by analyzing twenty-five years of daily stock data from fifty-one major stocks across ",
                    "8 different economic sectors (including the S&P 500 index as a benchmark). Our goal is to compare traditional trading strategies with ",
                    "several machine learning and deep learning models to determine which approach performs the best in market conditions. ",
                    "This is important in helping investors make more informed decisions."
                ], className="text-light"),
            ], width=12, className="mb-5")
        ]),

        # Methodology
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-balance-scale feature-icon")),
                        html.H4("Fair Comparison", className="card-title text-white"),
                        html.P("To ensure a fair competition, we built a standardized testing environment that treats every strategy equally. We didn’t simply look at who made the most money; we looked at risk, consistency, and performance during downturns.", className="card-text text-muted")
                    ])
                ], className="h-100 bg-dark border-secondary")
            ], width=12, md=4, className="mb-4"),
             dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-chart-line feature-icon")),
                        html.H4("Traditional Strategies", className="card-title text-white"),
                        html.P("For traditional statistical strategies, we chose Momentum, SMA Crossover, and Bollinger Bands. These serve as the baseline for proven technical analysis.", className="card-text text-muted")
                    ])
                ], className="h-100 bg-dark border-secondary")
            ], width=12, md=4, className="mb-4"),
             dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(html.I(className="fas fa-brain feature-icon")),
                        html.H4("AI & Machine Learning", className="card-title text-white"),
                        html.P("We created advanced models including Transformers (Deep Learning) and PPO (Reinforcement Learning). As a benchmark, we used Walk-Forward Validation: training on past data and trading on 'future' unseen data.", className="card-text text-muted")
                    ])
                ], className="h-100 bg-dark border-secondary")
            ], width=12, md=4, className="mb-4"),
        ]),

        # Findings
        dbc.Row([
            dbc.Col([
                html.H3("Key Findings", className="text-warning border-bottom border-secondary pb-2 mb-4 mt-4"),
                html.Ul([
                    html.Li("Simple strategies like Momentum were surprisingly hard to beat during Bull Markets (like 2010-2019), since they captured the big moves without overthinking.", className="text-light mb-3"),
                    html.Li("Our initial neural networks were too conservative. We pivoted to Reinforcement Learning (PPO) to teach active trading behaviors.", className="text-light mb-3"),
                    html.Li([
                        "Our best results came from ",
                        html.Span("Sector-Specific PPO agents", className="text-success fw-bold"),
                        ". By training separate models for 'Tech' vs 'Utilities,' the model learned to adapt its aggression levels to the volatility of each sector, outperforming generic models."
                    ], className="text-light mb-3"),
                ])
            ], width=12)
        ]),
        
    ], className="intro-section")
])

# --- Dashboard Layout ---
control_panel = dbc.Card([
    dbc.CardHeader([html.I(className="fas fa-sliders-h me-2"), "Strategy Config"]),
    dbc.CardBody([
        html.Label("Asset Ticker", className="text-light fw-bold mt-2"),
        dcc.Dropdown(
            id="ticker-drop", 
            options=[{"label": f"{t} ({SECTOR_MAP.get(t, 'N/A')})", "value": t} for t in all_tickers],
            value=all_tickers[0] if all_tickers else None, 
            clearable=False, 
            className="mb-3", 
            style={"color": "#000"}
        ),
        
        html.Label("Strategy Model", className="text-light fw-bold"),
        dcc.Dropdown(
            id="strategy-drop", 
            options=[{"label": k, "value": k} for k in STRATEGIES.keys()], 
            value="2. Traditional: SMA Crossover", 
            clearable=False, 
            className="mb-4", 
            style={"color": "#000"}
        ),
        
        html.Hr(className="border-secondary"),
        
        html.Div(id="sma-params", children=[
            html.Label(["Fast MA Period", InfoIcon("i-n1", "Short-term moving average")]),
            dcc.Slider(id="n1-slider", min=5, max=50, step=1, value=10, 
                      marks={10:'10', 30:'30', 50:'50'}, 
                      tooltip={"always_visible": False, "placement": "bottom"}),
            html.Br(),
            html.Label(["Slow MA Period", InfoIcon("i-n2", "Long-term moving average")]),
            dcc.Slider(id="n2-slider", min=20, max=100, step=5, value=20, 
                      marks={20:'20', 60:'60', 100:'100'}, 
                      tooltip={"always_visible": False, "placement": "bottom"}),
        ], style={"display": "none"}),

        html.Div(id="boll-params", children=[
            html.Label(["Lookback Window", InfoIcon("i-bbn", "Period for MA Calculation")]),
            dcc.Slider(id="bb-n-slider", min=10, max=50, step=1, value=20, 
                      marks={10:'10', 30:'30', 50:'50'}, 
                      tooltip={"always_visible": False}),
            html.Br(),
            html.Label(["Std Dev Multiplier", InfoIcon("i-bbk", "Width of the bands")]),
            dcc.Slider(id="bb-k-slider", min=1.0, max=3.0, step=0.1, value=2.0, 
                      marks={1:'1', 2:'2', 3:'3'}, 
                      tooltip={"always_visible": False}),
        ], style={"display": "none"}),

        html.Div(id="mom-params", children=[
            html.Label(["Lookback Days", InfoIcon("i-lb", "Period to compare price against")]),
            dcc.Slider(id="lb-slider", min=5, max=126, step=1, value=20, 
                      marks={5:'1W', 21:'1M', 63:'3M'}, 
                      tooltip={"always_visible": False}),
        ], style={"display": "none"}),
    ])
], className="border-0 h-100")

layout_dashboard = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.H2([html.I(className="fas fa-chart-line text-success me-3"), 
                    "Active Trading Analysis"], 
                   className="text-light fw-bold mb-0"),
            html.P("Compare algorithmic performance against Buy & Hold benchmarks.", 
                   className="text-muted small mb-0")
        ]), width=12, className="py-3 border-bottom border-secondary mb-4")
    ]),

    # DEBUG CONSOLE AREA
    dbc.Row([
        dbc.Col(
            dbc.Collapse(
                dbc.Card([
                    dbc.CardHeader("⚠️ System Debug Console (Auto-detected Error)", className="bg-danger text-white fw-bold"),
                    dbc.CardBody(html.Pre(id="debug-output", className="text-danger m-0", style={"whiteSpace": "pre-wrap"}))
                ], className="border-danger mb-4"),
                id="debug-collapse",
                is_open=False
            ), width=12
        )
    ]),

    dbc.Row([
        MetricCard("Total Return", "res-return", "text-warning"),
        MetricCard("Sharpe Ratio", "res-sharpe", "text-success"),
        MetricCard("Sortino Ratio", "res-sortino", "text-info"),
        MetricCard("Max Drawdown", "res-mdd", "text-danger"),
        MetricCard("Win Rate", "res-win", "text-light-purple"),
        MetricCard("Profit Factor", "res-pf", "text-light"),
    ], className="mb-4 g-2"),

    dbc.Row([
        dbc.Col(control_panel, width=12, lg=3, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range:", className="fw-bold text-light me-2"),
                            dcc.DatePickerRange(
                                id='date-picker',
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                start_date=(pd.Timestamp(max_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d'),
                                end_date=max_date,
                                style={"fontSize": "0.8rem"},
                                className="d-inline-block"
                            )
                        ], width=12, md=6, className="mb-2 mb-md-0"),
                        dbc.Col([
                            html.Div([
                                dbc.Button("1Y", id="btn-1y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("3Y", id="btn-3y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("5Y", id="btn-5y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("10Y", id="btn-10y", color="dark", size="sm", className="range-btn border-secondary"),
                                dbc.Button("MAX", id="btn-max", color="dark", size="sm", className="range-btn border-secondary"),
                            ], className="d-flex justify-content-md-end")
                        ], width=12, md=6)
                    ], align="center")
                ], className="py-2 px-3")
            ], className="mb-3"),

            dbc.Tabs([
                dbc.Tab(label="Equity Curve Only", tab_id="tab-perf", children=[
                    dcc.Loading(dcc.Graph(id="graph-equity", style={"height": "600px"}), 
                               type="graph", color="#00bc8c")
                ]),
                dbc.Tab(label="Risk Analysis", tab_id="tab-risk", children=[
                    dcc.Loading(dcc.Graph(id="graph-risk", style={"height": "600px"}), 
                               type="graph", color="#00bc8c")
                ]),
                dbc.Tab(label="Monte Carlo Sim", tab_id="tab-monte", children=[
                    dbc.CardBody([
                        dbc.Button("Run Simulation (50 Paths)", id="btn-monte", color="primary", className="mb-3"),
                        dcc.Loading(dcc.Graph(id="graph-monte", style={"height": "550px"}), 
                                   type="graph", color="#00bc8c")
                    ])
                ]),
                dbc.Tab(label="Strategy Optimization", tab_id="tab-opt", children=[
                    dbc.CardBody([
                        dbc.Button("Run Grid Search Optimization", id="btn-opt", color="danger", className="mb-3"),
                        dcc.Loading(dcc.Graph(id="graph-heat", style={"height": "550px"}), 
                                   type="graph", color="#00bc8c")
                    ])
                ]),
            ], active_tab="tab-perf", className="nav-tabs")
        ], width=12, lg=9)
    ])
], fluid=True, className="pb-5")

# --- MAIN APP LAYOUT (Multi-Page Wrapper) ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# =========================================================
# 7. Callbacks
# =========================================================

# --- Page Routing Callback ---
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/dashboard':
        return layout_dashboard
    else:
        return layout_intro

# --- Existing Callbacks ---

@app.callback(
    [Output('date-picker', 'start_date'), Output('date-picker', 'end_date')],
    [Input('btn-1y', 'n_clicks'), Input('btn-3y', 'n_clicks'), 
     Input('btn-5y', 'n_clicks'), Input('btn-10y', 'n_clicks'), Input('btn-max', 'n_clicks')],
    [State('date-picker', 'end_date')]
)
def update_date_range(b1, b3, b5, b10, bmax, current_end):
    triggered = ctx.triggered_id
    end_dt = pd.Timestamp(max_date)
    if triggered == 'btn-1y': start_dt = end_dt - pd.DateOffset(years=1)
    elif triggered == 'btn-3y': start_dt = end_dt - pd.DateOffset(years=3)
    elif triggered == 'btn-5y': start_dt = end_dt - pd.DateOffset(years=5)
    elif triggered == 'btn-10y': start_dt = end_dt - pd.DateOffset(years=10)
    elif triggered == 'btn-max': start_dt = pd.Timestamp(min_date)
    else: return dash.no_update, dash.no_update
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

@app.callback(
    [Output("sma-params", "style"), Output("boll-params", "style"), Output("mom-params", "style")],
    Input("strategy-drop", "value")
)
def toggle_params(strategy):
    show, hide = {"display": "block"}, {"display": "none"}
    if not strategy: return hide, hide, hide
    if "SMA" in strategy: return show, hide, hide
    if "Bollinger" in strategy: return hide, show, hide
    if "Momentum" in strategy: return hide, hide, show
    return hide, hide, hide

@app.callback(
    [Output("graph-equity", "figure"), Output("graph-risk", "figure"),
     Output("res-return", "children"), Output("res-sharpe", "children"), 
     Output("res-sortino", "children"), Output("res-mdd", "children"), 
     Output("res-win", "children"), Output("res-pf", "children"),
     Output("debug-output", "children"), Output("debug-collapse", "is_open")],
    [Input("ticker-drop", "value"), Input("strategy-drop", "value"),
     Input("date-picker", "start_date"), Input("date-picker", "end_date"),
     Input("n1-slider", "value"), Input("n2-slider", "value"),
     Input("bb-n-slider", "value"), Input("bb-k-slider", "value"),
     Input("lb-slider", "value")]
)
def run_backtest(ticker, strategy_name, start, end, n1, n2, bb_n, bb_k, lb):
    if not ticker: return go.Figure(), go.Figure(), *["-"]*6, "", False
    
    data = get_stock_data(ticker, start, end)
    if data.empty or len(data) < 10:
        return go.Figure(), go.Figure(), "No Data", "N/A", "N/A", "N/A", "N/A", "N/A", "No data found for ticker", True

    StrategyClass = STRATEGIES[strategy_name]
    initial_cash = 50000 
    benchmark_equity = (data["Close"] / data["Close"].iloc[0]) * initial_cash
    
    equity = pd.Series()
    trades_df = None
    strategy_failed = False
    error_msg = "" 
    
    ret, sharpe, sortino, mdd, win_rate, pf, var = 0,0,0,0,0,0,0
    
    # Reset Action Mode Default
    if "PPO Sector" in strategy_name:
        PPOStrategy.action_mode = "3actions"
    else:
        PPOStrategy.action_mode = "5actions"

    if StrategyClass is None:
        equity = benchmark_equity
        sharpe, sortino, mdd, win_rate, var = calculate_benchmark_metrics(equity)
        ret = ((equity.iloc[-1] - initial_cash) / initial_cash) * 100
        pf = "N/A"
    else:
        if "SMA" in strategy_name: SmaCross.n1, SmaCross.n2 = n1, n2
        elif "Bollinger" in strategy_name: BollingerBandStrategy.n, BollingerBandStrategy.k = bb_n, bb_k
        elif "Momentum" in strategy_name: MomentumStrategy.lookback = lb
        
        kwargs = {}
        if "ML:" in strategy_name:
            model, err = load_ppo_model(strategy_name, ticker)
            if model is None:
                strategy_failed = True
                error_msg = err or "Model load failed"
                equity = benchmark_equity
                sharpe, sortino, mdd, win_rate, var = calculate_benchmark_metrics(equity)
                ret = ((equity.iloc[-1] - initial_cash) / initial_cash) * 100
                pf = "N/A"
            else:
                # 🔑 KEY LOGIC: Tell PPOStrategy which mode to use
                if "PPO Regime" in strategy_name or "PPO Sector" in strategy_name:
                    PPOStrategy.action_mode = "3actions"
                else:
                    PPOStrategy.action_mode = "5actions"
                    
                kwargs["model_instance"] = model 
        
        if not strategy_failed:
            try:
                bt = Backtest(data, StrategyClass, cash=initial_cash, commission=.001)
                stats = bt.run(**kwargs)
                equity = stats["_equity_curve"]["Equity"]
                ret = stats["Return [%]"]
                sharpe = stats["Sharpe Ratio"]
                sortino = stats["Sortino Ratio"]
                mdd = stats["Max. Drawdown [%]"]
                win_rate = stats["Win Rate [%]"]
                pf = stats["Profit Factor"]
                trades_df = stats['_trades']
                var = calculate_var(equity)
            except Exception as e:
                # print(f"Backtest error: {e}")
                traceback.print_exc()
                strategy_failed = True
                error_msg = f"Runtime Error: {str(e)}"
                equity = benchmark_equity
                sharpe, sortino, mdd, win_rate, var = calculate_benchmark_metrics(equity)
                ret = ((equity.iloc[-1] - initial_cash) / initial_cash) * 100
                pf = "N/A"

    fig_eq = go.Figure()
    line_color = '#00bc8c' if not strategy_failed else '#e74c3c'
    name = 'Active Strategy' if not strategy_failed else 'Strategy Failed'
    
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name=name, line=dict(color=line_color, width=2)))
    
    if StrategyClass is not None and not strategy_failed:
        fig_eq.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, mode='lines', name='Buy & Hold', 
                                    line=dict(color='#dddddd', width=1.5, dash='dash')))

    if trades_df is not None and not trades_df.empty:
        if 'EntryTime' in trades_df.columns:
            entry_times = trades_df['EntryTime']
            entry_vals = [equity.loc[t] if t in equity.index else None for t in entry_times]
            fig_eq.add_trace(go.Scatter(x=entry_times, y=entry_vals, mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='#00ff00')))
        
        if 'ExitTime' in trades_df.columns:
            exit_times = trades_df['ExitTime']
            exit_vals = [equity.loc[t] if t in equity.index else None for t in exit_times]
            fig_eq.add_trace(go.Scatter(x=exit_times, y=exit_vals, mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='#ff0000')))

    fig_eq.update_layout(
        template="plotly_dark",
        title=dict(text=f"Equity Curve - {name}", font=dict(color="#e74c3c" if strategy_failed else "white")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(title="Portfolio Value ($)", showgrid=True, gridcolor='#333'),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(color="white")),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(30, 30, 30, 0.95)",
            font=dict(color="white", size=13),
            bordercolor="#444"
        )
    )

    fig_risk = go.Figure(data=[go.Histogram(x=equity.pct_change().dropna(), nbinsx=60, marker_color='#375a7f', opacity=0.8)])
    fig_risk.add_vline(x=var, line_color="#f39c12", line_dash="dash", annotation_text="VaR 95%", annotation_font_color="#f39c12")
    fig_risk.update_layout(
        title="Daily Returns Distribution", 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(255,255,255,0.03)',
        hoverlabel=dict(bgcolor="rgba(30, 30, 30, 0.95)", font=dict(color="white"))
    )

    fmt = lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else str(x)
    fmt_pct = lambda x: f"{x:.2f}%" if isinstance(x, (float, int)) else str(x)
    
    # Return debug info if failed
    debug_show = True if strategy_failed else False
    
    return fig_eq, fig_risk, fmt_pct(ret), fmt(sharpe), fmt(sortino), fmt_pct(mdd), fmt_pct(win_rate), fmt(pf), error_msg, debug_show

@app.callback(
    Output("graph-monte", "figure"), 
    [Input("btn-monte", "n_clicks")], 
    [State("ticker-drop", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")]
)
def run_monte(n, ticker, start, end):
    if not n: return go.Figure()
    data = get_stock_data(ticker, start, end)
    if data.empty: return go.Figure()
    last = data["Close"].iloc[-1]
    vol = data["Close"].pct_change().std()
    fig = go.Figure()
    for _ in range(50):
        prices = [last]
        for _ in range(60):
            prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
        fig.add_trace(go.Scatter(y=prices, mode='lines', line=dict(width=1, color='rgba(0,188,140,0.2)'), showlegend=False))
    fig.update_layout(
        title="Monte Carlo Simulation", 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(255,255,255,0.03)',
        hoverlabel=dict(bgcolor="rgba(30, 30, 30, 0.95)", font=dict(color="white"))
    )
    return fig

@app.callback(
    Output("graph-heat", "figure"), 
    [Input("btn-opt", "n_clicks")], 
    [State("ticker-drop", "value"), State("strategy-drop", "value"), State("date-picker", "start_date"), State("date-picker", "end_date")]
)
def run_opt(n, ticker, strategy, start, end):
    if not n: return go.Figure()
    data = get_stock_data(ticker, start, end)
    if data.empty: return go.Figure()
    res, x_ax, y_ax, xl, yl = [], [], [], "", ""
    if "SMA" in strategy:
        x_r, y_r = range(20, 80, 10), range(5, 35, 5)
        xl, yl = "Slow MA", "Fast MA"
        x_ax, y_ax = list(x_r), list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                if y >= x: row.append(0)
                else:
                    SmaCross.n1, SmaCross.n2 = y, x
                    try: row.append(Backtest(data, SmaCross, cash=50000, commission=.001).run()["Return [%]"])
                    except: row.append(0)
            res.append(row)
    elif "Bollinger" in strategy:
        x_r, y_r = [1.5, 2.0, 2.5], range(10, 50, 10)
        xl, yl = "StdDev (K)", "Window (N)"
        x_ax, y_ax = x_r, list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                BollingerBandStrategy.n, BollingerBandStrategy.k = y, x
                try: row.append(Backtest(data, BollingerBandStrategy, cash=50000, commission=.001).run()["Return [%]"])
                except: row.append(0)
            res.append(row)
    elif "Momentum" in strategy:
        x_r, y_r = [-0.01, 0.0, 0.01], range(10, 60, 10)
        xl, yl = "Threshold", "Lookback"
        x_ax, y_ax = x_r, list(y_r)
        for y in y_r:
            row = []
            for x in x_r:
                MomentumStrategy.lookback, MomentumStrategy.threshold = y, x
                try: row.append(Backtest(data, MomentumStrategy, cash=50000, commission=.001).run()["Return [%]"])
                except: row.append(0)
            res.append(row)
    else: return go.Figure()
    fig = go.Figure(data=go.Heatmap(z=res, x=x_ax, y=y_ax, colorscale='Viridis', colorbar=dict(title="Return %")))
    fig.update_layout(
        title=f"Parameter Heatmap: {yl} vs {xl}", 
        xaxis_title=xl, yaxis_title=yl, 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor="rgba(30, 30, 30, 0.95)", font=dict(color="white"))
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8050)
