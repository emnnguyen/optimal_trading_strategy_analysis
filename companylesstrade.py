import pandas as pd
import numpy as np
import glob
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
from collections import defaultdict
import os

TRAIN_RATIO = 0.7
CASH = 10_000
COMMISSION = 0.001 
WINDOW = 10
POSITION_SIZE = 0.2 

DATA_DIR = "/Users/vanessaliu/Desktop/STA160/featurecompany"

SECTOR_CONFIGS_EVIDENCE_BASED = {
    'Technology': {
        'annual_return': 0.28, 'annual_volatility': 0.35, 'min_hold_days': 10,
        'trade_frequency_optimal': 15,
    },
    'Financials': {
        'annual_return': 0.15, 'annual_volatility': 0.28, 'min_hold_days': 10,
        'trade_frequency_optimal': 12,
    },
    'Healthcare': {
        'annual_return': 0.12, 'annual_volatility': 0.18, 'min_hold_days': 15,
        'trade_frequency_optimal': 8,
    },
    'Consumer Discretionary': {
        'annual_return': 0.22, 'annual_volatility': 0.32, 'min_hold_days': 10,
        'trade_frequency_optimal': 14,
    },
    'Consumer Staples': {
        'annual_return': 0.10, 'annual_volatility': 0.14, 'min_hold_days': 20,
        'trade_frequency_optimal': 6,
    },
    'Energy': {
        'annual_return': 0.18, 'annual_volatility': 0.42, 'min_hold_days': 15,
        'trade_frequency_optimal': 12,
    },
    'Utilities': {
        'annual_return': 0.09, 'annual_volatility': 0.16, 'min_hold_days': 30,
        'trade_frequency_optimal': 5,
    },
    'Industrials': {
        'annual_return': 0.14, 'annual_volatility': 0.24, 'min_hold_days': 7,
        'trade_frequency_optimal': 10,
    },
    'Communication Services': {
        'annual_return': 0.16, 'annual_volatility': 0.26, 'min_hold_days': 5,
        'trade_frequency_optimal': 12,
    },
    'Real Estate': {
        'annual_return': 0.11, 'annual_volatility': 0.20, 'min_hold_days': 15,
        'trade_frequency_optimal': 8,
    },
}

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

def load_data():
    pattern = os.path.join(DATA_DIR, "*_features.csv")
    data_files = glob.glob(pattern)
    print(f"Found {len(data_files)} stocks in {DATA_DIR}")

    if len(data_files) == 0:
        raise ValueError("No data files found! Please check DATA_DIR path.")

    dfs = []
    stock_names = []

    for file in data_files:
        df = pd.read_csv(file)
        df = df.sort_values("Date").reset_index(drop=True)
        dfs.append(df)
        name = os.path.basename(file).replace("_features.csv", "")
        stock_names.append(name)

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    
    common_cols = list(common_cols)

    all_train_data = []
    all_test_data = []

    final_feature_cols = []
    
    for i, df in enumerate(dfs):
        df2 = df[["Date", "Close"] + [c for c in common_cols if c not in ["Date", "Close"]]].copy()
        df2 = add_trading_signals(df2)
        
        if i == 0:
            potential_cols = [c for c in df2.columns if c not in ["Date", "Close"]]
        
            safe_keywords = ['return', 'volatility', 'trend', 'strength', 'position', 'vs', 'ratio']
            final_feature_cols = [
                c for c in potential_cols 
                if any(k in c for k in safe_keywords) 
                and 'ma_' not in c 
            ]
            print(f"✅ Selected Features for RL ({len(final_feature_cols)}): {final_feature_cols}")

        n_train = int(len(df2) * TRAIN_RATIO)
        train_data = df2.iloc[:n_train].reset_index(drop=True)
        test_data = df2.iloc[n_train:].reset_index(drop=True)

        all_train_data.append(train_data)
        all_test_data.append(test_data)

    return all_train_data, all_test_data, stock_names, final_feature_cols


def add_trading_signals(df):
    df = df.copy()
    
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    
    ma_5 = df['Close'].rolling(5).mean()
    ma_20 = df['Close'].rolling(20).mean()
    ma_50 = df['Close'].rolling(50).mean()
    
    df['price_vs_ma5'] = df['Close'] / ma_5 - 1
    df['price_vs_ma20'] = df['Close'] / ma_20 - 1
    
    df['trend_5_20'] = (ma_5 - ma_20) / ma_20
    df['trend_20_50'] = (ma_20 - ma_50) / ma_50
    
    df['volatility'] = df['Close'].pct_change().rolling(20).std()
    
    rolling_mean = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    df['bollinger_position'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-8)
    
    df['strength_rsi_proxy'] = df['Close'].rolling(14).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
    )
    
    return df.fillna(0)


def build_trading_config_from_sector(sector_name, sector_conf):
    config = {
        "name": f"{sector_name} strategy",
        "min_hold_days": sector_conf["min_hold_days"],
        "optimal_trades_year": sector_conf.get("trade_frequency_optimal", 10),
    }
    return config


def group_data_by_sector(all_train_data, all_test_data, stock_names):
    sector_train_data = defaultdict(list)
    sector_test_data = defaultdict(list)
    sector_stock_names = defaultdict(list)

    for train_df, test_df, name in zip(all_train_data, all_test_data, stock_names):
        sector = SECTOR_MAP.get(name)
        if sector is None:
            continue
        
        sector_train_data[sector].append(train_df)
        sector_test_data[sector].append(test_df)
        sector_stock_names[sector].append(name)
    
    return sector_train_data, sector_test_data, sector_stock_names


class ConfigurableStockEnv(gym.Env):
    def __init__(self, df_list, feature_cols, style_config, 
                 cash=CASH, commission=COMMISSION, window=WINDOW):
        super().__init__()

        self.df_list = df_list
        self.feature_cols = feature_cols
        self.window = window
        self.commission = commission
        self.initial_cash = cash
        self.config = style_config

        self.trade_penalty_factor = 0.0005 
        
        obs_dim = len(feature_cols) * window + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # ✅ 使用老式 RandomState，NumPy 1.x / 2.x 都兼容
        self.rng = np.random.RandomState()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 如有 seed，则设置随机数种子
        if seed is not None:
            self.rng.seed(seed)

        # 用 RandomState 来选择当前股票
        self.current_stock_idx = self.rng.randint(0, len(self.df_list))
        self.df = self.df_list[self.current_stock_idx].reset_index(drop=True)

        self.features = self.df[self.feature_cols].values
        self.prices = self.df["Close"].values
        self.max_step = len(self.df) - 1

        self.step_idx = self.window - 1
        self.cash = self.initial_cash
        self.position = 0 
        self.trade_count = 0
        self.last_action = 0
        
        self.last_buy_price = 0
        self.hold_days = 0
        self.portfolio_value = self.initial_cash
        
        return self._get_obs(), {}
       
    def _get_obs(self):
        start = max(0, self.step_idx - self.window + 1)
        hist = self.features[start:self.step_idx + 1]

        if len(hist) < self.window:
            pad = np.zeros((self.window - len(hist), hist.shape[1]))
            hist = np.vstack([pad, hist])

        price = self.prices[self.step_idx]
        
        cash_ratio = self.cash / self.portfolio_value
        position_val_ratio = (self.position * price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        return np.concatenate([
            hist.flatten(),
            [cash_ratio],           
            [position_val_ratio],    
            [self.last_action / 2.0],
            [min(self.hold_days, 100) / 100.0] 
        ]).astype(np.float32)

    def step(self, action):
        prev_portfolio_value = self.cash + self.position * self.prices[self.step_idx]
        
        self.step_idx += 1
        current_price = self.prices[self.step_idx]
        
        reward = 0
        trade_occurred = False
        min_hold = self.config['min_hold_days']

        # Action 1: Buy
        if action == 1: 
            if self.cash > 100: 
                buy_amount = self.cash * (1 - self.commission)
                self.position += buy_amount / current_price
                self.cash = 0
                
                self.last_buy_price = current_price
                self.hold_days = 0
                self.trade_count += 1
                trade_occurred = True
            else:
                reward -= 0.01

        # Action 2: Sell
        elif action == 2:
            if self.position > 0:
                if self.hold_days < min_hold:
                    reward -= 0.5 
                
                revenue = self.position * current_price * (1 - self.commission)
                self.cash += revenue
                
                profit_pct = (current_price - self.last_buy_price) / (self.last_buy_price + 1e-8)
                reward += profit_pct * 5.0 
                
                self.position = 0
                self.trade_count += 1
                trade_occurred = True
            else:
                reward -= 0.01

        # Action 0: Hold
        elif action == 0:
            if self.position > 0:
                self.hold_days += 1
        
                if current_price > self.prices[self.step_idx - 1]:
                    reward += 0.05
            else:
                reward -= 0.005

        self.portfolio_value = self.cash + self.position * current_price
        
        step_return = (self.portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)
        reward += step_return * 100
        
        if trade_occurred:
            reward -= self.trade_penalty_factor

        done = self.step_idx >= self.max_step
        
        if done:
            final_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
            reward += final_return * 10 
            if self.trade_count == 0:
                reward -= 2.0

        info = {
            "equity": self.portfolio_value,
            "trade_count": self.trade_count
        }

        return self._get_obs(), reward, done, False, info


def train_sector_models(all_train_data, all_test_data, stock_names, common_cols):
    os.makedirs("models", exist_ok=True)

    sector_train_data, _, sector_stock_names = group_data_by_sector(
        all_train_data, all_test_data, stock_names
    )
    
    models = {}
    
    for sector, df_list in sector_train_data.items():
        if sector not in SECTOR_CONFIGS_EVIDENCE_BASED:
            continue
        
        sector_conf = SECTOR_CONFIGS_EVIDENCE_BASED[sector]
        style_config = build_trading_config_from_sector(sector, sector_conf)
        
        print(f"\n{'='*60}")
        print(f"🚀 Training sector model: {sector}")
        print(f"   Stocks: {len(df_list)}")
        print(f"{'='*60}")
        
        n_envs = 8 
        train_env = SubprocVecEnv([
            (lambda config=style_config, data_list=df_list: 
                ConfigurableStockEnv(data_list, common_cols, config))
            for _ in range(n_envs)
        ])
        
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            batch_size=256,
            n_steps=256,
            learning_rate=3e-4,
            n_epochs=10,
            ent_coef=0.01,
            gamma=0.99,
            device='auto'
        )
        
        model.learn(total_timesteps=80_000)
        
        save_name = os.path.join("models", f"ppo_sector_{sector.replace(' ', '_')}")
        model.save(save_name)   
        print(f"✅ Saved sector model to {save_name}.zip")
        
        models[sector] = {
            "model": model,
            "config": style_config,
        }
        
        train_env.close()
    
    return models


def evaluate_and_save(models, all_test_data, stock_names, common_cols):
    results = []
    name_to_testdf = {name: df for name, df in zip(stock_names, all_test_data)}
    
    for sector, model_info in models.items():
        model = model_info["model"]
        config = model_info["config"]
        
        for stock_name, stock_sector in SECTOR_MAP.items():
            if stock_sector != sector or stock_name not in name_to_testdf:
                continue
            
            test_df = name_to_testdf[stock_name]
            env = ConfigurableStockEnv([test_df], common_cols, config)
            obs, _ = env.reset()
            
            equity_curve = []
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                equity_curve.append(info["equity"])

            final_equity = info["equity"]
            ret_pct = (final_equity - CASH) / CASH * 100
            trades = info["trade_count"]
            
            # Baseline: Buy & Hold
            bh_start = test_df["Close"].iloc[0]
            bh_end = test_df["Close"].iloc[-1]
            bh_ret = (bh_end - bh_start) / bh_start * 100
            
            results.append({
                "Stock": stock_name,
                "Sector": sector,
                "RL_Return_Pct": round(ret_pct, 2),
                "RL_Trades": trades,
                "BH_Return_Pct": round(bh_ret, 2),
                "RL_vs_BH": round(ret_pct - bh_ret, 2)
            })
            
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("RL_Return_Pct", ascending=False)
    
    print("\n📊 Evaluation Results:")
    print(df_res.head(10))
    
    df_res.to_csv("final_strategy_results.csv", index=False)
    print("\n✅ Saved results to 'final_strategy_results.csv'")
    
    if len(results) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_res["RL_Trades"], df_res["RL_Return_Pct"], alpha=0.6)
        plt.xlabel("Number of Trades")
        plt.ylabel("Return (%)")
        plt.title("RL Strategy: Trades vs Return")
        plt.axhline(0, linestyle='--')
        plt.savefig("trades_vs_return.png")


if __name__ == "__main__":
    
    print("📂 Loading data...")
    all_train_data, all_test_data, stock_names, common_cols = load_data()
    
    if not all_train_data:
        print("❌ No data loaded. Exiting.")
        exit()

    print(f"🎯 Features used for training: {common_cols}")
    
    print("\n🎯 Training sector-based models...")
    models = train_sector_models(all_train_data, all_test_data, stock_names, common_cols)
    
    print("\n📈 Evaluating strategies...")
    evaluate_and_save(models, all_test_data, stock_names, common_cols)
