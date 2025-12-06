import pandas as pd
import numpy as np
import glob
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
from collections import defaultdict

TRAIN_RATIO = 0.7
CASH = 10_000
COMMISSION = 0.001
WINDOW = 10
POSITION_SIZE = 0.2  

DATA_DIR = "/Users/vanessaliu/Desktop/STA160/featurecompany/clean_out(data_with_features_cleaned)"


SECTOR_CONFIGS_EVIDENCE_BASED = {
    'Technology': {
        'annual_return': 0.28,
        'annual_volatility': 0.35,
        'sharpe_ratio': 0.80,
        'max_drawdown': -0.25,
        'min_hold_days': 10,
        'optimal_hold_days': 60,
        'profit_threshold': 0.025,
        'stop_loss': -0.045,
        'momentum_half_life': 90,
        'trade_frequency_optimal': 18,  
    },
    'Financials': {
        'annual_return': 0.15,
        'annual_volatility': 0.28,
        'sharpe_ratio': 0.54,
        'max_drawdown': -0.35,
        'min_hold_days': 10,
        'optimal_hold_days': 120,
        'profit_threshold': 0.035,
        'stop_loss': -0.04,
        'interest_rate_sensitivity': 2.5,
        'credit_cycle_beta': 1.8,
        'trade_frequency_optimal': 12,
    },
    'Healthcare': {
        'annual_return': 0.12,
        'annual_volatility': 0.18,
        'sharpe_ratio': 0.67,
        'max_drawdown': -0.15,
        'min_hold_days': 15,
        'optimal_hold_days': 180,
        'profit_threshold': 0.045,
        'stop_loss': -0.03,
        'defensive_beta': 0.75,
        'quality_score_weight': 1.5,
        'trade_frequency_optimal': 10,
    },
    'Consumer Discretionary': {
        'annual_return': 0.22,
        'annual_volatility': 0.32,
        'sharpe_ratio': 0.69,
        'max_drawdown': -0.28,
        'min_hold_days': 10,
        'optimal_hold_days': 45,
        'profit_threshold': 0.02,
        'stop_loss': -0.055,
        'consumer_sentiment_beta': 1.8,
        'trade_frequency_optimal': 14,
    },
    'Consumer Staples': {
        'annual_return': 0.10,
        'annual_volatility': 0.14,
        'sharpe_ratio': 0.71,
        'max_drawdown': -0.12,
        'min_hold_days': 20,
        'optimal_hold_days': 240,
        'profit_threshold': 0.05,
        'stop_loss': -0.025,
        'dividend_yield_avg': 0.028,
        'recession_beta': 0.6,
        'trade_frequency_optimal': 8,
    },
    'Energy': {
        'annual_return': 0.18,
        'annual_volatility': 0.42,
        'sharpe_ratio': 0.43,
        'max_drawdown': -0.40,
        'min_hold_days': 15,
        'optimal_hold_days': 30,
        'profit_threshold': 0.025,
        'stop_loss': -0.065,
        'oil_price_beta': 1.5,
        'seasonality_factor': 1.3,
        'trade_frequency_optimal': 14,
    },
    'Utilities': {
        'annual_return': 0.09,
        'annual_volatility': 0.16,
        'sharpe_ratio': 0.56,
        'max_drawdown': -0.18,
        'min_hold_days': 30,
        'optimal_hold_days': 365,
        'profit_threshold': 0.06,
        'stop_loss': -0.02,
        'interest_rate_beta': -2.0,
        'dividend_yield_avg': 0.035,
        'trade_frequency_optimal': 6,
    },
    'Industrials': {
        'annual_return': 0.14,
        'annual_volatility': 0.24,
        'sharpe_ratio': 0.58,
        'max_drawdown': -0.22,
        'min_hold_days': 7,
        'optimal_hold_days': 90,
        'profit_threshold': 0.03,
        'stop_loss': -0.04,
        'gdp_beta': 1.4,
        'ism_manufacturing_beta': 1.6,
        'trade_frequency_optimal': 12,
    },
    'Communication Services': {
        'annual_return': 0.16,
        'annual_volatility': 0.26,
        'sharpe_ratio': 0.62,
        'max_drawdown': -0.24,
        'min_hold_days': 5,
        'optimal_hold_days': 75,
        'profit_threshold': 0.028,
        'stop_loss': -0.045,
        'trade_frequency_optimal': 16,
    },
    'Real Estate': {
        'annual_return': 0.11,
        'annual_volatility': 0.20,
        'sharpe_ratio': 0.55,
        'max_drawdown': -0.20,
        'min_hold_days': 15,
        'optimal_hold_days': 180,
        'profit_threshold': 0.04,
        'stop_loss': -0.035,
        'interest_rate_beta': -1.5,
        'dividend_yield_avg': 0.04,
        'trade_frequency_optimal': 10,
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
    """加载并预处理股票数据"""
    pattern = DATA_DIR + "/*_features_deep_clean.csv"
    data_files = glob.glob(pattern)
    print(f"Found {len(data_files)} stocks")

    dfs = []
    stock_names = []

    for file in data_files:
        df = pd.read_csv(file)
        df = df.sort_values("Date").reset_index(drop=True)
        dfs.append(df)
        name = file.split("/")[-1].replace("_features_deep_clean.csv", "")
        stock_names.append(name)

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    common_cols = [c for c in common_cols if c not in ["Date", "Close"]]

    all_train_data = []
    all_test_data = []

    for df in dfs:
        df2 = df[["Date", "Close"] + common_cols].copy()
        df2 = add_trading_signals(df2)

        n_train = int(len(df2) * TRAIN_RATIO)
        train_data = df2.iloc[:n_train].reset_index(drop=True)
        test_data = df2.iloc[n_train:].reset_index(drop=True)

        all_train_data.append(train_data)
        all_test_data.append(test_data)

    common_cols = [c for c in all_train_data[0].columns if c not in ["Date", "Close"]]
    return all_train_data, all_test_data, stock_names, common_cols


def add_trading_signals(df):
    df = df.copy()

    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)

    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()

    df['price_vs_ma5'] = df['Close'] / df['ma_5'] - 1
    df['price_vs_ma20'] = df['Close'] / df['ma_20'] - 1


    df['trend_5_20'] = (df['ma_5'] - df['ma_20']) / df['ma_20']
    df['trend_20_50'] = (df['ma_20'] - df['ma_50']) / df['ma_50']

    df['volatility'] = df['Close'].pct_change().rolling(20).std()

    rolling_mean = df['Close'].rolling(20).mean()
    rolling_std = df['Close'].rolling(20).std()
    df['price_position'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-8)

    df['strength'] = df['Close'].rolling(10).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
    )

    return df.fillna(0)


def build_trading_config_from_sector(sector_name, sector_conf):
    vol = sector_conf['annual_volatility']
    if vol <= 0.18:
        trade_cost_penalty = 0.004
    elif vol <= 0.28:
        trade_cost_penalty = 0.002
    else:
        trade_cost_penalty = 0.0

    target_trades_per_year = sector_conf.get("trade_frequency_optimal", 10)
    target_daily_trade_freq = target_trades_per_year / 252.0

    config = {
        "name": f"{sector_name} sector strategy",
        "trade_cost_penalty": trade_cost_penalty,
        "profit_reward_scale": 100,   
        "future_reward_scale": 50,    
        "final_reward_scale": 10,
        "min_hold_days": sector_conf["min_hold_days"],

        "target_daily_trade_freq": target_daily_trade_freq,
        "trade_freq_penalty": 0.5,
        "trade_frequency_optimal": target_trades_per_year,
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

        obs_dim = len(feature_cols) * window + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_stock_idx = np.random.randint(0, len(self.df_list))
        self.df = self.df_list[self.current_stock_idx].reset_index(drop=True)

        self.features = self.df[self.feature_cols].values
        self.prices = self.df["Close"].values
        self.max_step = len(self.df) - 1

        self.step_idx = self.window - 1
        self.cash = self.initial_cash
        self.position = 0.0
        self.trade_count = 0
        self.last_action = 0

        self.last_buy_price = 0.0
        self.hold_days = 0
        self.profitable_trades = 0
        self.total_trades = 0

    
        days_in_episode = self.max_step - (self.window - 1)
        years_in_episode = max(days_in_episode / 252.0, 0.1)
        optimal_per_year = self.config.get("trade_frequency_optimal", 12)
        self.target_trades_episode = max(
            1,
            int(optimal_per_year * years_in_episode)
        )

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        start = max(0, self.step_idx - self.window + 1)
        hist = self.features[start:self.step_idx + 1]

        if len(hist) < self.window:
            pad = np.tile(self.features[0], (self.window - len(hist), 1))
            hist = np.vstack([pad, hist])

        price = self.prices[self.step_idx]
        return np.concatenate([
            hist.flatten(),
            [self.cash / self.initial_cash],
            [self.position * price / self.initial_cash],
            [self.last_action],
            [self.hold_days / 100.0],
        ])

    def step(self, action):
        price = self.prices[self.step_idx]
        old_equity = self.cash + self.position * price

        trade_occurred = False
        min_hold_days = self.config['min_hold_days']

        if self.position > 0:
            self.hold_days += 1

        if action == 1:
            available_cash = self.cash * POSITION_SIZE
            if available_cash > 10:
                shares_to_buy = (available_cash * (1 - self.commission)) / price
                self.position += shares_to_buy
                self.cash -= available_cash
                self.last_buy_price = price
                self.trade_count += 1
                self.total_trades += 1
                trade_occurred = True

        elif action == 2:
            if self.position > 0 and self.hold_days >= min_hold_days:
                shares_to_sell = self.position * POSITION_SIZE
                cash_received = shares_to_sell * price * (1 - self.commission)
                self.cash += cash_received

                if price > self.last_buy_price:
                    self.profitable_trades += 1

                self.position -= shares_to_sell
                if self.position < 0.01:
                    self.position = 0.0
                    self.hold_days = 0

                self.trade_count += 1
                self.total_trades += 1
                trade_occurred = True
            else:
                action = 0  

        self.step_idx += 1
        new_price = self.prices[self.step_idx]
        new_equity = self.cash + self.position * new_price

        reward = self._calculate_reward(
            action, trade_occurred, old_equity, new_equity, price, new_price
        )

        done = self.step_idx >= self.max_step

        if done:
            final_return = (new_equity - self.initial_cash) / self.initial_cash
            reward += final_return * self.config['final_reward_scale']

            if self.total_trades >= 5:
                reward += 0.3

            if self.total_trades > 0:
                win_rate = self.profitable_trades / self.total_trades
                if win_rate > 0.6:
                    reward += 0.5


        self.last_action = action

        info = {
            "equity": new_equity,
            "trade_count": self.trade_count,
            "profitable_trades": self.profitable_trades,
            "total_trades": self.total_trades,
        }

        return self._get_obs(), reward, done, False, info

    def _calculate_reward(self, action, trade_occurred,
                          old_equity, new_equity, old_price, new_price):
        
        equity_change = (new_equity - old_equity) / (old_equity + 1e-8)
        reward = equity_change * 100

        if trade_occurred:
            reward += 0.3  

            if self.trade_count > self.target_trades_episode * 1.5:
                reward -= 0.2

        reward = np.clip(reward, -10, 10)
        return reward


def train_sector_models(all_train_data, all_test_data, stock_names, common_cols):
    sector_train_data, sector_test_data, sector_stock_names = group_data_by_sector(
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
        print(f"   Name: {style_config['name']}")
        print(f"   Stocks: {', '.join(sector_stock_names[sector])}")
        print(f"{'='*60}")

        n_envs = 4
        train_env = SubprocVecEnv([
            (lambda config=style_config, data_list=df_list:
             ConfigurableStockEnv(data_list, common_cols, config))
            for _ in range(n_envs)
        ])

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            batch_size=128,
            n_steps=128,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
        )

        model.learn(total_timesteps=50_000)
        model.save(f"ppo_sector_{sector}")

        models[sector] = {
            "model": model,
            "config": style_config,
        }

        train_env.close()

    return models


def evaluate_all_models(models, all_test_data, stock_names, common_cols):
    all_results = defaultdict(list)
    name_to_testdf = {name: df for name, df in zip(stock_names, all_test_data)}

    for sector, model_info in models.items():
        model = model_info["model"]
        config = model_info["config"]

        print(f"\n📊 Evaluating sector model: {sector}")

        for stock_name, stock_sector in SECTOR_MAP.items():
            if stock_sector != sector or stock_name not in name_to_testdf:
                continue

            test_df = name_to_testdf[stock_name]
            env = ConfigurableStockEnv([test_df], common_cols, config)
            obs, _ = env.reset()

            equity_curve = []
            actions_log = []
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                equity_curve.append(info["equity"])
                actions_log.append(int(action))

            final_equity = info["equity"]
            ret_pct = (final_equity - CASH) / CASH * 100
            trades = info["trade_count"]
            profitable = info.get("profitable_trades", 0)
            total = info.get("total_trades", 0)
            win_rate = (profitable / total * 100) if total > 0 else 0

            returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0.0])
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

            peak = np.maximum.accumulate(equity_curve)
            drawdown = (np.array(equity_curve) - peak) / peak
            max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

            # Buy & Hold
            bh_start_price = test_df["Close"].iloc[0]
            bh_end_price = test_df["Close"].iloc[-1]
            bh_equity = CASH * (bh_end_price / bh_start_price)
            bh_ret_pct = (bh_equity - CASH) / CASH * 100

            all_results[stock_name].append({
                'style': sector,
                'style_name': config['name'],
                'return_pct': ret_pct,
                'num_trades': trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'final_equity': final_equity,
                'num_buy': actions_log.count(1),
                'num_sell': actions_log.count(2),
                'num_hold': actions_log.count(0),
                'bh_return_pct': bh_ret_pct,
                'bh_equity': bh_equity,
            })

    return all_results


def select_best_strategies(all_results):
    best_strategies = []

    for stock_name, results in all_results.items():
        df_results = pd.DataFrame(results)

        df_results['score'] = (
            df_results['return_pct'] * 0.5 +
            df_results['sharpe_ratio'] * 10 * 0.3 +
            -df_results['max_drawdown'] * 0.2
        )

        best_idx = df_results['score'].idxmax()
        best = df_results.iloc[best_idx]

        bh_ret = results[0]['bh_return_pct']
        bh_eq = results[0]['bh_equity']

        best_strategies.append({
            'stock_name': stock_name,
            'best_style': best['style'],
            'best_style_name': best['style_name'],
            'return_pct': best['return_pct'],
            'num_trades': best['num_trades'],
            'win_rate': best['win_rate'],
            'sharpe_ratio': best['sharpe_ratio'],
            'max_drawdown': best['max_drawdown'],
            'score': best['score'],
            'bh_return_pct': bh_ret,
            'bh_equity': bh_eq,
            'all_strategies': df_results.to_dict('records'),
        })

    return pd.DataFrame(best_strategies)


def visualize_comparison(all_results, save_path='strategy_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    styles = sorted({r['style'] for stock_results in all_results.values() for r in stock_results})
    style_names = styles

    avg_returns = []
    avg_trades = []
    avg_sharpe = []
    avg_drawdown = []

    for style in styles:
        style_results = []
        for stock_results in all_results.values():
            for r in stock_results:
                if r['style'] == style:
                    style_results.append(r)

        if len(style_results) == 0:
            continue

        avg_returns.append(np.mean([r['return_pct'] for r in style_results]))
        avg_trades.append(np.mean([r['num_trades'] for r in style_results]))
        avg_sharpe.append(np.mean([r['sharpe_ratio'] for r in style_results]))
        avg_drawdown.append(np.mean([r['max_drawdown'] for r in style_results]))

    axes[0, 0].bar(range(len(style_names)), avg_returns)
    axes[0, 0].set_title('Average Return by Strategy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].set_xticks(range(len(style_names)))
    axes[0, 0].set_xticklabels(style_names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)

    axes[0, 1].bar(range(len(style_names)), avg_trades)
    axes[0, 1].set_title('Average Number of Trades', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Trades')
    axes[0, 1].set_xticks(range(len(style_names)))
    axes[0, 1].set_xticklabels(style_names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    axes[1, 0].bar(range(len(style_names)), avg_sharpe)
    axes[1, 0].set_title('Average Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_xticks(range(len(style_names)))
    axes[1, 0].set_xticklabels(style_names, rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    axes[1, 1].bar(range(len(style_names)), avg_drawdown)
    axes[1, 1].set_title('Average Max Drawdown', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].set_xticks(range(len(style_names)))
    axes[1, 1].set_xticklabels(style_names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comparison chart to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("📂 Loading data...")
    all_train_data, all_test_data, stock_names, common_cols = load_data()

    print("\n🎯 Training sector-based models...")
    models = train_sector_models(all_train_data, all_test_data, stock_names, common_cols)

    print("\n📈 Evaluating all sector strategies on test data...")
    all_results = evaluate_all_models(models, all_test_data, stock_names, common_cols)

    print("\n🏆 Selecting best strategy for each stock...")
    best_strategies_df = select_best_strategies(all_results)

    best_strategies_df.to_csv("best_strategies_per_stock.csv", index=False)

    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)

    print("\n🎯 Strategy Distribution (by sector):")
    strategy_counts = best_strategies_df['best_style'].value_counts()
    for style, count in strategy_counts.items():
        print(f"   {style:30s}: {count:2d} stocks ({count/len(best_strategies_df)*100:.1f}%)")

    print("\n📈 Overall Performance (RL vs Buy & Hold):")
    avg_rl_ret = best_strategies_df['return_pct'].mean()
    avg_bh_ret = best_strategies_df['bh_return_pct'].mean()
    print(f"   Avg RL Return:        {avg_rl_ret:6.2f}%")
    print(f"   Avg B&H Return:       {avg_bh_ret:6.2f}%")

    avg_rl_eq = CASH * (1 + avg_rl_ret / 100)
    avg_bh_eq = CASH * (1 + avg_bh_ret / 100)
    print(f"   Avg RL Final Equity:  ${avg_rl_eq:8.2f}")
    print(f"   Avg B&H Final Equity: ${avg_bh_eq:8.2f}")

    print(f"   Median RL Return:     {best_strategies_df['return_pct'].median():6.2f}%")
    print(f"   Best RL Return:       {best_strategies_df['return_pct'].max():6.2f}% "
          f"({best_strategies_df.loc[best_strategies_df['return_pct'].idxmax(), 'stock_name']})")
    print(f"   Average Trades:       {best_strategies_df['num_trades'].mean():6.1f}")
    print(f"   Average Win Rate:     {best_strategies_df['win_rate'].mean():6.1f}%")
    print(f"   Average Sharpe:       {best_strategies_df['sharpe_ratio'].mean():6.2f}")
    print(f"   Average Max DD:       {best_strategies_df['max_drawdown'].mean():6.2f}%")

    print("\n🏆 Top 5 Performers (RL vs Buy & Hold):")
    top5 = best_strategies_df.nlargest(5, 'return_pct')
    for idx, row in top5.iterrows():
        rl_eq = CASH * (1 + row['return_pct'] / 100)
        bh_eq = row['bh_equity']
        print(f"   {row['stock_name']:8s}: RL {row['return_pct']:6.2f}% (${rl_eq:8.0f}) "
              f"| B&H {row['bh_return_pct']:6.2f}% (${bh_eq:8.0f}) "
              f"({row['best_style_name']}, {row['num_trades']:.0f} trades)")

    visualize_comparison(all_results)

    detailed_results = []
    for stock_name, results in all_results.items():
        for r in results:
            r['stock_name'] = stock_name
            detailed_results.append(r)

    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv("all_strategies_detailed.csv", index=False)

    print("\n✅ Saved files:")
    print("   - best_strategies_per_stock.csv")
    print("   - all_strategies_detailed.csv")
    print("   - strategy_comparison.png")
    print("   - ppo_sector_*.zip (for each sector)")

