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
MIN_REGIME_LENGTH = 90

DATA_DIR = "/Users/vanessaliu/Desktop/STA160/featurecompany/clean_out(data_with_features_cleaned)"


def classify_market_regime(df, lookback=90):
    df = df.copy()
    
    if 'ma_50' not in df.columns or 'ma_200' not in df.columns:
        df['ma_50'] = df['Close'].rolling(50).mean()
        df['ma_200'] = df['Close'].rolling(200).mean()
    
    ma50 = df['ma_50']
    ma200 = df['ma_200']
    
    regimes = []
    
    for i in range(lookback - 1, len(df)):
        window_fast = ma50.iloc[i - lookback + 1: i + 1]
        window_slow = ma200.iloc[i - lookback + 1: i + 1]
        
        cond = window_fast > window_slow
        
        if cond.all():
            regime = 'bull'
        elif (~cond).all():
            regime = 'bear'
        else:
            regime = 'sideways'
        
        regimes.append(regime)
    
    full_regimes = ['neutral'] * (lookback - 1) + regimes
    return full_regimes


def add_market_regime_labels(df):
    df = df.copy()
    regimes = classify_market_regime(df, lookback=90)
    df['market_regime'] = regimes
    return df


def split_by_regime(all_data, stock_names, train_ratio=0.7, min_regime_length=MIN_REGIME_LENGTH):
    print("\n🔍 Analyzing market regimes...")
    
    regime_data = {
        'bull': {'train': [], 'test': [], 'dates': []},
        'bear': {'train': [], 'test': [], 'dates': []},
        'sideways': {'train': [], 'test': [], 'dates': []}
    }
    
    regime_stats = {'bull': 0, 'bear': 0, 'sideways': 0, 'neutral': 0}
    
    for stock_name, df in zip(stock_names, all_data):
        df = add_market_regime_labels(df)
        
        for regime in df['market_regime'].unique():
            if regime in regime_stats:
                regime_stats[regime] += (df['market_regime'] == regime).sum()
        
        for regime in ['bull', 'bear', 'sideways']:
            mask = (df['market_regime'] == regime).values
            start_idx = None

            for i, flag in enumerate(mask):
                if flag and start_idx is None:
                    start_idx = i

                is_last = i == len(mask) - 1
                should_close = (not flag or is_last) and start_idx is not None
                if should_close:
                    end_idx = i if not flag else i + 1
                    seg = df.iloc[start_idx:end_idx].copy()

                    if len(seg) >= min_regime_length:
                        n_train = int(len(seg) * train_ratio)
                        train_segment = seg.iloc[:n_train].reset_index(drop=True)
                        test_segment = seg.iloc[n_train:].reset_index(drop=True)

                        if len(train_segment) >= 60 and len(test_segment) >= 20:
                            regime_data[regime]['train'].append(train_segment)
                            regime_data[regime]['test'].append(test_segment)

                            if 'Date' in train_segment.columns:
                                regime_data[regime]['dates'].append({
                                    'stock': stock_name,
                                    'train_start': train_segment['Date'].iloc[0],
                                    'train_end': train_segment['Date'].iloc[-1],
                                    'test_start': test_segment['Date'].iloc[0],
                                    'test_end': test_segment['Date'].iloc[-1],
                                })

                    start_idx = None
    
    print("\n📊 Market Regime Statistics:")
    print(f"{'Regime':<15} {'Total Days':<15} {'Train Segments':<20} {'Test Segments':<20}")
    print("-" * 70)
    for regime in ['bull', 'bear', 'sideways']:
        total_days = regime_stats.get(regime, 0)
        n_train = len(regime_data[regime]['train'])
        n_test = len(regime_data[regime]['test'])
        print(f"{regime.capitalize():<15} {total_days:<15} {n_train:<20} {n_test:<20}")
    
    print(f"\nNeutral days (insufficient data): {regime_stats.get('neutral', 0)}")
    
    return regime_data


REGIME_CONFIGS = {
    'bull': {
        'name': 'Bull Market Strategy',
        'description': '牛市策略：持有为主，让利润奔跑',
        'min_hold_days': 5,
        'optimal_hold_days': 90,
        'position_size': 0.90,
        'profit_threshold': 0.15,
        'stop_loss': -0.08,
        'trade_cost_penalty': 0.01,
        'profit_reward_scale': 120,
        'future_reward_scale': 40,
        'final_reward_scale': 40,
        'equity_scale': 30,
        'overtrade_penalty': 0.2,
        'min_invest_ratio': 0.5,
        'expected_trades': '3-5次/年',
    },
    'bear': {
        'name': 'Bear Market Strategy',
        'description': '熊市策略：保守防御，快速止损',
        'min_hold_days': 2,
        'optimal_hold_days': 20,
        'position_size': 0.30,
        'profit_threshold': 0.05,
        'stop_loss': -0.03,
        'trade_cost_penalty': 0.005,
        'profit_reward_scale': 200,
        'future_reward_scale': 30,
        'final_reward_scale': 5,
        'expected_trades': '5-15次/年',
    },
    'sideways': {
        'name': 'Sideways Market Strategy',
        'description': '震荡市策略：高抛低吸，频繁交易',
        'min_hold_days': 3,
        'optimal_hold_days': 40,
        'position_size': 0.60,
        'profit_threshold': 0.08,
        'stop_loss': -0.05,
        'trade_cost_penalty': 0.0,
        'profit_reward_scale': 120,
        'future_reward_scale': 60,
        'final_reward_scale': 10,
        'expected_trades': '15-30次/年',
    }
}


def load_data():
    pattern = DATA_DIR + "/*_features_deep_clean.csv"
    data_files = glob.glob(pattern)
    print(f"📂 Found {len(data_files)} stock files")

    all_dfs = []
    stock_names = []

    for file in data_files:
        df = pd.read_csv(file)
        df = df.sort_values("Date").reset_index(drop=True)
        
        name = file.split("/")[-1].replace("_features_deep_clean.csv", "")
        stock_names.append(name)
        
        df = add_trading_signals(df)
        all_dfs.append(df)

    common_cols = set(all_dfs[0].columns)
    for df in all_dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    
    common_cols = [c for c in common_cols if c not in ["Date", "Close", "market_regime"]]
    
    print(f"📊 Common features: {len(common_cols)}")
    
    return all_dfs, stock_names, common_cols


def add_trading_signals(df):
    df = df.copy()
    
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)
    
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_60'] = df['Close'].rolling(60).mean()
    
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()
    
    df['price_vs_ma5'] = df['Close'] / df['ma_5'] - 1
    df['price_vs_ma20'] = df['Close'] / df['ma_20'] - 1
    df['price_vs_ma60'] = df['Close'] / df['ma_60'] - 1
    
    df['trend_50_200'] = (df['ma_50'] - df['ma_200']) / df['ma_200']
    
    df['trend_5_20'] = (df['ma_5'] - df['ma_20']) / df['ma_20']
    df['trend_20_60'] = (df['ma_20'] - df['ma_60']) / df['ma_60']
    
    df['volatility_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['volatility_60'] = df['Close'].pct_change().rolling(60).std() * np.sqrt(252)
    
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    df['strength'] = df['Close'].rolling(10).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
    )
    
    return df.fillna(0)


class RegimeSpecificEnv(gym.Env):
    def __init__(self, df_list, feature_cols, regime_config,
                 cash=CASH, commission=COMMISSION, window=WINDOW):
        super().__init__()

        self.df_list = df_list
        self.feature_cols = feature_cols
        self.window = window
        self.commission = commission
        self.initial_cash = cash
        self.config = regime_config

        self.is_bull = (self.config.get('name', '') == 'Bull Market Strategy')

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
        self.position = 0
        self.trade_count = 0
        self.last_action = 0

        self.last_buy_price = 0
        self.hold_days = 0
        self.profitable_trades = 0
        self.total_trades = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        start = max(0, self.step_idx - self.window + 1)
        hist = self.features[start:self.step_idx + 1]

        if len(hist) < self.window:
            pad = np.tile(self.features[0], (self.window - len(hist), 1))
            hist = np.vstack([pad, hist])

        return np.concatenate([
            hist.flatten(),
            [self.cash / self.initial_cash],
            [self.position * self.prices[self.step_idx] / self.initial_cash],
            [self.last_action],
            [self.hold_days / 100.0]
        ])

    def step(self, action):
        price = self.prices[self.step_idx]
        old_equity = self.cash + self.position * price

        trade_occurred = False

        if self.position > 0:
            self.hold_days += 1

        if action == 1 and self.cash > 0:
            self.position = (self.cash * (1 - self.commission)) / price
            self.cash = 0
            self.last_buy_price = price
            self.hold_days = 0
            self.trade_count += 1
            self.total_trades += 1
            trade_occurred = True

        elif action == 2 and self.position > 0:
            if self.hold_days >= self.config['min_hold_days']:
                self.cash = self.position * price * (1 - self.commission)

                if price > self.last_buy_price:
                    self.profitable_trades += 1

                self.position = 0
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

        terminated = done
        truncated = False
        info = {
            "equity": new_equity,
            "trade_count": self.trade_count,
            "profitable_trades": self.profitable_trades,
            "total_trades": self.total_trades
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_reward(self, action, trade_occurred, old_equity, new_equity, old_price, new_price):
        reward = 0.0

        equity_change = (new_equity - old_equity) / (old_equity + 1e-8)

        if self.is_bull:
            equity_scale = self.config.get('equity_scale', 30.0)
        else:
            equity_scale = 100.0

        reward += equity_change * equity_scale

        if trade_occurred:
            penalty = self.config['trade_cost_penalty']

            if self.is_bull and self.total_trades > 3:
                penalty += self.config.get('overtrade_penalty', 0.2)

            reward -= penalty

        if trade_occurred:
            if action == 1:
                future_idx = min(self.step_idx + 5, self.max_step)
                future_price = self.prices[future_idx]
                future_return = (future_price - old_price) / old_price
                reward += future_return * self.config['future_reward_scale']

            elif action == 2:
                if old_price > self.last_buy_price:
                    profit = (old_price - self.last_buy_price) / (self.last_buy_price + 1e-8)
                    reward += profit * self.config['profit_reward_scale']
                else:
                    loss = (old_price - self.last_buy_price) / (self.last_buy_price + 1e-8)
                    if loss > -0.05:
                        reward += 0.05
                    else:
                        reward -= 0.3

        position_value = self.position * new_price
        position_ratio = position_value / (new_equity + 1e-8)

        if self.is_bull:
            min_invest = self.config.get('min_invest_ratio', 0.5)
            if position_ratio < min_invest:
                reward -= 0.1
            else:
                reward += 0.02
        else:
            if position_ratio < 0.05 or position_ratio > 0.95:
                reward -= 0.05

        if self.step_idx >= 5:
            recent_trend = (new_price - self.prices[self.step_idx - 5]) / (self.prices[self.step_idx - 5] + 1e-8)

            if position_ratio > 0.5 and recent_trend > 0:
                reward += 0.1
            elif position_ratio < 0.5 and recent_trend < 0:
                reward += 0.1

            if self.is_bull and self.position > 0 and action == 0 and recent_trend > 0:
                reward += 0.05

        if self.config['name'] == 'Bear Market Strategy':
            reward = np.clip(reward, -8, 8)
        elif self.config['name'] == 'Bull Market Strategy':
            reward = np.clip(reward, -12, 12)
        else:
            reward = np.clip(reward, -10, 10)

        return reward


def train_regime_models(regime_data, common_cols):
    models = {}
    
    for regime in ['bull', 'bear', 'sideways']:
        train_segments = regime_data[regime]['train']
        
        if len(train_segments) < 10:
            print(f"\n⚠️  Skipping {regime}: insufficient training data")
            continue
        
        config = REGIME_CONFIGS[regime]
        
        print(f"\n{'='*70}")
        print(f"🚀 Training {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Training segments: {len(train_segments)}")
        print(f"   Expected trades: {config['expected_trades']}")
        print(f"{'='*70}")
        
        n_envs = 4
        train_env = SubprocVecEnv([
            (lambda cfg=config, data=train_segments: 
                RegimeSpecificEnv(data, common_cols, cfg))
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
        model.save(f"ppo_regime_{regime}")
        
        models[regime] = {
            "model": model,
            "config": config
        }
        
        train_env.close()
        
        print(f"✅ Completed training for {regime} market")
    
    return models


def evaluate_regime_models(models, regime_data, common_cols):
    results = []
    
    for regime in ['bull', 'bear', 'sideways']:
        if regime not in models:
            continue
        
        model = models[regime]['model']
        config = models[regime]['config']
        test_segments = regime_data[regime]['test']
        
        print(f"\n📊 Evaluating {config['name']} on {len(test_segments)} test segments...")
        
        for i, test_df in enumerate(test_segments):
            env = RegimeSpecificEnv([test_df], common_cols, config)
            obs, info = env.reset()
            done = False
            
            equity_curve = []
            actions_log = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                equity_curve.append(info["equity"])
                actions_log.append(action)
            
            final_equity = info["equity"]
            ret_pct = (final_equity - CASH) / CASH * 100
            trades = info["trade_count"]
            profitable = info.get("profitable_trades", 0)
            total = info.get("total_trades", 0)
            win_rate = (profitable / total * 100) if total > 0 else 0
            
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdown = np.min(drawdown) * 100
            
            results.append({
                'regime': regime,
                'regime_name': config['name'],
                'segment_id': i,
                'return_pct': ret_pct,
                'num_trades': trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'final_equity': final_equity,
                'num_buy': actions_log.count(1),
                'num_sell': actions_log.count(2),
                'num_hold': actions_log.count(0),
            })
            
            if i < 3:
                print(f"  Segment {i}: Return={ret_pct:6.2f}%, Trades={trades:3d}, WinRate={win_rate:.1f}%")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("📂 Loading all stock data...")
    all_dfs, stock_names, common_cols = load_data()
    
    print("\n🔍 Splitting data by market regime...")
    regime_data = split_by_regime(all_dfs, stock_names, train_ratio=TRAIN_RATIO)
    
    print("\n🎯 Training regime-specific models...")
    models = train_regime_models(regime_data, common_cols)
    
    print("\n📈 Evaluating models on test data...")
    results_df = evaluate_regime_models(models, regime_data, common_cols)
    
    results_df.to_csv("regime_based_results.csv", index=False)
    
    print("\n" + "="*80)
    print("📊 REGIME-BASED TRADING RESULTS")
    print("="*80)
    
    for regime in ['bull', 'bear', 'sideways']:
        regime_results = results_df[results_df['regime'] == regime]
        if len(regime_results) == 0:
            continue
        
        print(f"\n{regime.upper()} MARKET:")
        print(f"  Test Segments:    {len(regime_results)}")
        print(f"  Avg Return:       {regime_results['return_pct'].mean():6.2f}%")
        print(f"  Median Return:    {regime_results['return_pct'].median():6.2f}%")
        print(f"  Best Return:      {regime_results['return_pct'].max():6.2f}%")
        print(f"  Worst Return:     {regime_results['return_pct'].min():6.2f}%")
        print(f"  Avg Trades:       {regime_results['num_trades'].mean():6.1f}")
        print(f"  Avg Win Rate:     {regime_results['win_rate'].mean():6.1f}%")
        print(f"  Avg Sharpe:       {regime_results['sharpe_ratio'].mean():6.2f}")
        print(f"  Avg Max DD:       {regime_results['max_drawdown'].mean():6.2f}%")
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE:")
    print(f"  Average Return:   {results_df['return_pct'].mean():6.2f}%")
    print(f"  Average Trades:   {results_df['num_trades'].mean():6.1f}")
    print(f"  Average Win Rate: {results_df['win_rate'].mean():6.1f}%")
    print(f"  Average Sharpe:   {results_df['sharpe_ratio'].mean():6.2f}")
    
    print("\n✅ Results saved to: regime_based_results.csv")
    print("✅ Models saved: ppo_regime_bull.zip, ppo_regime_bear.zip, ppo_regime_sideways.zip")
