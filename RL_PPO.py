import os 
import pandas as pd
import numpy as np
import glob
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

TRAIN_RATIO = 0.7
CASH = 10_000
COMMISSION = 0.001
WINDOW = 10
DATA_DIR = "/Users/vanessaliu/Desktop/STA160/featurecompany"


def add_features(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close * 1.01
    low = df["Low"] if "Low" in df.columns else close * 0.99

    for d in [5, 10, 20, 60, 120]:
        df[f"ret_{d}d"] = close.pct_change(d)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi14"] = 100 - (100 / (1 + rs))

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-8)
    df["bb_pos"] = (close - mid) / (2 * std + 1e-8)

    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = true_range.rolling(14).mean()

    exclude_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    feat_cols = [c for c in df.columns if c not in exclude_cols]

    for col in feat_cols:
        series = df[col]
        mean = series.mean()
        std = series.std()
        if std > 1e-8:
            df[col] = (series - mean) / std
        else:
            df[col] = 0.0

    df = df.fillna(0)
    return df


def load_data():
    pattern = DATA_DIR + "/*_features.csv"
    data_files = glob.glob(pattern)
    print(f"Found {len(data_files)} stocks")

    dfs = []
    stock_names = []

    for file in data_files:
        df = pd.read_csv(file)
        df = df.sort_values("Date").reset_index(drop=True)
        df = add_features(df)
        dfs.append(df)
        name = file.split("/")[-1].replace("_features_deep_clean.csv", "")
        stock_names.append(name)

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    common_cols = [c for c in common_cols if c not in ["Date", "Close"]]
    print("Common feature columns:", len(common_cols))

    all_train_data = []
    all_test_data = []

    for df in dfs:
        df2 = df[["Date", "Close"] + common_cols].copy()
        n_train = int(len(df2) * TRAIN_RATIO)
        train_data = df2.iloc[:n_train].reset_index(drop=True)
        test_data = df2.iloc[n_train:].reset_index(drop=True)
        all_train_data.append(train_data)
        all_test_data.append(test_data)

    return all_train_data, all_test_data, stock_names, common_cols


class FastStockEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df_list, feature_cols, cash=10000, commission=0.001, window=10):
        super().__init__()
        self.df_list = df_list
        self.feature_cols = feature_cols
        self.window = window
        self.commission = commission
        self.initial_cash = cash

        obs_dim = len(feature_cols) * window + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)
        self.rng = np.random.default_rng()
        self.reset(seed=None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_stock_idx = self.rng.integers(0, len(self.df_list))
        self.df = self.df_list[self.current_stock_idx].reset_index(drop=True)

        self.features = self.df[self.feature_cols].values
        self.prices = self.df["Close"].values
        self.max_step = len(self.df) - 1
        self.step_idx = self.window - 1

        self.cash = self.initial_cash
        self.position = 0.0
        self.trade_count = 0
        self.days_since_trade = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        start = max(0, self.step_idx - self.window + 1)
        hist = self.features[start:self.step_idx + 1]

        if len(hist) < self.window:
            pad = np.tile(self.features[0], (self.window - len(hist), 1))
            hist = np.vstack([pad, hist])

        return np.concatenate([
            hist.flatten(),
            [self.cash / self.initial_cash],
            [(self.position * self.prices[self.step_idx]) / self.initial_cash],
            [min(self.days_since_trade / 10.0, 1.0)]
        ]).astype(np.float32)

    def step(self, action):
        price = self.prices[self.step_idx]
        old_equity = self.cash + self.position * price

        trade_happened = False

        if action == 1:
            buy_amount = self.cash * 0.20
            if buy_amount > 10:
                shares = (buy_amount * (1 - self.commission)) / price
                self.position += shares
                self.cash -= buy_amount
                trade_happened = True

        elif action == 2:
            buy_amount = self.cash * 0.40
            if buy_amount > 10:
                shares = (buy_amount * (1 - self.commission)) / price
                self.position += shares
                self.cash -= buy_amount
                trade_happened = True

        elif action == 3:
            sell_shares = self.position * 0.20
            if sell_shares * price > 10:
                sell_value = sell_shares * price * (1 - self.commission)
                self.position -= sell_shares
                self.cash += sell_value
                trade_happened = True

        elif action == 4:
            sell_shares = self.position * 0.40
            if sell_shares * price > 10:
                sell_value = sell_shares * price * (1 - self.commission)
                self.position -= sell_shares
                self.cash += sell_value
                trade_happened = True

        if trade_happened:
            self.trade_count += 1
            self.days_since_trade = 0
        else:
            self.days_since_trade += 1

        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.max_step

        new_price = self.prices[self.step_idx]
        new_equity = self.cash + self.position * new_price

        profit_reward = (new_equity - old_equity) / old_equity
        profit_reward = np.clip(profit_reward, -0.1, 0.1)

        trade_bonus = 0.002 if trade_happened else 0.0
        inactivity_penalty = -0.001 * max(0, self.days_since_trade - 5)

        position_ratio = (self.position * new_price) / new_equity if new_equity > 0 else 0
        position_bonus = 0.0005 if 0.3 < position_ratio < 0.8 else 0.0

        reward = profit_reward + trade_bonus + inactivity_penalty + position_bonus

        obs = self._get_obs()
        info = {
            "equity": new_equity,
            "trade_count": self.trade_count
        }

        return obs, reward, terminated, truncated, info



def train_rl_model(all_train_data, common_cols):
    print("=" * 80)
    print("TRAINING RL MODEL")
    print("=" * 80)

    train_env = DummyVecEnv([
        (lambda cols=common_cols: FastStockEnv(all_train_data, cols))
    ])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=128,
        n_steps=256,
        learning_rate=3e-4,
        n_epochs=10,
        ent_coef=0.05,
        clip_range=0.2,
        gamma=0.99
    )

    model.learn(total_timesteps=100_000)

   
    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_fast_multi_stock"
    model.save(model_path)   
    print(f"✅ Saved model to {model_path}.zip")

    train_env.close()
    return model



def test_rl_model(model, test_df, common_cols, stock_name):
    env = FastStockEnv([test_df], common_cols)
    obs, info = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    final_equity = info["equity"]
    ret_pct = (final_equity - CASH) / CASH * 100
    trades = info["trade_count"]

    print(f"{stock_name}: return={ret_pct:.2f}%, trades={trades}, final_equity={final_equity:.2f}")

    return {
        "stock_name": stock_name,
        "return_pct": ret_pct,
        "num_trades": trades,
        "final_equity": final_equity
    }


if __name__ == "__main__":
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    all_train_data, all_test_data, stock_names, common_cols = load_data()

    print("=" * 80)
    print("STEP 2: TRAINING RL MODEL")
    print("=" * 80)
    rl_model = train_rl_model(all_train_data, common_cols)

    print("=" * 80)
    print("STEP 3: TESTING RL MODEL ON TEST SETS")
    print("=" * 80)
    rl_results = []
    for stock_name, test_df in zip(stock_names, all_test_data):
        result = test_rl_model(rl_model, test_df, common_cols, stock_name)
        rl_results.append(result)

    df_results = pd.DataFrame(rl_results)
    df_results.to_csv("rl_results_only.csv", index=False)
    print("Saved RL-only results to rl_results_only.csv")
