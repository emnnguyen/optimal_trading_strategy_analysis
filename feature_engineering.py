"""
Feature Engineering for Trading Strategy
=========================================
Creates technical indicators and features from OHLCV data for ML trading strategies.

Usage:
    python feature_engineering.py

Or import as module:
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_with_features = fe.create_features(df)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Generate technical indicators and features from OHLCV data.
    
    Features created:
    - Price momentum (multiple timeframes)
    - Moving averages (SMA, EMA)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Volume features
    - Volatility measures
    - Price patterns and ratios
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, df, drop_na=True):
        """
        Create all features from OHLCV dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: Open, High, Low, Close, Volume
            Index should be DatetimeIndex
        drop_na : bool
            Whether to drop NaN rows (from rolling calculations)
            
        Returns:
        --------
        pd.DataFrame with original data + engineered features
        """
        df = df.copy()
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        print("Creating features...")
        
        # 1. Price-based features
        df = self._create_price_features(df)
        
        # 2. Moving averages
        df = self._create_moving_averages(df)
        
        # 3. Momentum indicators
        df = self._create_momentum_indicators(df)
        
        # 4. Volatility features
        df = self._create_volatility_features(df)
        
        # 5. Volume features
        df = self._create_volume_features(df)
        
        # 6. Pattern recognition features
        df = self._create_pattern_features(df)
        
        # 7. Trend indicators
        df = self._create_trend_indicators(df)
        
        if drop_na:
            initial_rows = len(df)
            df = df.dropna()
            print(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        print(f"Total features created: {len(self.feature_names)}")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    
    def _create_price_features(self, df):
        """Basic price-based features"""
        print("  - Price features...")
        
        # Returns over multiple periods
        for period in [1, 2, 3, 5, 10, 20, 30]:
            df[f'returns_{period}d'] = df['Close'].pct_change(period)
            self.feature_names.append(f'returns_{period}d')
        
        # Log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        self.feature_names.append('log_returns')
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        self.feature_names.extend(['high_low_ratio', 'close_open_ratio'])
        
        # Intraday range
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        self.feature_names.append('daily_range')
        
        # Gap (today's open vs yesterday's close)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        self.feature_names.append('gap')
        
        return df
    
    def _create_moving_averages(self, df):
        """Moving average features"""
        print("  - Moving averages...")
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']
            self.feature_names.extend([f'sma_{window}', f'price_to_sma_{window}'])
        
        # Exponential Moving Averages
        for span in [12, 26, 50]:
            df[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            df[f'price_to_ema_{span}'] = df['Close'] / df[f'ema_{span}']
            self.feature_names.extend([f'ema_{span}', f'price_to_ema_{span}'])
        
        # Moving average crossovers
        df['sma_20_50_cross'] = df['sma_20'] / df['sma_50']
        df['sma_50_200_cross'] = df['sma_50'] / df['sma_200']
        self.feature_names.extend(['sma_20_50_cross', 'sma_50_200_cross'])
        
        return df
    
    def _create_momentum_indicators(self, df):
        """Momentum-based technical indicators"""
        print("  - Momentum indicators...")
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['Close'], 7)
        self.feature_names.extend(['rsi_14', 'rsi_7'])
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        self.feature_names.extend(['macd', 'macd_signal', 'macd_diff'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['stochastic_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
        self.feature_names.extend(['stochastic_k', 'stochastic_d'])
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        self.feature_names.append('williams_r')
        
        # Rate of Change (ROC)
        for period in [10, 20]:
            df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                    df['Close'].shift(period) * 100)
            self.feature_names.append(f'roc_{period}')
        
        # Money Flow Index (MFI)
        df['mfi'] = self._calculate_mfi(df)
        self.feature_names.append('mfi')
        
        return df
    
    def _create_volatility_features(self, df):
        """Volatility and risk measures"""
        print("  - Volatility features...")
        
        # Historical volatility (rolling std of returns)
        for window in [10, 20, 30, 60]:
            df[f'volatility_{window}'] = df['returns_1d'].rolling(window=window).std()
            self.feature_names.append(f'volatility_{window}')
        
        # Bollinger Bands
        for window in [20]:
            sma = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_middle_{window}'] = sma
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
            df[f'bb_position_{window}'] = (df['Close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            self.feature_names.extend([f'bb_upper_{window}', f'bb_lower_{window}', 
                                      f'bb_width_{window}', f'bb_position_{window}'])
        
        # Average True Range (ATR)
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_ratio'] = df['atr_14'] / df['Close']
        self.feature_names.extend(['atr_14', 'atr_ratio'])
        
        # Parkinson volatility (uses high-low)
        df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(df['High']/df['Low'])**2)
        self.feature_names.append('parkinson_vol')
        
        return df
    
    def _create_volume_features(self, df):
        """Volume-based features"""
        print("  - Volume features...")
        
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_sma_{window}']
            self.feature_names.extend([f'volume_sma_{window}', f'volume_ratio_{window}'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['Volume'].pct_change(5)
        self.feature_names.append('volume_roc')
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ema']
        self.feature_names.extend(['obv', 'obv_ratio'])
        
        # Volume-Weighted Average Price (VWAP) approximation
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['price_to_vwap'] = df['Close'] / df['vwap']
        self.feature_names.extend(['vwap', 'price_to_vwap'])
        
        # Force Index
        df['force_index'] = df['Close'].diff() * df['Volume']
        df['force_index_ema'] = df['force_index'].ewm(span=13, adjust=False).mean()
        self.feature_names.append('force_index_ema')
        
        return df
    
    def _create_pattern_features(self, df):
        """Price pattern recognition features"""
        print("  - Pattern features...")
        
        # Distance from 52-week high/low
        df['dist_from_52w_high'] = (df['High'].rolling(252).max() - df['Close']) / df['Close']
        df['dist_from_52w_low'] = (df['Close'] - df['Low'].rolling(252).min()) / df['Close']
        self.feature_names.extend(['dist_from_52w_high', 'dist_from_52w_low'])
        
        # New highs/lows
        df['is_52w_high'] = (df['Close'] == df['High'].rolling(252).max()).astype(int)
        df['is_52w_low'] = (df['Close'] == df['Low'].rolling(252).min()).astype(int)
        self.feature_names.extend(['is_52w_high', 'is_52w_low'])
        
        # Consecutive up/down days
        df['consecutive_up'] = self._count_consecutive(df['Close'].diff() > 0)
        df['consecutive_down'] = self._count_consecutive(df['Close'].diff() < 0)
        self.feature_names.extend(['consecutive_up', 'consecutive_down'])
        
        # Price channels
        df['donchian_upper'] = df['High'].rolling(20).max()
        df['donchian_lower'] = df['Low'].rolling(20).min()
        df['donchian_position'] = (df['Close'] - df['donchian_lower']) / (df['donchian_upper'] - df['donchian_lower'])
        self.feature_names.append('donchian_position')
        
        return df
    
    def _create_trend_indicators(self, df):
        """Trend strength and direction indicators"""
        print("  - Trend indicators...")
        
        # ADX (Average Directional Index) - simplified version
        df['adx'] = self._calculate_adx(df, 14)
        self.feature_names.append('adx')
        
        # Aroon Indicator
        for window in [25]:
            df[f'aroon_up_{window}'] = df['High'].rolling(window).apply(
                lambda x: (window - x.argmax()) / window * 100, raw=True)
            df[f'aroon_down_{window}'] = df['Low'].rolling(window).apply(
                lambda x: (window - x.argmin()) / window * 100, raw=True)
            df[f'aroon_oscillator_{window}'] = df[f'aroon_up_{window}'] - df[f'aroon_down_{window}']
            self.feature_names.append(f'aroon_oscillator_{window}')
        
        # Linear regression slope (trend strength)
        for window in [10, 20]:
            df[f'lr_slope_{window}'] = df['Close'].rolling(window).apply(
                lambda x: self._calculate_slope(x), raw=False)
            self.feature_names.append(f'lr_slope_{window}')
        
        # R-squared of linear regression (trend consistency)
        df['lr_r2_20'] = df['Close'].rolling(20).apply(
            lambda x: self._calculate_r2(x), raw=False)
        self.feature_names.append('lr_r2_20')
        
        return df
    
    # ========== Helper Functions ==========
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index (simplified)"""
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = self._calculate_atr(df, period)
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _count_consecutive(self, series):
        """Count consecutive True values"""
        cumsum = series.cumsum()
        reset = cumsum.where(~series, 0)
        return cumsum - reset.ffill().fillna(0)
    
    def _calculate_slope(self, y):
        """Calculate linear regression slope"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_r2(self, y):
        """Calculate R-squared of linear regression"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r2
    
    def get_feature_names(self):
        """Return list of all created feature names"""
        return self.feature_names


# ========== Convenience Functions ==========

def load_and_engineer_features(filepath, date_column='Date', drop_na=True):
    """
    Load data and create features in one step.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with OHLCV data
    date_column : str
        Name of date column
    drop_na : bool
        Whether to drop NaN rows
        
    Returns:
    --------
    pd.DataFrame with features
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert date to datetime and set as index
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    
    # Sort by date
    df = df.sort_index()
    
    # Create features
    fe = FeatureEngineer()
    df_with_features = fe.create_features(df, drop_na=drop_na)
    
    return df_with_features


def process_multiple_stocks(stock_data_dict, drop_na=True):
    """
    Process multiple stocks and create features for each.
    
    Parameters:
    -----------
    stock_data_dict : dict
        Dictionary of {stock_name: dataframe} with OHLCV data
    drop_na : bool
        Whether to drop NaN rows
        
    Returns:
    --------
    dict of {stock_name: dataframe_with_features}
    """
    results = {}
    fe = FeatureEngineer()
    
    for stock_name, df in stock_data_dict.items():
        print(f"\nProcessing {stock_name}...")
        df = df.copy()
        df = df.sort_index()
        results[stock_name] = fe.create_features(df, drop_na=drop_na)
    
    return results


# ========== Example Usage ==========

if __name__ == "__main__":
    """
    Example usage of the feature engineering module.
    """
    
    # Example 1: Create sample data
    print("="*60)
    print("FEATURE ENGINEERING EXAMPLE")
    print("="*60)
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Simulate price data
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    
    sample_data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.randn(n) * 0.01),
        'High': close_prices * (1 + np.abs(np.random.randn(n) * 0.02)),
        'Low': close_prices * (1 - np.abs(np.random.randn(n) * 0.02)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    print("\nSample data (first 5 rows):")
    print(sample_data.head())
    
    # Example 2: Create features
    print("\n" + "="*60)
    print("Creating features...")
    print("="*60)
    
    fe = FeatureEngineer()
    df_with_features = fe.create_features(sample_data, drop_na=True)
    
    print("\nData with features (first 5 rows, selected columns):")
    selected_cols = ['Close', 'returns_1d', 'sma_20', 'rsi_14', 'macd', 'bb_position_20', 'volume_ratio_20']
    print(df_with_features[selected_cols].head())
    
    print("\n" + "="*60)
    print("Feature Summary:")
    print("="*60)
    print(f"Total features created: {len(fe.get_feature_names())}")
    print(f"Original shape: {sample_data.shape}")
    print(f"Final shape: {df_with_features.shape}")
    
    print("\nAll feature names:")
    for i, feature in enumerate(fe.get_feature_names(), 1):
        print(f"{i:3d}. {feature}")
    
    print("\n" + "="*60)
    print("Feature statistics:")
    print("="*60)
    print(df_with_features[fe.get_feature_names()[:10]].describe())
    
    # Example 3: Save to file
    output_file = 'sample_data_with_features.csv'
    df_with_features.to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("""
    # Method 1: Use FeatureEngineer class
    from feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    df_with_features = fe.create_features(your_df)
    
    # Method 2: Load CSV and process in one step
    from feature_engineering import load_and_engineer_features
    
    df_with_features = load_and_engineer_features('your_data.csv')
    
    # Method 3: Process multiple stocks
    from feature_engineering import process_multiple_stocks
    
    stocks = {
        'AAPL': aapl_df,
        'MSFT': msft_df,
        'GOOGL': googl_df
    }
    results = process_multiple_stocks(stocks)
    """)
