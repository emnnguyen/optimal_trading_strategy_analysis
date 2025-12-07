"""
Utility Functions for Trading Transformer System

This module provides helper functions for:
- Data loading from CSV files
- Data preprocessing and normalization
- Sequence creation for transformer input
- Label generation for training
- Position sizing calculations
- Risk management utilities

Author: Trading ML System
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Constants and Configuration
# ============================================================================

# Signal class indices
SIGNAL_BUY = 0
SIGNAL_HOLD = 1
SIGNAL_SELL = 2

# Signal class names
SIGNAL_NAMES = {
    SIGNAL_BUY: 'BUY',
    SIGNAL_HOLD: 'HOLD',
    SIGNAL_SELL: 'SELL'
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_single_stock_csv(
    filepath: str,
    date_column: str = 'date',
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load a single stock CSV file.
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        required_columns: Optional list of required columns to validate
        
    Returns:
        DataFrame with stock data, indexed by date
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Parse dates
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    
    # Sort by date
    df.sort_index(inplace=True)
    
    # Validate required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return df


def load_multiple_stocks(
    data_dir: str,
    pattern: str = "*.csv",
    date_column: str = 'date'
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV files for multiple stocks from a directory.
    
    Args:
        data_dir: Directory containing CSV files
        pattern: Glob pattern for CSV files
        date_column: Name of date column
        
    Returns:
        Dictionary mapping stock symbol to DataFrame
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_files = glob.glob(os.path.join(data_dir, pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir} matching {pattern}")
    
    stocks = {}
    for filepath in csv_files:
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            df = load_single_stock_csv(filepath, date_column)
            stocks[symbol] = df
            print(f"Loaded {symbol}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    print(f"\nLoaded {len(stocks)} stocks successfully")
    return stocks


# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def get_feature_columns(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
    """
    Get list of feature columns from DataFrame.
    
    Args:
        df: Input DataFrame
        exclude_columns: Columns to exclude (e.g., OHLCV for label creation)
        
    Returns:
        List of feature column names
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Common non-feature columns to exclude
    default_exclude = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol', 'label']
    all_exclude = set(exclude_columns + default_exclude)
    
    feature_cols = [col for col in df.columns if col.lower() not in all_exclude]
    return feature_cols


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill_drop') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        method: Method to use - 'ffill_drop' (forward fill then drop remaining NaNs),
                'ffill' (only forward fill), 'drop' (only drop NaNs)
                
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == 'ffill_drop':
        df = df.ffill()
        df = df.dropna()
    elif method == 'ffill':
        df = df.ffill()
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def create_labels(
    df: pd.DataFrame,
    close_column: str = 'close',
    forward_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02
) -> pd.Series:
    """
    Create trading labels based on forward returns.
    
    Args:
        df: DataFrame with price data
        close_column: Name of close price column
        forward_days: Number of days to look ahead for returns
        buy_threshold: Return threshold for BUY signal (default 2%)
        sell_threshold: Return threshold for SELL signal (default -2%)
        
    Returns:
        Series with labels (0=BUY, 1=HOLD, 2=SELL)
    """
    if close_column not in df.columns:
        raise ValueError(f"Close column '{close_column}' not found in DataFrame")
    
    # Calculate forward returns
    forward_returns = df[close_column].pct_change(periods=forward_days).shift(-forward_days)
    
    # Create labels
    labels = pd.Series(index=df.index, dtype=int)
    labels[:] = SIGNAL_HOLD  # Default to HOLD
    labels[forward_returns > buy_threshold] = SIGNAL_BUY
    labels[forward_returns < sell_threshold] = SIGNAL_SELL
    
    return labels


def fit_scalers(
    stocks_data: Dict[str, pd.DataFrame],
    feature_columns: List[str]
) -> StandardScaler:
    """
    Fit a StandardScaler on training data from all stocks.
    
    Args:
        stocks_data: Dictionary of stock DataFrames
        feature_columns: List of feature column names
        
    Returns:
        Fitted StandardScaler
    """
    # Concatenate all feature data
    all_features = []
    for symbol, df in stocks_data.items():
        features = df[feature_columns].values
        all_features.append(features)
    
    all_features = np.vstack(all_features)
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    return scaler


def normalize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Normalize features using a fitted scaler.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names to normalize
        scaler: Fitted StandardScaler
        
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    df[feature_columns] = scaler.transform(df[feature_columns].values)
    return df


# ============================================================================
# Sequence Creation Functions
# ============================================================================

def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    seq_length: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for transformer input.
    
    Args:
        features: Feature array of shape (num_samples, num_features)
        labels: Label array of shape (num_samples,)
        seq_length: Length of each sequence
        
    Returns:
        Tuple of (sequences, sequence_labels)
        - sequences: shape (num_sequences, seq_length, num_features)
        - sequence_labels: shape (num_sequences,)
    """
    num_samples = len(features)
    
    if num_samples < seq_length:
        raise ValueError(f"Not enough samples ({num_samples}) for sequence length ({seq_length})")
    
    sequences = []
    sequence_labels = []
    
    for i in range(seq_length, num_samples):
        # Get sequence of features leading up to this point
        seq = features[i - seq_length:i]
        # Label is for the current day (predicting based on past seq_length days)
        label = labels[i]
        
        if not np.isnan(label):
            sequences.append(seq)
            sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)


def prepare_stock_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: StandardScaler,
    seq_length: int = 40,
    create_label: bool = True,
    close_column: str = 'close'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare a single stock's data for training/inference.
    
    Args:
        df: Stock DataFrame
        feature_columns: List of feature columns
        scaler: Fitted StandardScaler
        seq_length: Sequence length
        create_label: Whether to create labels (True for training, False for inference)
        close_column: Name of close price column
        
    Returns:
        Tuple of (sequences, labels)
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create labels if needed
    if create_label:
        labels = create_labels(df, close_column=close_column)
    else:
        labels = np.zeros(len(df))  # Dummy labels for inference
    
    # Normalize features
    df_normalized = normalize_features(df, feature_columns, scaler)
    features = df_normalized[feature_columns].values
    
    # Create sequences
    sequences, seq_labels = create_sequences(features, labels.values, seq_length)
    
    return sequences, seq_labels


def prepare_all_stocks(
    stocks_data: Dict[str, pd.DataFrame],
    feature_columns: List[str],
    scaler: StandardScaler,
    seq_length: int = 40,
    close_column: str = 'close'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data from all stocks for training.
    
    Args:
        stocks_data: Dictionary of stock DataFrames
        feature_columns: List of feature columns
        scaler: Fitted StandardScaler
        seq_length: Sequence length
        close_column: Name of close price column
        
    Returns:
        Tuple of (all_sequences, all_labels)
    """
    all_sequences = []
    all_labels = []
    
    for symbol, df in stocks_data.items():
        try:
            sequences, labels = prepare_stock_data(
                df, feature_columns, scaler, seq_length, 
                create_label=True, close_column=close_column
            )
            all_sequences.append(sequences)
            all_labels.append(labels)
            print(f"  {symbol}: {len(sequences)} sequences")
        except Exception as e:
            print(f"  Warning: Failed to process {symbol}: {e}")
    
    return np.vstack(all_sequences), np.concatenate(all_labels)


# ============================================================================
# Dataset Classes
# ============================================================================

class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading sequences.
    
    Args:
        sequences: Feature sequences of shape (num_samples, seq_length, num_features)
        labels: Labels of shape (num_samples,)
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights (inverse frequency)
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Compute inverse frequency weights
    weights = total / (len(unique) * counts)
    
    # Normalize weights
    weights = weights / weights.sum() * len(unique)
    
    return torch.FloatTensor(weights)


# ============================================================================
# Position Sizing Functions
# ============================================================================

def calculate_realized_volatility(
    returns: Union[pd.Series, np.ndarray],
    window: int = 20,
    annualization_factor: float = 252.0
) -> Union[pd.Series, float]:
    """
    Calculate realized volatility (rolling standard deviation of returns).
    
    Args:
        returns: Daily returns series
        window: Rolling window size (default 20 days)
        annualization_factor: Factor to annualize volatility (default 252 trading days)
        
    Returns:
        Annualized rolling volatility
    """
    if isinstance(returns, pd.Series):
        rolling_std = returns.rolling(window=window).std()
        return rolling_std * np.sqrt(annualization_factor)
    else:
        std = np.std(returns[-window:]) if len(returns) >= window else np.std(returns)
        return std * np.sqrt(annualization_factor)


def calculate_position_size(
    realized_volatility: float,
    target_volatility: float = 0.20,
    max_position: float = 1.0,
    min_position: float = 0.1
) -> float:
    """
    Calculate position size using target volatility approach.
    
    Args:
        realized_volatility: Stock's annualized realized volatility
        target_volatility: Target portfolio volatility (default 20%)
        max_position: Maximum position size as fraction of capital
        min_position: Minimum position size as fraction of capital
        
    Returns:
        Position size as fraction of capital
    """
    if realized_volatility <= 0:
        return min_position
    
    # Scale position inversely to volatility
    position_size = target_volatility / realized_volatility
    
    # Apply bounds
    position_size = max(min_position, min(max_position, position_size))
    
    return position_size


# ============================================================================
# Model Persistence Functions
# ============================================================================

def save_scaler(scaler: StandardScaler, filepath: str):
    """Save scaler to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath: str) -> StandardScaler:
    """Load scaler from file."""
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def save_feature_columns(feature_columns: List[str], filepath: str):
    """Save feature columns list to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature columns saved to {filepath}")


def load_feature_columns(filepath: str) -> List[str]:
    """Load feature columns list from file."""
    with open(filepath, 'rb') as f:
        feature_columns = pickle.load(f)
    return feature_columns


# ============================================================================
# Data Splitting Functions
# ============================================================================

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature sequences
        y: Labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_ratio, random_state=random_state, stratify=y
    )
    
    # Second split: val vs test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_fraction, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def temporal_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data temporally (no shuffling, preserves time order).
    Better for time series to avoid look-ahead bias.
    
    Args:
        X: Feature sequences (assumed to be in temporal order)
        y: Labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Performance Metrics
# ============================================================================

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from prices."""
    return prices.pct_change().dropna()


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate annualized Sortino ratio (using downside deviation).
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        Tuple of (max_drawdown, peak_idx, trough_idx)
    """
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max
    
    max_dd = drawdowns.min()
    trough_idx = drawdowns.idxmin()
    peak_idx = equity_curve[:trough_idx].idxmax()
    
    return max_dd, peak_idx, trough_idx


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: float = 252.0
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Daily returns series
        periods_per_year: Number of trading periods per year
        
    Returns:
        Calmar ratio
    """
    # Build equity curve
    equity = (1 + returns).cumprod()
    
    # Annualized return
    total_return = equity.iloc[-1] - 1
    num_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    
    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity)
    
    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / abs(max_dd)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test position sizing
    print("\nPosition sizing tests:")
    for vol in [0.15, 0.20, 0.30, 0.50, 0.80]:
        pos = calculate_position_size(vol)
        print(f"  Volatility {vol:.0%} -> Position size {pos:.2%}")
    
    # Test class weights
    print("\nClass weight tests:")
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    weights = compute_class_weights(labels)
    print(f"  Labels: {labels}")
    print(f"  Weights: {weights}")
    
    print("\nUtility module loaded successfully!")
