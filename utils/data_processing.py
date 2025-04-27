"""
Utility functions for data processing across the project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


def normalize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize data in specified columns to 0-1 range.
    
    Args:
        data: DataFrame to normalize
        columns: List of column names to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    result = data.copy()
    for column in columns:
        if column in result:
            min_val = result[column].min()
            max_val = result[column].max()
            if max_val > min_val:  # Prevent division by zero
                result[column] = (result[column] - min_val) / (max_val - min_val)
    return result


def create_technical_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Create common technical indicators for time series data.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for the price data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Moving averages
    df['ma7'] = df[price_col].rolling(window=7).mean()
    df['ma21'] = df[price_col].rolling(window=21).mean()
    
    # Price momentum
    df['returns'] = df[price_col].pct_change()
    df['returns_7d'] = df[price_col].pct_change(periods=7)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=21).std()
    
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df


def resample_time_series(df: pd.DataFrame, time_col: str, freq: str = 'D') -> pd.DataFrame:
    """
    Resample time series data to a specified frequency.
    
    Args:
        df: DataFrame with time series data
        time_col: Column name containing timestamp data
        freq: Frequency to resample to ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        Resampled DataFrame
    """
    # Ensure timestamp column is datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    
    # Resample
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled.reset_index()


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a DataFrame column using z-score method.
    
    Args:
        df: DataFrame to check
        column: Column name to check for outliers
        threshold: Z-score threshold (default: 3.0)
        
    Returns:
        Boolean Series where True indicates an outlier
    """
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return abs(z_scores) > threshold 