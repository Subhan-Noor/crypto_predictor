"""
Feature Engineering for Crypto Price Prediction

This module creates technical indicators and features for ML models:
- Moving averages (SMA, EMA)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price-based features (volatility, returns)
- Sentiment features
- Lagged features
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.logger import logger


class CryptoFeatureEngineer:
    """Generates features for crypto price prediction models"""
    
    def __init__(self, feature_window: int = 14):
        """
        Initialize feature engineer
        
        Args:
            feature_window: Number of days to use for rolling calculations
        """
        self.feature_window = feature_window
        
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price features
        """
        df = df.copy()
        
        # Price returns
        df['price_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread  
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Volume-based features
        df['volume_sma'] = df['volume'].rolling(window=self.feature_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price volatility (rolling standard deviation)
        df['volatility'] = df['price_return'].rolling(window=self.feature_window).std()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Get data size to adapt window sizes
        data_size = len(df)
        
        # Adaptive window sizes based on data size
        window_7 = min(7, data_size - 1)
        window_14 = min(14, data_size - 1)
        window_20 = min(20, data_size - 1)
        window_30 = min(30, data_size - 1)
        
        # Moving Averages (only if we have enough data)
        if window_7 > 1:
            df['sma_7'] = ta.trend.sma_indicator(df['close'], window=window_7)
            df['ema_7'] = ta.trend.ema_indicator(df['close'], window=window_7)
        else:
            df['sma_7'] = np.nan
            df['ema_7'] = np.nan
            
        if window_14 > 1:
            df['sma_14'] = ta.trend.sma_indicator(df['close'], window=window_14)
            df['ema_14'] = ta.trend.ema_indicator(df['close'], window=window_14)
        else:
            df['sma_14'] = np.nan
            df['ema_14'] = np.nan
            
        if window_30 > 1:
            df['sma_30'] = ta.trend.sma_indicator(df['close'], window=window_30)
        else:
            df['sma_30'] = np.nan
        
        # RSI (Relative Strength Index)
        if window_14 > 1:
            df['rsi'] = ta.momentum.rsi(df['close'], window=window_14)
        else:
            df['rsi'] = np.nan
        
        # MACD (requires at least 26 data points for default settings)
        if data_size >= 26:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        else:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_diff'] = np.nan
        
        # Bollinger Bands
        if window_20 > 1:
            bollinger = ta.volatility.BollingerBands(df['close'], window=window_20)
            df['bb_high'] = bollinger.bollinger_hband()
            df['bb_low'] = bollinger.bollinger_lband()
            df['bb_mid'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        else:
            df['bb_high'] = np.nan
            df['bb_low'] = np.nan
            df['bb_mid'] = np.nan
            df['bb_width'] = np.nan
            df['bb_position'] = np.nan
        
        # Stochastic Oscillator (requires at least 14 data points)
        if data_size >= 14:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
        else:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
        
        # Average True Range (volatility)
        if data_size >= 14:
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        else:
            df['atr'] = np.nan
        
        # Williams %R (requires at least 14 data points)
        if data_size >= 14:
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        else:
            df['williams_r'] = np.nan
        
        # Money Flow Index (requires at least 14 data points)
        if data_size >= 14:
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        else:
            df['mfi'] = np.nan
        
        return df
    
    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-based features
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()
        
        # Combined sentiment
        df['combined_sentiment'] = (df['twitter_sentiment'] + df['reddit_sentiment']) / 2
        
        # Sentiment moving averages
        df['twitter_sentiment_sma'] = df['twitter_sentiment'].rolling(window=7).mean()
        df['reddit_sentiment_sma'] = df['reddit_sentiment'].rolling(window=7).mean()
        df['combined_sentiment_sma'] = df['combined_sentiment'].rolling(window=7).mean()
        
        # Sentiment volatility
        df['twitter_sentiment_vol'] = df['twitter_sentiment'].rolling(window=7).std()
        df['reddit_sentiment_vol'] = df['reddit_sentiment'].rolling(window=7).std()
        
        # Sentiment momentum (rate of change)
        df['twitter_sentiment_momentum'] = df['twitter_sentiment'].diff(7)
        df['reddit_sentiment_momentum'] = df['reddit_sentiment'].diff(7)
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 7]) -> pd.DataFrame:
        """
        Add lagged features for temporal patterns
        
        Args:
            df: DataFrame with features
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lagged features
        """
        df = df.copy()
        
        # Key features to lag
        features_to_lag = [
            'close', 'price_return', 'volatility', 'rsi', 'macd',
            'bb_position', 'twitter_sentiment', 'reddit_sentiment'
        ]
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend and momentum features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        # Price trend (slope of linear regression)
        def calculate_trend(series, window=7):
            """Calculate trend slope over a window"""
            trends = []
            for i in range(len(series)):
                if i < window - 1:
                    trends.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    trend = np.polyfit(x, y, 1)[0]
                    trends.append(trend)
            return pd.Series(trends, index=series.index)
        
        # Use smaller windows if data is limited
        data_size = len(df)
        window_7 = min(7, data_size - 1)
        window_feature = min(self.feature_window, data_size - 1)
        
        if window_7 > 1:
            df['price_trend_7'] = calculate_trend(df['close'], window_7)
        else:
            df['price_trend_7'] = np.nan
            
        if window_feature > 1:
            df[f'price_trend_{self.feature_window}'] = calculate_trend(df['close'], window_feature)
        else:
            df[f'price_trend_{self.feature_window}'] = np.nan
        
        # Price position relative to recent high/low
        df['price_vs_high_7'] = df['close'] / df['high'].rolling(min(7, data_size)).max()
        df['price_vs_low_7'] = df['close'] / df['low'].rolling(min(7, data_size)).min()
        df[f'price_vs_high_{self.feature_window}'] = df['close'] / df['high'].rolling(min(self.feature_window, data_size)).max()
        df[f'price_vs_low_{self.feature_window}'] = df['close'] / df['low'].rolling(min(self.feature_window, data_size)).min()
        
        # Moving average convergence/divergence ratios
        if 'sma_7' in df.columns and 'sma_14' in df.columns:
            df['sma_ratio_7_14'] = df['sma_7'] / df['sma_14']
        if 'sma_7' in df.columns:
            df['price_vs_sma_7'] = df['close'] / df['sma_7']
        if 'sma_14' in df.columns:
            df['price_vs_sma_14'] = df['close'] / df['sma_14']
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Raw DataFrame with OHLCV and sentiment data
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Start with a copy
        features_df = df.copy()
        
        # Add different types of features
        features_df = self.add_price_features(features_df)
        features_df = self.add_technical_indicators(features_df)
        features_df = self.add_sentiment_features(features_df)
        features_df = self.add_trend_features(features_df)
        features_df = self.add_lagged_features(features_df)
        
        # Remove rows with too many NaN values (from rolling calculations)
        # For small datasets, be more lenient
        if len(features_df) < 20:
            # For small datasets, keep more rows
            threshold = len(features_df.columns) * 0.3  # Only 30% of features need to be non-null
        else:
            # For larger datasets, be more strict
            threshold = len(features_df.columns) * 0.8  # 80% of features need to be non-null
            
        features_df = features_df.dropna(thresh=threshold)
        
        logger.info(f"Feature engineering complete. Shape: {features_df.shape}")
        logger.info(f"Features created: {len(features_df.columns)} columns")
        
        return features_df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata and target)
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        # Exclude these columns from features
        exclude_cols = [
            'date', 'target', 'future_close', 'price_change_pct',
            # Original OHLCV columns (we'll use engineered features instead)
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with too many NaN values
        nan_threshold = 0.5  # Allow up to 50% NaN values
        valid_feature_cols = []
        
        for col in feature_cols:
            if col in df.columns:
                nan_ratio = df[col].isna().sum() / len(df) if len(df) > 0 else 1.0
                if nan_ratio < nan_threshold:
                    valid_feature_cols.append(col)
        
        logger.info(f"Selected {len(valid_feature_cols)} feature columns")
        return valid_feature_cols
    
    def prepare_features_for_ml(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare features and targets for ML models
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Dictionary with 'X' (features) and 'y' (targets)
        """
        feature_cols = self.get_feature_columns(df)
        
        # Extract features and target
        X = df[feature_cols].fillna(0).values  # Fill remaining NaN with 0
        y = df['target'].values
        
        # Remove any infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape}")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_cols
        }


# Utility function for quick feature engineering
def engineer_features(df: pd.DataFrame, feature_window: int = 14) -> pd.DataFrame:
    """
    Quick utility function for feature engineering
    
    Args:
        df: Raw DataFrame with OHLCV and sentiment data
        feature_window: Window size for rolling calculations
        
    Returns:
        DataFrame with engineered features
    """
    engineer = CryptoFeatureEngineer(feature_window=feature_window)
    return engineer.create_features(df) 