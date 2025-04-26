"""
Data cleaning and normalization
Feature engineering (moving averages, RSI, MACD, etc.)
Time series transformations
Data merging (price data with sentiment data)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import warnings
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processes cryptocurrency data by cleaning and normalizing it.
    Handles price data, volume data, and derived indicators.
    """
    
    def __init__(self, remove_outliers: bool = True):
        """
        Initialize the data processor.
        
        Args:
            remove_outliers (bool): Whether to remove statistical outliers
        """
        self.remove_outliers = remove_outliers
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean cryptocurrency price data.
        
        Args:
            df (pd.DataFrame): Raw price DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned price data
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")
            
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers if requested
        if self.remove_outliers:
            df = self._remove_price_outliers(df)
            df = self._remove_volume_outliers(df)
        
        # Ensure data consistency
        df = self._ensure_ohlc_consistency(df)
        
        # Add basic quality indicators
        df = self._add_quality_indicators(df)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features for model training.
        
        Args:
            df (pd.DataFrame): Cleaned price DataFrame
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Normalized DataFrame and scaling info
        """
        df = df.copy()
        
        # Separate price and volume data
        price_columns = ['open', 'high', 'low', 'close']
        volume_columns = ['volume']
        
        # Fit scalers if not already fit
        if not hasattr(self.price_scaler, 'center_'):
            self.price_scaler.fit(df[price_columns])
        if not hasattr(self.volume_scaler, 'center_'):
            self.volume_scaler.fit(df[volume_columns])
        
        # Transform the data
        df[price_columns] = self.price_scaler.transform(df[price_columns])
        df[volume_columns] = self.volume_scaler.transform(df[volume_columns])
        
        # Store scaling info
        scaling_info = {
            'price_scaler': self.price_scaler,
            'volume_scaler': self.volume_scaler
        }
        
        return df, scaling_info
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Check for missing values
        missing_count = df.isnull().sum()
        if missing_count.any():
            warnings.warn(f"Found missing values:\n{missing_count[missing_count > 0]}")
        
        # Forward fill missing values (use previous day's data)
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining missing values (for first rows)
        df = df.fillna(method='bfill')
        
        return df
    
    def _remove_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers from price data.
        Uses Interquartile Range (IQR) method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Replace outliers with bounds
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def _remove_volume_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove statistical outliers from volume data.
        Uses a more lenient approach than price outlier removal.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with volume outliers removed
        """
        Q1 = df['volume'].quantile(0.25)
        Q3 = df['volume'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 5 * IQR  # More lenient lower bound
        upper_bound = Q3 + 5 * IQR  # More lenient upper bound
        
        # Replace outliers with bounds
        df.loc[df['volume'] < lower_bound, 'volume'] = lower_bound
        df.loc[df['volume'] > upper_bound, 'volume'] = upper_bound
        
        return df
    
    def _ensure_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure OHLC data is consistent (high >= open,close >= low).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with consistent OHLC values
        """
        # Ensure high is highest
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is lowest
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df
    
    def _add_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add data quality indicators.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with quality indicators
        """
        # Calculate daily price range
        df['daily_range'] = (df['high'] - df['low']) / df['low']
        
        # Calculate price jump indicator
        df['price_jump'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Flag suspicious patterns
        df['suspicious'] = (
            (df['daily_range'] > 0.5) |  # More than 50% daily range
            (df['price_jump'] > 0.3) |   # More than 30% price jump
            (df['volume'] == 0)          # Zero volume
        )
        
        return df
    
    def inverse_transform_prices(self, normalized_prices: np.ndarray) -> np.ndarray:
        """
        Convert normalized prices back to original scale.
        
        Args:
            normalized_prices (np.ndarray): Normalized price data
            
        Returns:
            np.ndarray: Original scale prices
        """
        if not hasattr(self.price_scaler, 'center_'):
            raise ValueError("Price scaler has not been fit yet")
        
        return self.price_scaler.inverse_transform(normalized_prices)
    
    def get_feature_ranges(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Get the valid ranges for each feature.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict[str, Tuple[float, float]]: Feature ranges
        """
        ranges = {}
        
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            ranges[column] = (lower_bound, upper_bound)
        
        return ranges
    
    def add_technical_indicators(self, df: pd.DataFrame, 
                               ma_periods: List[int] = [7, 14, 21, 50, 200],
                               rsi_period: int = 14,
                               bb_period: int = 20,
                               macd_params: Tuple[int, int, int] = (12, 26, 9)) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data
            ma_periods (List[int]): Periods for moving averages
            rsi_period (int): Period for RSI calculation
            bb_period (int): Period for Bollinger Bands
            macd_params (Tuple[int, int, int]): MACD parameters (fast, slow, signal)
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Add moving averages
        for period in ma_periods:
            df[f'SMA_{period}'] = self._calculate_sma(df['close'], period)
            df[f'EMA_{period}'] = self._calculate_ema(df['close'], period)
        
        # Add RSI
        df['RSI'] = self._calculate_rsi(df['close'], rsi_period)
        
        # Add MACD
        macd_data = self._calculate_macd(df['close'], *macd_params)
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Histogram'] = macd_data['Histogram']
        
        # Add Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], bb_period)
        df['BB_Upper'] = bb_data['Upper']
        df['BB_Middle'] = bb_data['Middle']
        df['BB_Lower'] = bb_data['Lower']
        
        # Add volume indicators
        df['Volume_SMA'] = self._calculate_sma(df['volume'], 20)
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        # Add momentum indicators
        df['ROC'] = self._calculate_roc(df['close'], 14)  # Rate of Change
        df['MFI'] = self._calculate_mfi(df, 14)  # Money Flow Index
        
        return df
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period, min_periods=1).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int, 
                       slow_period: int, signal_period: int) -> Dict[str, pd.Series]:
        """Calculate MACD, Signal line, and Histogram"""
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)
        
        macd = fast_ema - slow_ema
        signal = self._calculate_ema(macd, signal_period)
        histogram = macd - signal
        
        return {
            'MACD': macd,
            'Signal': signal,
            'Histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, 
                                 period: int, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = self._calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        }
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # Calculate positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow[price_diff > 0] = money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for feature engineering."""
    df = df.copy()
    
    # Moving averages
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    
    # Price momentum
    df['returns'] = df['close'].pct_change()
    df['returns_7d'] = df['close'].pct_change(periods=7)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=21).std()
    
    # Trading volume features
    df['volume_ma7'] = df['volume'].rolling(window=7).mean()
    df['volume_ma21'] = df['volume'].rolling(window=21).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def prepare_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for model training.
    
    Args:
        df: Raw cryptocurrency data DataFrame
    
    Returns:
        Processed DataFrame ready for training
    """
    logger.info("Preparing data for training...")
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Handle missing values
    df = df.dropna()
    
    # Ensure data is sorted by time
    df = df.sort_index()
    
    logger.info(f"Data preparation completed. Shape: {df.shape}")
    return df

def split_data(df: pd.DataFrame, train_size: float = 0.6, val_size: float = 0.2) -> tuple:
    """
    Split data into training, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df

# Example usage
if __name__ == "__main__":
    # Create processor instance
    processor = DataProcessor(remove_outliers=True)
    
    # Example DataFrame
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=100),
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(102, 10, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }).set_index('date')
    
    # Clean and normalize data
    cleaned_df = processor.clean_price_data(df)
    normalized_df, scaling_info = processor.normalize_features(cleaned_df)
    
    print("\nCleaned Data Sample:")
    print(cleaned_df.head())
    print("\nNormalized Data Sample:")
    print(normalized_df.head())