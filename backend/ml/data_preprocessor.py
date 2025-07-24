"""
Data Preprocessing Pipeline for Crypto Price Prediction

This module handles:
- Combining price and sentiment data
- Creating ML-ready datasets
- Data validation and cleaning
- Train/test splitting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database import db_manager
from app.logger import logger


class CryptoDataPreprocessor:
    """Preprocesses crypto data for ML model training"""
    
    def __init__(self, prediction_horizon: int = 7):
        """
        Initialize preprocessor
        
        Args:
            prediction_horizon: Number of days ahead to predict (default: 7)
        """
        self.prediction_horizon = prediction_horizon
        self.feature_window = 14  # Days of historical data for features
        self.min_data_points = 30  # Minimum data points needed for training
        
    async def load_data(self, currency: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """
        Load price and sentiment data from database
        
        Args:
            currency: 'BTC' or 'ETH'
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            
        Returns:
            Dictionary containing 'prices' and 'sentiment' DataFrames
        """
        logger.info(f"Loading data for {currency}")
        
        try:
            # Build query filters
            filters = {'currency': currency}
            
            # Load price data
            price_records = await db_manager.get_records('crypto_prices', filters)
            prices_df = pd.DataFrame([
                {
                    'date': record['date'],
                    'open': float(record['open']),
                    'high': float(record['high']),
                    'low': float(record['low']),
                    'close': float(record['close']),
                    'volume': float(record['volume'])
                }
                for record in price_records
            ])
            
            # Load sentiment data
            sentiment_records = await db_manager.get_records('crypto_sentiment', filters)
            sentiment_df = pd.DataFrame([
                {
                    'date': record['date'],
                    'twitter_sentiment': float(record['twitter_sentiment']) if record['twitter_sentiment'] else None,
                    'reddit_sentiment': float(record['reddit_sentiment']) if record['reddit_sentiment'] else None
                }
                for record in sentiment_records
            ])
            
            # Convert date columns to datetime
            if not prices_df.empty:
                prices_df['date'] = pd.to_datetime(prices_df['date'])
                prices_df = prices_df.sort_values('date').reset_index(drop=True)
                
            if not sentiment_df.empty:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
            
            # Filter by date range if provided
            if start_date:
                prices_df = prices_df[prices_df['date'] >= start_date]
                sentiment_df = sentiment_df[sentiment_df['date'] >= start_date]
                
            if end_date:
                prices_df = prices_df[prices_df['date'] <= end_date]
                sentiment_df = sentiment_df[sentiment_df['date'] <= end_date]
            
            logger.info(f"Loaded {len(prices_df)} price records and {len(sentiment_df)} sentiment records")
            
            return {
                'prices': prices_df,
                'sentiment': sentiment_df
            }
            
        except Exception as e:
            logger.error(f"Error loading data for {currency}: {str(e)}")
            raise
    
    def merge_data(self, prices_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge price and sentiment data on date (sentiment is optional)
        
        Args:
            prices_df: Price data DataFrame
            sentiment_df: Sentiment data DataFrame (can be empty)
            
        Returns:
            Merged DataFrame with price and optional sentiment data
        """
        # If no sentiment data, create neutral sentiment columns
        if sentiment_df.empty:
            logger.info("No sentiment data available, using neutral sentiment values")
            merged_df = prices_df.copy()
            merged_df['twitter_sentiment'] = 0.0  # Neutral sentiment
            merged_df['reddit_sentiment'] = 0.0   # Neutral sentiment
        else:
            # Merge on date (left join to keep all price data)
            merged_df = pd.merge(prices_df, sentiment_df, on='date', how='left')
            
            # Forward fill missing sentiment values (carry last known sentiment)
            merged_df['twitter_sentiment'] = merged_df['twitter_sentiment'].fillna(method='ffill')
            merged_df['reddit_sentiment'] = merged_df['reddit_sentiment'].fillna(method='ffill')
            
            # If still missing, fill with neutral sentiment (0.0)
            merged_df['twitter_sentiment'] = merged_df['twitter_sentiment'].fillna(0.0)
            merged_df['reddit_sentiment'] = merged_df['reddit_sentiment'].fillna(0.0)
        
        return merged_df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create prediction labels based on future price movement
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added 'target' column (1=UP, 0=DOWN)
        """
        df = df.copy()
        
        # Calculate future price (N days ahead)
        df['future_close'] = df['close'].shift(-self.prediction_horizon)
        
        # Calculate price change percentage
        df['price_change_pct'] = (df['future_close'] - df['close']) / df['close'] * 100
        
        # Create binary target: 1 if price increases, 0 if decreases
        # Using a small threshold to ignore very small movements
        threshold = 0.5  # 0.5% threshold
        df['target'] = (df['price_change_pct'] > threshold).astype(int)
        
        # Remove rows where we don't have future data
        df = df.dropna(subset=['future_close']).reset_index(drop=True)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that we have sufficient data for training
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            True if data is valid for training
        """
        if len(df) < self.min_data_points:
            logger.error(f"Insufficient data: {len(df)} records, need at least {self.min_data_points}")
            return False
            
        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for null values in critical columns
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Null values found: {null_counts.to_dict()}")
            
        return True
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets (chronological split)
        
        Args:
            df: Preprocessed DataFrame
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Data split: {len(train_df)} training, {len(test_df)} testing samples")
        
        return train_df, test_df
    
    async def prepare_ml_dataset(self, currency: str) -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline
        
        Args:
            currency: 'BTC' or 'ETH'
            
        Returns:
            Dictionary with 'train' and 'test' DataFrames ready for ML
        """
        logger.info(f"Preparing ML dataset for {currency}")
        
        # Load data
        data = await self.load_data(currency)
        
        if data['prices'].empty:
            raise ValueError(f"No price data available for {currency}")
        
        # Merge price and sentiment data
        merged_df = self.merge_data(data['prices'], data['sentiment'])
        
        # Create prediction labels
        labeled_df = self.create_labels(merged_df)
        
        # Validate data
        if not self.validate_data(labeled_df):
            raise ValueError("Data validation failed")
        
        # Split into train/test
        train_df, test_df = self.split_data(labeled_df)
        
        logger.info(f"Dataset prepared successfully for {currency}")
        
        return {
            'train': train_df,
            'test': test_df,
            'full': labeled_df
        }


# Utility function for quick dataset preparation
async def prepare_dataset(currency: str, prediction_horizon: int = 7) -> Dict[str, pd.DataFrame]:
    """
    Quick utility function to prepare a dataset
    
    Args:
        currency: 'BTC' or 'ETH'
        prediction_horizon: Days ahead to predict
        
    Returns:
        Dictionary with train/test DataFrames
    """
    preprocessor = CryptoDataPreprocessor(prediction_horizon=prediction_horizon)
    return await preprocessor.prepare_ml_dataset(currency) 