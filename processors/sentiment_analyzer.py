"""
NLP processing for news headlines and social posts
Sentiment classification and scoring
Aggregate sentiment metrics calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Processes and normalizes sentiment data from multiple sources:
    - Fear & Greed Index
    - News sentiment
    - Social media sentiment (Reddit)
    
    All sentiment scores are normalized to [-1, 1] range where:
    -1 = Extremely Negative
    0 = Neutral
    1 = Extremely Positive
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Default weights for sentiment aggregation
        self.source_weights = {
            'fear_greed': 0.4,  # Fear & Greed Index is a strong market indicator
            'news': 0.35,       # News has significant but slightly less impact
            'reddit': 0.25      # Social sentiment can be noisy
        }
        
        # Fear & Greed thresholds
        self.extreme_fear_threshold = 25
        self.fear_threshold = 40
        self.neutral_threshold = 60
        self.greed_threshold = 75
    
    def normalize_fear_greed(self, fear_greed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Fear & Greed Index from [0, 100] to [-1, 1]
        using a piecewise linear transformation that emphasizes extreme values
        
        Args:
            fear_greed_df: DataFrame with fear & greed data
            
        Returns:
            DataFrame with normalized sentiment
        """
        if fear_greed_df is None or fear_greed_df.empty:
            return None
            
        try:
            # Create copy to avoid modifying original
            df = fear_greed_df.copy()
            
            def piecewise_normalize(x):
                if x <= self.extreme_fear_threshold:
                    # Extreme fear region: [0, 25] -> [-1, -0.6]
                    return -1.0 + 0.4 * (x / self.extreme_fear_threshold)
                elif x <= self.fear_threshold:
                    # Fear region: [25, 40] -> [-0.6, -0.2]
                    return -0.6 + 0.4 * ((x - self.extreme_fear_threshold) / (self.fear_threshold - self.extreme_fear_threshold))
                elif x <= self.neutral_threshold:
                    # Neutral region: [40, 60] -> [-0.2, 0.2]
                    return -0.2 + 0.4 * ((x - self.fear_threshold) / (self.neutral_threshold - self.fear_threshold))
                elif x <= self.greed_threshold:
                    # Greed region: [60, 75] -> [0.2, 0.6]
                    return 0.2 + 0.4 * ((x - self.neutral_threshold) / (self.greed_threshold - self.neutral_threshold))
                else:
                    # Extreme greed region: [75, 100] -> [0.6, 1.0]
                    return 0.6 + 0.4 * ((x - self.greed_threshold) / (100 - self.greed_threshold))
            
            df['normalized_sentiment'] = df['value'].apply(piecewise_normalize)
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing fear & greed data: {e}")
            return None
    
    def normalize_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalize news sentiment, handling outliers
        
        Args:
            news_df: DataFrame with news sentiment
            
        Returns:
            DataFrame with normalized sentiment
        """
        if news_df is None or news_df.empty:
            return None
            
        try:
            df = news_df.copy()
            
            # Calculate weighted sentiment based on source reliability
            # Could be extended with source-specific weights
            df['weighted_sentiment'] = df['sentiment']
            
            # Handle outliers using winsorization
            q_low = df['weighted_sentiment'].quantile(0.05)
            q_high = df['weighted_sentiment'].quantile(0.95)
            df['normalized_sentiment'] = df['weighted_sentiment'].clip(q_low, q_high)
            
            # Normalize to [-1, 1] if not already in that range
            if df['normalized_sentiment'].abs().max() > 1:
                df['normalized_sentiment'] = self.scaler.fit_transform(
                    df['normalized_sentiment'].values.reshape(-1, 1)
                ).flatten()
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing news sentiment: {e}")
            return None
    
    def normalize_reddit_sentiment(self, reddit_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalize Reddit sentiment, incorporating post scores
        
        Args:
            reddit_df: DataFrame with Reddit sentiment
            
        Returns:
            DataFrame with normalized sentiment
        """
        if reddit_df is None or reddit_df.empty:
            return None
            
        try:
            df = reddit_df.copy()
            
            # Calculate engagement score
            df['engagement'] = np.log1p(df['score'] + df['num_comments'])
            
            # Normalize engagement to [0, 1] for weighting
            df['engagement_weight'] = (df['engagement'] - df['engagement'].min()) / \
                                    (df['engagement'].max() - df['engagement'].min())
            
            # Apply engagement weight to sentiment
            df['weighted_sentiment'] = df['sentiment'] * (0.5 + 0.5 * df['engagement_weight'])
            
            # Normalize to [-1, 1]
            df['normalized_sentiment'] = self.scaler.fit_transform(
                df['weighted_sentiment'].values.reshape(-1, 1)
            ).flatten()
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing Reddit sentiment: {e}")
            return None
    
    def aggregate_sentiment(self, 
                          fear_greed_df: Optional[pd.DataFrame],
                          news_df: Optional[pd.DataFrame],
                          reddit_df: Optional[pd.DataFrame],
                          timeframe: str = '1D') -> pd.DataFrame:
        """
        Aggregate normalized sentiment from all sources into a single score
        
        Args:
            fear_greed_df: Normalized fear & greed data
            news_df: Normalized news sentiment
            reddit_df: Normalized Reddit sentiment
            timeframe: Resampling timeframe ('1D' for daily, '1H' for hourly)
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        try:
            sentiment_dfs = []
            weights = []
            column_names = []
            
            # Process Fear & Greed Index
            if fear_greed_df is not None and not fear_greed_df.empty:
                fear_greed = self.normalize_fear_greed(fear_greed_df)
                if fear_greed is not None:
                    sentiment_dfs.append(
                        fear_greed['normalized_sentiment'].to_frame('fear_greed')
                    )
                    weights.append(self.source_weights['fear_greed'])
                    column_names.append('fear_greed')
            
            # Process News Sentiment
            if news_df is not None and not news_df.empty:
                news = self.normalize_news_sentiment(news_df)
                if news is not None:
                    sentiment_dfs.append(
                        news['normalized_sentiment'].to_frame('news')
                    )
                    weights.append(self.source_weights['news'])
                    column_names.append('news')
            
            # Process Reddit Sentiment
            if reddit_df is not None and not reddit_df.empty:
                reddit = self.normalize_reddit_sentiment(reddit_df)
                if reddit is not None:
                    sentiment_dfs.append(
                        reddit['normalized_sentiment'].to_frame('reddit')
                    )
                    weights.append(self.source_weights['reddit'])
                    column_names.append('reddit')
            
            if not sentiment_dfs:
                logger.warning("No valid sentiment data available")
                return None
            
            # Combine all sentiment sources
            combined = pd.concat(sentiment_dfs, axis=1)
            
            # Normalize weights to sum to 1
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted average sentiment
            combined['aggregate_sentiment'] = np.average(
                combined[column_names].fillna(0), 
                axis=1,
                weights=weights
            )
            
            # Resample to desired timeframe
            resampled = combined.resample(timeframe).agg({
                col: 'mean' for col in combined.columns
            })
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment: {e}")
            return None
    
    def get_sentiment_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate various sentiment metrics from the aggregated data
        
        Args:
            df: DataFrame with aggregated sentiment
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            metrics = {
                'current_sentiment': df['aggregate_sentiment'].iloc[-1],
                'sentiment_ma_7d': df['aggregate_sentiment'].rolling(7).mean().iloc[-1],
                'sentiment_volatility': df['aggregate_sentiment'].std(),
                'sentiment_trend': 'neutral'
            }
            
            # Calculate trend
            last_7d = df['aggregate_sentiment'].iloc[-7:]
            slope = np.polyfit(range(len(last_7d)), last_7d, 1)[0]
            
            if slope > 0.01:
                metrics['sentiment_trend'] = 'positive'
            elif slope < -0.01:
                metrics['sentiment_trend'] = 'negative'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating sentiment metrics: {e}")
            return None