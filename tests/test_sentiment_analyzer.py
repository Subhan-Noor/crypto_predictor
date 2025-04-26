import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from processors.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture
def sample_fear_greed_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
    data = {
        'timestamp': dates,
        'value': [25, 45, 65, 75, 50, 30, 20],  # Mix of fear and greed values
        'classification': ['Fear', 'Fear', 'Greed', 'Extreme Greed', 'Neutral', 'Fear', 'Extreme Fear']
    }
    return pd.DataFrame(data).set_index('timestamp')

@pytest.fixture
def sample_news_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
    data = {
        'published_at': dates,
        'title': [
            'Bitcoin Surges to New Heights',
            'Market Crash Fears Intensify',
            'Neutral Market Analysis',
            'Positive Regulatory News',
            'Minor Market Correction',
            'Bullish Market Indicators',
            'Concerning Market Trends'
        ],
        'sentiment': [0.8, -0.7, 0.1, 0.6, -0.3, 0.5, -0.4]
    }
    return pd.DataFrame(data).set_index('published_at')

@pytest.fixture
def sample_reddit_data():
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
    data = {
        'timestamp': dates,
        'title': [
            'To the moon! 🚀',
            'Why I\'m bearish',
            'Technical Analysis',
            'Great news everyone!',
            'Market Discussion',
            'Bullish indicators',
            'Worried about trends'
        ],
        'sentiment': [0.9, -0.6, 0.0, 0.7, 0.2, 0.6, -0.5],
        'score': [100, 50, 30, 200, 40, 150, 80],
        'num_comments': [50, 30, 20, 100, 25, 75, 40]
    }
    return pd.DataFrame(data).set_index('timestamp')

def test_fear_greed_normalization(sentiment_analyzer, sample_fear_greed_data):
    """Test Fear & Greed Index normalization"""
    normalized = sentiment_analyzer.normalize_fear_greed(sample_fear_greed_data)
    
    assert normalized is not None
    assert 'normalized_sentiment' in normalized.columns
    assert normalized['normalized_sentiment'].min() >= -1
    assert normalized['normalized_sentiment'].max() <= 1
    
    # Test extreme values
    assert normalized.loc[normalized['value'] == 20]['normalized_sentiment'].iloc[0] < -0.5  # Extreme fear
    assert normalized.loc[normalized['value'] == 75]['normalized_sentiment'].iloc[0] > 0.5   # Extreme greed

def test_news_sentiment_normalization(sentiment_analyzer, sample_news_data):
    """Test news sentiment normalization"""
    normalized = sentiment_analyzer.normalize_news_sentiment(sample_news_data)
    
    assert normalized is not None
    assert 'normalized_sentiment' in normalized.columns
    assert normalized['normalized_sentiment'].min() >= -1
    assert normalized['normalized_sentiment'].max() <= 1
    
    # Test sentiment preservation
    most_positive = normalized.loc[normalized['sentiment'] == 0.8]
    most_negative = normalized.loc[normalized['sentiment'] == -0.7]
    assert most_positive['normalized_sentiment'].iloc[0] > most_negative['normalized_sentiment'].iloc[0]

def test_reddit_sentiment_normalization(sentiment_analyzer, sample_reddit_data):
    """Test Reddit sentiment normalization with engagement weighting"""
    normalized = sentiment_analyzer.normalize_reddit_sentiment(sample_reddit_data)
    
    assert normalized is not None
    assert 'normalized_sentiment' in normalized.columns
    assert 'engagement_weight' in normalized.columns
    assert normalized['normalized_sentiment'].min() >= -1
    assert normalized['normalized_sentiment'].max() <= 1
    
    # Test engagement weighting
    high_engagement = normalized.loc[normalized['score'] + normalized['num_comments'] == 300]  # 200 + 100
    low_engagement = normalized.loc[normalized['score'] + normalized['num_comments'] == 50]    # 30 + 20
    assert high_engagement['engagement_weight'].iloc[0] > low_engagement['engagement_weight'].iloc[0]

def test_sentiment_aggregation(sentiment_analyzer, sample_fear_greed_data, sample_news_data, sample_reddit_data):
    """Test aggregation of multiple sentiment sources"""
    aggregated = sentiment_analyzer.aggregate_sentiment(
        sample_fear_greed_data,
        sample_news_data,
        sample_reddit_data
    )
    
    assert aggregated is not None
    assert 'aggregate_sentiment' in aggregated.columns
    assert 'fear_greed' in aggregated.columns
    assert 'news' in aggregated.columns
    assert 'reddit' in aggregated.columns
    
    # Test that aggregate sentiment is within bounds
    assert aggregated['aggregate_sentiment'].min() >= -1
    assert aggregated['aggregate_sentiment'].max() <= 1
    
    # Test that weights are properly applied
    weights_sum = (
        sentiment_analyzer.source_weights['fear_greed'] +
        sentiment_analyzer.source_weights['news'] +
        sentiment_analyzer.source_weights['reddit']
    )
    assert abs(weights_sum - 1.0) < 1e-10  # Should sum to 1

def test_sentiment_metrics(sentiment_analyzer, sample_fear_greed_data, sample_news_data, sample_reddit_data):
    """Test sentiment metrics calculation"""
    aggregated = sentiment_analyzer.aggregate_sentiment(
        sample_fear_greed_data,
        sample_news_data,
        sample_reddit_data
    )
    metrics = sentiment_analyzer.get_sentiment_metrics(aggregated)
    
    assert metrics is not None
    assert 'current_sentiment' in metrics
    assert 'sentiment_ma_7d' in metrics
    assert 'sentiment_volatility' in metrics
    assert 'sentiment_trend' in metrics
    
    # Test trend calculation
    assert metrics['sentiment_trend'] in ['positive', 'negative', 'neutral']
    
    # Test metric bounds
    assert -1 <= metrics['current_sentiment'] <= 1
    assert -1 <= metrics['sentiment_ma_7d'] <= 1
    assert metrics['sentiment_volatility'] >= 0

def test_missing_data_handling(sentiment_analyzer):
    """Test handling of missing or empty data"""
    # Test with None inputs
    result = sentiment_analyzer.aggregate_sentiment(None, None, None)
    assert result is None
    
    # Test with empty DataFrames
    empty_df = pd.DataFrame()
    result = sentiment_analyzer.aggregate_sentiment(empty_df, empty_df, empty_df)
    assert result is None
    
    # Test with partial data
    partial_data = pd.DataFrame({
        'value': [50],
        'classification': ['Neutral']
    }, index=[datetime.now()])
    
    result = sentiment_analyzer.aggregate_sentiment(partial_data, None, None)
    assert result is not None
    assert 'fear_greed' in result.columns
    assert 'aggregate_sentiment' in result.columns 