"""
Specialized model for sentiment impact
Correlation analysis between sentiment and price movements
"""
from typing import Dict, Optional
import random  # Temporary for demo
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import json
import os
from data_fetchers.sentiment_fetcher import SentimentFetcher

class SentimentModel:
    def __init__(self):
        self.sentiment_fetcher = SentimentFetcher()
        self.cache_dir = 'data/sentiment'
        self.cache_duration = timedelta(minutes=5)
        
        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_current_sentiment(self, symbol: str) -> float:
        """
        Get current sentiment score for a cryptocurrency.
        Returns a value between 0 (very negative) and 1 (very positive).
        Uses the exact same sentiment data as shown on the Analysis page.
        """
        try:
            # Check cache first for consistency
            cache_file = os.path.join(self.cache_dir, f"{symbol.lower()}_sentiment.json")
            
            # Use cache if it exists and is recent
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check if cache is recent (within 5 minutes)
                    cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01T00:00:00'))
                    if datetime.now() - cache_time < self.cache_duration:
                        # Return the exact same data used by the Analysis page
                        if 'data' in cache_data and len(cache_data['data']) > 0:
                            return cache_data['data'][-1]['sentiment']
                except Exception as e:
                    print(f"Error reading sentiment cache: {e}")
            
            # If not in cache or cache expired, fetch fresh data
            sentiment_data = self.sentiment_fetcher.get_historical_sentiment(symbol, days=7)
            
            # Cache the result for consistency
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'data': sentiment_data
            }
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
            except Exception as e:
                print(f"Error writing sentiment cache: {e}")
            
            # Return the latest sentiment value
            if sentiment_data and len(sentiment_data) > 0:
                return sentiment_data[-1]['sentiment']
            else:
                # Fallback if no data
                return 0.5
            
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            # Return neutral sentiment on error
            return 0.5

    def _get_fear_greed_index(self) -> float:
        """Get and normalize the Fear & Greed Index"""
        try:
            response = requests.get(self.fear_greed_url)
            if response.status_code == 200:
                data = response.json()
                # Fear & Greed Index is 0-100, normalize to 0-1
                value = int(data['data'][0]['value'])
                return value / 100
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
        
        # Return neutral score on error
        return 0.5

    def _analyze_recent_news(self, symbol: str) -> float:
        """
        Analyze recent news sentiment for a cryptocurrency.
        Returns a value between 0 (very negative) and 1 (very positive).
        Currently using mock data - will implement full news analysis later.
        """
        # Check cache
        cache_key = f"{symbol}_news"
        if cache_key in self.news_cache:
            cached_data = self.news_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['sentiment']

        # For demo, return slightly randomized sentiment based on symbol
        # In production, this would analyze real news data
        base_sentiment = {
            'BTC': 0.65,  # Generally positive
            'ETH': 0.60,  # Moderately positive
            'XRP': 0.50,  # Neutral
            'ADA': 0.55,  # Slightly positive
            'SOL': 0.58,  # Moderately positive
            'DOT': 0.52,  # Slightly positive
        }.get(symbol, 0.5)  # Default to neutral

        # Add some random variation (±10%)
        sentiment = base_sentiment + (random.random() - 0.5) * 0.2
        sentiment = max(0.0, min(1.0, sentiment))

        # Cache the result
        self.news_cache[cache_key] = {
            'timestamp': datetime.now(),
            'sentiment': sentiment
        }

        return sentiment