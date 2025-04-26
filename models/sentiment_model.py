"""
Specialized model for sentiment impact
Correlation analysis between sentiment and price movements
"""
from typing import Dict, Optional
import random  # Temporary for demo
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

class SentimentModel:
    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.news_cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(minutes=5)

    def get_current_sentiment(self, symbol: str) -> float:
        """
        Get current sentiment score for a cryptocurrency.
        Returns a value between 0 (very negative) and 1 (very positive).
        For now, returns a combination of:
        - Fear & Greed Index (normalized)
        - Recent news sentiment
        - Social media sentiment (mock data for now)
        """
        try:
            # Get Fear & Greed Index
            fear_greed_score = self._get_fear_greed_index()
            
            # Get news sentiment (mock for now)
            news_sentiment = self._analyze_recent_news(symbol)
            
            # Combine scores (simple average for now)
            # We'll weight these properly in the full implementation
            combined_score = (fear_greed_score + news_sentiment) / 2
            
            return max(0.0, min(1.0, combined_score))
        
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