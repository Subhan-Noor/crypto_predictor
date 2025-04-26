"""
News API integration for crypto news headlines
Social media sentiment scraping (Twitter/X, Reddit)
Fear & Greed Index historical data
Sentiment scoring system
"""

import requests
import pandas as pd
import praw
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List
import os
import json
from textblob import TextBlob
from dotenv import load_dotenv

class SentimentFetcher:
    """
    Fetches and processes sentiment data from multiple sources.
    All sources are free to use with appropriate rate limiting.
    """
    
    # Supported cryptocurrencies and their subreddits
    CRYPTO_SUBREDDITS = {
        'BTC': ['bitcoin', 'cryptocurrency'],
        'ETH': ['ethereum', 'cryptocurrency'],
        'XRP': ['ripple', 'cryptocurrency'],
        'ADA': ['cardano', 'cryptocurrency'],
        'SOL': ['solana', 'cryptocurrency'],
        'DOT': ['dot', 'cryptocurrency']
    }
    
    def __init__(self, cache_dir='data/sentiment'):
        """
        Initialize the sentiment fetcher.
        
        Args:
            cache_dir (str): Directory to store cached sentiment data
        """
        # Load environment variables
        load_dotenv()
        
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize Reddit client
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent='Crypto Sentiment Bot 1.0 (by /u/your_username)'
                )
                # Verify the credentials work
                self.reddit.user.me()
            except Exception as e:
                print(f"Warning: Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            print("Warning: Reddit credentials not found in environment variables")
            self.reddit = None
    
    def get_fear_greed_index(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch Fear & Greed Index data from alternative.me
        
        Args:
            days (int): Number of days of historical data to fetch
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with fear & greed data or None if failed
        """
        try:
            # Cache file path
            cache_file = os.path.join(self.cache_dir, f'fear_greed_{days}d.csv')
            
            # Check if we have cached data less than 1 day old
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                if (datetime.now() - pd.to_datetime(df['timestamp'].max())).days < 1:
                    return df.set_index('timestamp')
            
            # Fetch new data
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Fear & Greed API error: {response.status_code}")
                return None
            
            data = response.json()
            
            # Process the data
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['value'] = df['value'].astype(float)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            
            return df.set_index('timestamp')
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    def get_crypto_news(self, symbol: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch crypto news from CryptoPanic's free API
        
        Args:
            symbol (str): Optional cryptocurrency symbol to filter news
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with news data or None if failed
        """
        try:
            # Cache file path
            cache_file = os.path.join(self.cache_dir, f'news_{symbol or "all"}.csv')
            
            # Check if we have cached data less than 1 hour old
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                if (datetime.now() - pd.to_datetime(df['published_at'].max())).total_seconds() < 3600:
                    return df.set_index('published_at')
            
            # Construct API URL
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': os.getenv('CRYPTOPANIC_API_KEY', ''),  # Optional API key
                'currencies': symbol,
                'kind': 'news'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"CryptoPanic API error: {response.status_code}")
                return None
            
            data = response.json()
            
            # Process the news data
            news_data = []
            for item in data['results']:
                # Calculate sentiment using TextBlob
                sentiment = TextBlob(item['title']).sentiment.polarity
                
                news_data.append({
                    'title': item['title'],
                    'published_at': item['published_at'],
                    'url': item['url'],
                    'source': item['source']['title'],
                    'sentiment': sentiment
                })
            
            df = pd.DataFrame(news_data)
            df['published_at'] = pd.to_datetime(df['published_at'])
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            
            return df.set_index('published_at')
            
        except Exception as e:
            print(f"Error fetching crypto news: {e}")
            return None
    
    def get_reddit_sentiment(self, symbol: str, days: int = 1) -> Optional[pd.DataFrame]:
        """
        Fetch and analyze Reddit sentiment for a specific cryptocurrency
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with Reddit sentiment data or None if failed
        """
        if not self.reddit or symbol not in self.CRYPTO_SUBREDDITS:
            return None
            
        try:
            # Cache file path
            cache_file = os.path.join(self.cache_dir, f'reddit_{symbol}_{days}d.csv')
            
            # Check if we have cached data less than 1 hour old
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                if (datetime.now() - pd.to_datetime(df['timestamp'].max())).total_seconds() < 3600:
                    return df.set_index('timestamp')
            
            posts_data = []
            
            # Get posts from relevant subreddits
            for subreddit_name in self.CRYPTO_SUBREDDITS[symbol]:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get top and new posts
                for post in subreddit.top(time_filter='day', limit=100):
                    if (datetime.now() - datetime.fromtimestamp(post.created_utc)).days <= days:
                        # Calculate sentiment
                        title_sentiment = TextBlob(post.title).sentiment.polarity
                        
                        posts_data.append({
                            'timestamp': datetime.fromtimestamp(post.created_utc),
                            'title': post.title,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'sentiment': title_sentiment,
                            'subreddit': subreddit_name
                        })
            
            if not posts_data:
                return None
                
            df = pd.DataFrame(posts_data)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            
            return df.set_index('timestamp')
            
        except Exception as e:
            print(f"Error fetching Reddit sentiment: {e}")
            return None
    
    def get_aggregated_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get aggregated sentiment from all available sources
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with aggregated sentiment or None if failed
        """
        try:
            # Get data from all sources
            fear_greed_df = self.get_fear_greed_index()
            news_df = self.get_crypto_news(symbol)
            reddit_df = self.get_reddit_sentiment(symbol)
            
            # Create daily sentiment scores
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            sentiment_data = []
            
            for date in date_range:
                daily_sentiment = {
                    'date': date,
                    'fear_greed_index': None,
                    'news_sentiment': None,
                    'reddit_sentiment': None
                }
                
                # Add Fear & Greed Index
                if fear_greed_df is not None:
                    daily_fg = fear_greed_df[fear_greed_df.index.date == date.date()]
                    if not daily_fg.empty:
                        daily_sentiment['fear_greed_index'] = daily_fg['value'].mean()
                
                # Add news sentiment
                if news_df is not None:
                    daily_news = news_df[news_df.index.date == date.date()]
                    if not daily_news.empty:
                        daily_sentiment['news_sentiment'] = daily_news['sentiment'].mean()
                
                # Add Reddit sentiment
                if reddit_df is not None:
                    daily_reddit = reddit_df[reddit_df.index.date == date.date()]
                    if not daily_reddit.empty:
                        daily_sentiment['reddit_sentiment'] = daily_reddit['sentiment'].mean()
                
                sentiment_data.append(daily_sentiment)
            
            # Create DataFrame
            df = pd.DataFrame(sentiment_data)
            df = df.set_index('date')
            
            # Calculate composite sentiment (simple average of available indicators)
            df['composite_sentiment'] = df[['fear_greed_index', 'news_sentiment', 'reddit_sentiment']].mean(axis=1)
            
            return df
            
        except Exception as e:
            print(f"Error calculating aggregated sentiment: {e}")
            return None