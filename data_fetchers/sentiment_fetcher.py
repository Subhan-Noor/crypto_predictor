"""
Fetches and analyzes sentiment data from various sources:
- News API for news sentiment
- Reddit for social media sentiment
- Fear & Greed Index for market sentiment
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
import praw
from dotenv import load_dotenv
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import json
import time
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentFetcher:
    """
    Fetches and analyzes sentiment data from multiple sources.
    Implements real-time sentiment analysis using VADER and TextBlob.
    """
    
    def __init__(self, cache_dir='data/sentiment'):
        """
        Initialize the sentiment fetcher.
        
        Args:
            cache_dir (str): Directory to store cached sentiment data
        """
        # Load environment variables
        load_dotenv()
        
        self.cache_dir = cache_dir
        self.vader = SentimentIntensityAnalyzer()
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialize Reddit client
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        # Initialize News API client
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            print("Warning: NEWS_API_KEY not found in environment variables")
        
        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent='Crypto Sentiment Bot 1.0'
                )
                # Don't validate connection immediately - defer until needed
                # self.reddit.user.me()
            except Exception as e:
                print(f"Warning: Reddit API initialization failed: {e}")
                self.reddit = None
        else:
            print("Warning: Reddit credentials not found in environment variables")
            self.reddit = None

    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using VADER and TextBlob.
        Returns a weighted average of both scores.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']  # Already between -1 and 1
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity  # Already between -1 and 1
        
        # Return weighted average (VADER weighted more as it's better for social media)
        return 0.7 * vader_compound + 0.3 * textblob_polarity

    def get_fear_greed_index(self) -> Optional[pd.DataFrame]:
        """
        Get the Fear & Greed Index data from alternative.me API
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with fear & greed index data or None if failed
        """
        try:
            url = "https://api.alternative.me/fng/?limit=30&format=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'data' not in data:
                return None
                
            # Create DataFrame
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['value'] = df['value'].astype(float)
            
            # Normalize value to -1 to 1 range
            # Fear & Greed Index is 0-100, where 0 is extreme fear and 100 is extreme greed
            # We'll map 0-50 to -1-0 (fear) and 50-100 to 0-1 (greed)
            df['normalized_value'] = df['value'].apply(lambda x: (x - 50) / 50)
            
            df = df.set_index('timestamp').sort_index()
            return df
            
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None

    def get_crypto_news(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get crypto news from News API
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with news sentiment or None if failed
        """
        if not self.news_api_key:
            return None
            
        try:
            # Get news from the last 7 days (free API limit)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'cryptocurrency {symbol}',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'articles' not in data or not data['articles']:
                return None
                
            news_data = []
            for article in data['articles']:
                # Combine title and description for sentiment analysis
                text = f"{article['title']} {article['description'] or ''}"
                sentiment = self.analyze_text_sentiment(text)
                
                news_data.append({
                    'timestamp': pd.to_datetime(article['publishedAt']),
                    'title': article['title'],
                    'sentiment': sentiment  # Already between -1 and 1
                })
            
            if not news_data:
                return None
                
            # Create DataFrame and sort by date
            df = pd.DataFrame(news_data)
            df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            print(f"Error fetching news data: {e}")
            return None

    def get_reddit_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get real-time Reddit sentiment for a cryptocurrency
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with Reddit sentiment or None if failed
        """
        if not self.reddit:
            print("Reddit API not initialized, skipping Reddit sentiment")
            return None
            
        try:
            # Create empty list to store posts
            posts_data = []
            
            # Subreddits to search
            crypto_subreddits = [
                'CryptoCurrency', 
                'CryptoMarkets',
                f'{symbol}',
                'bitcoin'  # General crypto discussions
            ]
            
            for subreddit_name in crypto_subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Get recent hot posts
                    hot_posts = subreddit.hot(limit=10)
                    for post in hot_posts:
                        if post.selftext and len(post.selftext) > 20:
                            # Analyze sentiment
                            sentiment = self.analyze_text_sentiment(f"{post.title} {post.selftext}")
                            
                            posts_data.append({
                                'timestamp': datetime.fromtimestamp(post.created_utc),
                                'title': post.title,
                                'subreddit': subreddit_name,
                                'sentiment': sentiment,
                                'score': post.score,
                                'comments': post.num_comments
                            })
                    
                    # Get recent new posts
                    new_posts = subreddit.new(limit=10)
                    for post in new_posts:
                        if post.selftext and len(post.selftext) > 20:
                            # Analyze sentiment
                            sentiment = self.analyze_text_sentiment(f"{post.title} {post.selftext}")
                            
                            posts_data.append({
                                'timestamp': datetime.fromtimestamp(post.created_utc),
                                'title': post.title,
                                'subreddit': subreddit_name,
                                'sentiment': sentiment,
                                'score': post.score,
                                'comments': post.num_comments
                            })
                            
                except Exception as e:
                    print(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            if not posts_data:
                print(f"No Reddit data found for {symbol}")
                return None
                
            # Create DataFrame and sort by date
            df = pd.DataFrame(posts_data)
            df = df.set_index('timestamp').sort_index()
            
            # Weight sentiment by post popularity (score + comments)
            df['popularity'] = df['score'] + df['comments']
            df['weighted_sentiment'] = df['sentiment'] * df['popularity']
            
            return df
            
        except Exception as e:
            print(f"Error fetching Reddit sentiment: {e}")
            return None
    
    def get_historical_sentiment(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Get historical sentiment data for a cryptocurrency over a specified number of days
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            List[Dict]: List of daily sentiment data points
        """
        try:
            # Generate dates for the requested period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Try to get actual sentiment data
            news_sentiment = self.get_crypto_news(symbol)
            reddit_sentiment = self.get_reddit_sentiment(symbol)
            fear_greed = self.get_fear_greed_index()
            
            # Prepare the result
            result = []
            
            # If we have no real data, generate mock data
            if (news_sentiment is None or news_sentiment.empty) and \
               (reddit_sentiment is None or reddit_sentiment.empty) and \
               (fear_greed is None or fear_greed.empty):
                
                print(f"No sentiment data available for {symbol}, generating mock data")
                # Generate mock sentiment data
                base_sentiment = 0.2  # Slightly positive base sentiment
                
                for date in date_range:
                    # Add some randomness to the sentiment
                    daily_sentiment = base_sentiment + np.random.normal(0, 0.3)
                    # Clamp between -1 and 1
                    daily_sentiment = max(-1, min(1, daily_sentiment))
                    
                    result.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'sentiment': daily_sentiment,
                        'source': 'mock'
                    })
                    
                    # Randomly shift the base sentiment over time
                    base_sentiment += np.random.normal(0, 0.1)
                    base_sentiment = max(-0.5, min(0.5, base_sentiment))
                
                return result
            
            # If we have some real data, combine it with mock data for missing dates
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                sentiment_value = 0
                sources_used = []
                
                # Check if we have news sentiment for this date
                if news_sentiment is not None and not news_sentiment.empty:
                    day_news = news_sentiment[news_sentiment.index.date == date.date()]
                    if not day_news.empty:
                        sentiment_value += day_news['sentiment'].mean()
                        sources_used.append('news')
                
                # Check if we have Reddit sentiment for this date
                if reddit_sentiment is not None and not reddit_sentiment.empty:
                    day_reddit = reddit_sentiment[reddit_sentiment.index.date == date.date()]
                    if not day_reddit.empty:
                        if 'weighted_sentiment' in day_reddit.columns and 'popularity' in day_reddit.columns:
                            # Use weighted sentiment if available
                            total_popularity = day_reddit['popularity'].sum()
                            if total_popularity > 0:
                                weighted_avg = day_reddit['weighted_sentiment'].sum() / total_popularity
                                sentiment_value += weighted_avg
                            else:
                                sentiment_value += day_reddit['sentiment'].mean()
                        else:
                            sentiment_value += day_reddit['sentiment'].mean()
                        sources_used.append('reddit')
                
                # Check if we have Fear & Greed Index for this date
                if fear_greed is not None and not fear_greed.empty:
                    day_fg = fear_greed[fear_greed.index.date == date.date()]
                    if not day_fg.empty:
                        sentiment_value += day_fg['normalized_value'].mean()
                        sources_used.append('fear_greed')
                
                # If we have sources, calculate the average sentiment
                if sources_used:
                    sentiment_value /= len(sources_used)
                else:
                    # Generate mock data for dates with no real data
                    sentiment_value = np.random.normal(0.1, 0.3)  # Slightly positive with variance
                    sources_used = ['mock']
                
                # Clamp between -1 and 1
                sentiment_value = max(-1, min(1, sentiment_value))
                
                result.append({
                    'date': date_str,
                    'sentiment': sentiment_value,
                    'source': ','.join(sources_used)
                })
            
            return result
        
        except Exception as e:
            print(f"Error generating historical sentiment data: {e}")
            # Return mock data in case of error
            return self._generate_mock_sentiment(days)
    
    def _generate_mock_sentiment(self, days: int) -> List[Dict]:
        """Generate mock sentiment data when real data is not available"""
        result = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        base_sentiment = 0.2  # Slightly positive base sentiment
        
        for date in date_range:
            # Add some randomness to the sentiment
            daily_sentiment = base_sentiment + np.random.normal(0, 0.3)
            # Clamp between -1 and 1
            daily_sentiment = max(-1, min(1, daily_sentiment))
            
            result.append({
                'date': date.strftime('%Y-%m-%d'),
                'sentiment': daily_sentiment,
                'source': 'mock'
            })
            
            # Randomly shift the base sentiment over time
            base_sentiment += np.random.normal(0, 0.1)
            base_sentiment = max(-0.5, min(0.5, base_sentiment))
        
        return result

    def get_aggregated_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get aggregated sentiment from all available sources
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with aggregated sentiment or None if failed
        """
        try:
            # Get data from all sources - run in synchronous context
            # This avoids the asyncio.run() error in an async context
            fear_greed_df = self.get_fear_greed_index()
            news_df = self.get_crypto_news(symbol)
            reddit_df = self.get_reddit_sentiment(symbol)
            
            # Create daily sentiment scores
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            sentiment_data = []
            
            for date in date_range:
                data_point = {'date': date}
                has_data = False
                
                # Add Fear & Greed sentiment
                if fear_greed_df is not None:
                    mask = (fear_greed_df.index.date == date.date())
                    if mask.any():
                        value = fear_greed_df[mask]['normalized_value'].mean()
                        if pd.notna(value):
                            data_point['fear_greed'] = float(value)  # Already between -1 and 1
                            has_data = True
                
                # Add news sentiment
                if news_df is not None:
                    mask = (news_df.index.date == date.date())
                    if mask.any():
                        sentiments = news_df[mask]['sentiment'].dropna()
                        if not sentiments.empty:
                            data_point['news_sentiment'] = float(sentiments.mean())  # Already between -1 and 1
                            has_data = True
                
                # Add Reddit sentiment
                if reddit_df is not None:
                    mask = (reddit_df.index.date == date.date())
                    if mask.any():
                        reddit_data = reddit_df[mask].dropna(subset=['sentiment'])
                        if not reddit_data.empty:
                            # Weight sentiment by post score and comments
                            weights = reddit_data['score'] * (1 + np.log1p(reddit_data['comments']))
                            weighted_sentiment = np.average(
                                reddit_data['sentiment'],  # Already between -1 and 1
                                weights=weights
                            )
                            if not np.isnan(weighted_sentiment):
                                data_point['reddit_sentiment'] = float(weighted_sentiment)
                                has_data = True
                
                # Only add data points that have at least one source of sentiment
                if has_data:
                    sentiment_data.append(data_point)
            
            if not sentiment_data:
                print(f"No sentiment data available for {symbol}")
                return None
            
            # Create final DataFrame
            df = pd.DataFrame(sentiment_data)
            df = df.set_index('date')
            
            # Calculate aggregate sentiment (weighted average of available sources)
            weights = {
                'fear_greed': 0.3,
                'news_sentiment': 0.4,
                'reddit_sentiment': 0.3
            }
            
            # Get available sources and their weights
            available_sources = [col for col in weights.keys() if col in df.columns]
            if not available_sources:
                print(f"No sentiment sources available for {symbol}")
                return None
                
            # Normalize weights for available sources
            total_weight = sum(weights[source] for source in available_sources)
            normalized_weights = {source: weights[source]/total_weight for source in available_sources}
            
            # Calculate weighted average, replacing NaN with 0
            weighted_sum = 0
            for source, weight in normalized_weights.items():
                values = df[source].fillna(0)
                weighted_sum += values * weight
            df['sentiment'] = weighted_sum  # Will be between -1 and 1
            
            # Ensure all numeric columns have NaN filled with 0
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Convert all numeric columns to float
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Verify no NaN values remain
            if df.isna().any().any():
                print(f"Warning: NaN values found in columns: {df.columns[df.isna().any()].tolist()}")
                df = df.fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error calculating aggregated sentiment: {e}")
            import traceback
            traceback.print_exc()
            return None