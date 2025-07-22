import requests
import tweepy
import praw
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from transformers import pipeline
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings
from app.logger import logger


class FearGreedAPI:
    """Service to fetch Fear & Greed Index data"""
    
    def __init__(self):
        self.base_url = settings.fear_greed_api_url
    
    def get_current_index(self) -> Optional[Dict]:
        """Get current Fear & Greed Index"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]
            return None
            
        except requests.RequestException as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    def get_historical_index(self, limit: int = 30) -> Optional[List[Dict]]:
        """Get historical Fear & Greed Index data"""
        try:
            url = f"{self.base_url}?limit={limit}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except requests.RequestException as e:
            logger.error(f"Error fetching historical Fear & Greed Index: {e}")
            return None


class TwitterSentimentAPI:
    """Service to fetch and analyze Twitter sentiment"""
    
    def __init__(self):
        self.api = None
        self.sentiment_analyzer = None
        self._initialize_twitter_api()
        self._initialize_sentiment_analyzer()
    
    def _initialize_twitter_api(self):
        """Initialize Twitter API connection"""
        try:
            if settings.twitter_bearer_token:
                self.api = tweepy.Client(bearer_token=settings.twitter_bearer_token)
                logger.info("Twitter API initialized successfully")
            else:
                logger.warning("Twitter Bearer Token not found")
        except Exception as e:
            logger.error(f"Error initializing Twitter API: {e}")
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
    
    def fetch_tweets(self, query: str, max_results: int = 100) -> List[str]:
        """Fetch tweets for a given query"""
        if not self.api:
            return []
        
        try:
            tweets = tweepy.Paginator(
                self.api.search_recent_tweets,
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics']
            ).flatten(limit=max_results)
            
            return [tweet.text for tweet in tweets if tweet.text]
            
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return []
    
    def analyze_sentiment(self, texts: List[str]) -> float:
        """Analyze sentiment of a list of texts"""
        if not self.sentiment_analyzer or not texts:
            return 0.0
        
        try:
            sentiments = []
            for text in texts:
                # Clean text
                cleaned_text = text.replace('\n', ' ').strip()
                if len(cleaned_text) < 10:  # Skip very short texts
                    continue
                
                result = self.sentiment_analyzer(cleaned_text)
                
                # Convert to numerical score (-1 to 1)
                score = 0.0
                for label_score in result[0]:
                    if label_score['label'] == 'LABEL_2':  # Positive
                        score += label_score['score']
                    elif label_score['label'] == 'LABEL_0':  # Negative
                        score -= label_score['score']
                    # LABEL_1 is neutral, contributes 0
                
                sentiments.append(score)
            
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def get_crypto_sentiment(self, currency: str) -> float:
        """Get sentiment for a specific cryptocurrency"""
        query = f"#{currency} OR ${currency} OR {currency.lower()} crypto -is:retweet lang:en"
        tweets = self.fetch_tweets(query, max_results=50)
        return self.analyze_sentiment(tweets)


class RedditSentimentAPI:
    """Service to fetch and analyze Reddit sentiment"""
    
    def __init__(self):
        self.reddit = None
        self.sentiment_analyzer = None
        self._initialize_reddit_api()
        self._initialize_sentiment_analyzer()
    
    def _initialize_reddit_api(self):
        """Initialize Reddit API connection"""
        try:
            if settings.reddit_client_id and settings.reddit_client_secret:
                self.reddit = praw.Reddit(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent
                )
                logger.info("Reddit API initialized successfully")
            else:
                logger.warning("Reddit credentials not found")
        except Exception as e:
            logger.error(f"Error initializing Reddit API: {e}")
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
    
    def fetch_posts(self, subreddit: str, limit: int = 50) -> List[str]:
        """Fetch posts from a subreddit"""
        if not self.reddit:
            return []
        
        try:
            posts = []
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            for post in subreddit_obj.hot(limit=limit):
                if post.selftext and len(post.selftext) > 20:
                    posts.append(post.selftext)
                if post.title:
                    posts.append(post.title)
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    def analyze_sentiment(self, texts: List[str]) -> float:
        """Analyze sentiment of a list of texts"""
        if not self.sentiment_analyzer or not texts:
            return 0.0
        
        try:
            sentiments = []
            for text in texts:
                # Clean text
                cleaned_text = text.replace('\n', ' ').strip()
                if len(cleaned_text) < 10:  # Skip very short texts
                    continue
                
                result = self.sentiment_analyzer(cleaned_text)
                
                # Convert to numerical score (-1 to 1)
                score = 0.0
                for label_score in result[0]:
                    if label_score['label'] == 'LABEL_2':  # Positive
                        score += label_score['score']
                    elif label_score['label'] == 'LABEL_0':  # Negative
                        score -= label_score['score']
                    # LABEL_1 is neutral, contributes 0
                
                sentiments.append(score)
            
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def get_crypto_sentiment(self, currency: str) -> float:
        """Get sentiment for a specific cryptocurrency from relevant subreddits"""
        subreddits = ["cryptocurrency", "CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets"]
        
        if currency.upper() == "BTC":
            subreddits.extend(["Bitcoin", "btc"])
        elif currency.upper() == "ETH":
            subreddits.extend(["ethereum", "ethtrader"])
        
        all_posts = []
        for subreddit in subreddits:
            posts = self.fetch_posts(subreddit, limit=20)
            # Filter posts that mention the currency
            relevant_posts = [
                post for post in posts 
                if currency.lower() in post.lower() or currency.upper() in post
            ]
            all_posts.extend(relevant_posts)
        
        return self.analyze_sentiment(all_posts)


class SentimentService:
    """Main service to coordinate sentiment data collection"""
    
    def __init__(self):
        self.fear_greed = FearGreedAPI()
        self.twitter = TwitterSentimentAPI()
        self.reddit = RedditSentimentAPI()
    
    def get_sentiment_data(self, currency: str) -> Dict:
        """Get comprehensive sentiment data for a currency"""
        data = {
            "currency": currency.upper(),
            "date": datetime.now(),
            "fear_greed_index": None,
            "twitter_sentiment": None,
            "reddit_sentiment": None
        }
        
        # Get Fear & Greed Index (applies to overall crypto market)
        fear_greed_data = self.fear_greed.get_current_index()
        if fear_greed_data:
            data["fear_greed_index"] = int(fear_greed_data.get("value", 0))
        
        # Get Twitter sentiment
        twitter_sentiment = self.twitter.get_crypto_sentiment(currency)
        if twitter_sentiment is not None:
            data["twitter_sentiment"] = round(twitter_sentiment, 4)
        
        # Get Reddit sentiment
        reddit_sentiment = self.reddit.get_crypto_sentiment(currency)
        if reddit_sentiment is not None:
            data["reddit_sentiment"] = round(reddit_sentiment, 4)
        
        return data 