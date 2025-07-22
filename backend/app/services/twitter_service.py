import snscrape.modules.twitter as sntwitter
from typing import List

class TwitterScraper:
    """Service to scrape Twitter data using snscrape"""

    def __init__(self):
        pass

    def fetch_tweets(self, query: str, limit: int = 100) -> List[str]:
        """Fetch tweets for a given query using snscrape"""
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)
        return tweets

    def get_crypto_sentiment(self, currency: str) -> float:
        """Get sentiment for a specific cryptocurrency"""
        query = f"#{currency} OR ${currency} OR {currency.lower()} crypto"
        tweets = self.fetch_tweets(query, limit=50)
        # Placeholder for sentiment analysis logic
        return 0.0  # Replace with actual sentiment analysis 