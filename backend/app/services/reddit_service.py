import requests
from typing import List, Dict

class RedditScraper:
    """Service to fetch Reddit data using the Pushshift API"""

    def __init__(self):
        self.base_url = "https://api.pushshift.io/reddit"

    def fetch_posts(self, subreddit: str, limit: int = 50) -> List[Dict]:
        """Fetch posts from a subreddit using Pushshift API"""
        url = f"{self.base_url}/search/submission/"
        params = {
            "subreddit": subreddit,
            "size": limit,
            "sort": "desc",
            "sort_type": "created_utc"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("data", [])

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
                if currency.lower() in post.get("title", "").lower() or currency.upper() in post.get("title", "")
            ]
            all_posts.extend(relevant_posts)

        # Placeholder for sentiment analysis logic
        return 0.0  # Replace with actual sentiment analysis 