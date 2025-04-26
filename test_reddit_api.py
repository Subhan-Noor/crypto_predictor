"""
Test script to verify Reddit API credentials
"""

from data_fetchers.sentiment_fetcher import SentimentFetcher
import os
from dotenv import load_dotenv

def test_reddit_connection():
    # Load environment variables
    load_dotenv()
    
    # Debug: Print environment variables (safely)
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    print("\nChecking environment variables:")
    print(f"REDDIT_CLIENT_ID exists: {client_id is not None}")
    print(f"REDDIT_CLIENT_SECRET exists: {client_secret is not None}")
    
    if not client_id or not client_secret:
        print("\n❌ Environment variables not found!")
        print("Please make sure:")
        print("1. You have created a .env file in the project root directory")
        print("2. The .env file contains:")
        print("   REDDIT_CLIENT_ID=IHcf1DicWKyo3qrf1B6dBg")
        print("   REDDIT_CLIENT_SECRET=DaYXUXemeV1wt8oqX38cOEu4QY3hFg")
        return
    
    # Initialize sentiment fetcher
    print("\nInitializing SentimentFetcher...")
    fetcher = SentimentFetcher()
    
    # Try to get Reddit sentiment for BTC
    print("Testing Reddit API connection...")
    sentiment_data = fetcher.get_reddit_sentiment('BTC', days=1)
    
    if sentiment_data is not None:
        print("✅ Successfully connected to Reddit API!")
        print(f"\nFound {len(sentiment_data)} posts")
        print("\nSample of sentiment data:")
        print(sentiment_data.head())
    else:
        print("❌ Failed to connect to Reddit API")
        print("Please check your credentials in the .env file")

if __name__ == "__main__":
    test_reddit_connection() 