"""
Generate and Upload Synthetic Sentiment Data to Supabase

This script generates plausible sentiment data for BTC and ETH for the exact dates
that exist in the price data, ensuring proper merging during model training.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# Try to load environment variables from .env if not already set
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. If you want to load from .env, run: pip install python-dotenv")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

assert SUPABASE_URL and SUPABASE_KEY, "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your environment or backend/.env."

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

CURRENCIES = ["BTC", "ETH"]


def get_existing_price_dates(currency):
    """Fetch existing price data dates from Supabase"""
    url = f"{SUPABASE_URL}/rest/v1/crypto_prices"
    params = {
        "select": "date",
        "currency": f"eq.{currency}",
        "order": "date.desc"
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        data = response.json()
        # Extract dates and convert to date objects
        dates = [datetime.fromisoformat(item['date'].split('T')[0]).date() for item in data]
        return dates
    else:
        print(f"Failed to fetch price dates for {currency}: {response.status_code}")
        return []


def generate_synthetic_sentiment_for_dates(currency, dates):
    """Generate synthetic sentiment data for specific dates"""
    data = []
    for date in dates:
        # Simulate sentiment: mostly neutral, with some random spikes
        twitter_sentiment = float(np.clip(np.random.normal(0, 0.2), -1, 1))
        reddit_sentiment = float(np.clip(np.random.normal(0, 0.2), -1, 1))
        
        # Occasionally inject strong sentiment (5% chance)
        if np.random.rand() < 0.05:
            twitter_sentiment = float(np.clip(np.random.normal(0.7, 0.1), -1, 1)) if np.random.rand() < 0.5 else float(np.clip(np.random.normal(-0.7, 0.1), -1, 1))
        if np.random.rand() < 0.05:
            reddit_sentiment = float(np.clip(np.random.normal(0.7, 0.1), -1, 1)) if np.random.rand() < 0.5 else float(np.clip(np.random.normal(-0.7, 0.1), -1, 1))
        
        data.append({
            "currency": currency,
            "date": date.isoformat(),
            "twitter_sentiment": twitter_sentiment,
            "reddit_sentiment": reddit_sentiment
        })
    return data


def upload_sentiment_to_supabase(sentiment_data):
    """Upload sentiment data to Supabase"""
    url = f"{SUPABASE_URL}/rest/v1/crypto_sentiment"
    success = 0
    failed = 0
    
    for row in sentiment_data:
        payload = {
            "currency": row["currency"],
            "date": row["date"],
            "twitter_sentiment": row["twitter_sentiment"],
            "reddit_sentiment": row["reddit_sentiment"]
        }
        r = requests.post(url, headers=HEADERS, json=payload)
        if r.status_code in (200, 201, 409):  # 409 = already exists
            success += 1
        else:
            failed += 1
            print(f"Failed to upload: {payload} | {r.status_code} | {r.text}")
    
    print(f"Uploaded {success}/{len(sentiment_data)} sentiment records.")
    if failed > 0:
        print(f"Failed to upload {failed} records.")


def main():
    all_data = []
    
    for currency in CURRENCIES:
        print(f"Fetching existing price dates for {currency}...")
        price_dates = get_existing_price_dates(currency)
        
        if not price_dates:
            print(f"No price data found for {currency}, skipping...")
            continue
        
        print(f"Found {len(price_dates)} price records for {currency}")
        print(f"Generating synthetic sentiment for {currency}...")
        
        # Generate sentiment for existing price dates
        data = generate_synthetic_sentiment_for_dates(currency, price_dates)
        all_data.extend(data)
    
    if all_data:
        print(f"Generated {len(all_data)} total sentiment records. Uploading to Supabase...")
        upload_sentiment_to_supabase(all_data)
        print("Done!")
    else:
        print("No data to upload. Check if price data exists in the database.")


if __name__ == "__main__":
    main() 