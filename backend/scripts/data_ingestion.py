#!/usr/bin/env python3
"""
Data Ingestion Script for Crypto Price Prediction App
Updated for new data acquisition stack: Twint, Pushshift API, Binance
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.services.binance_service import BinancePriceFetcher
from app.services.twitter_service import TwitterScraper
from app.services.reddit_service import RedditScraper
from app.logger import logger

class DataIngestionManager:
    """Manages data ingestion for the crypto prediction app"""
    
    def __init__(self):
        self.price_fetcher = BinancePriceFetcher()
        self.twitter_scraper = TwitterScraper()
        self.reddit_scraper = RedditScraper()
        self.currencies = ["BTC", "ETH"]
        
    def fetch_and_store_prices(self, days_back: int = 30) -> Dict[str, int]:
        """Fetch and store price data for all currencies"""
        results = {}
        
        for currency in self.currencies:
            try:
                logger.info(f"Fetching price data for {currency}")
                
                # Get historical prices from Binance (now async)
                symbol = f"{currency}USDT"
                
                # Run async function in sync context
                import asyncio
                historical_data = asyncio.run(
                    self.price_fetcher.get_historical_prices(
                        symbol, interval="1d", limit=days_back
                    )
                )
                
                # Store in database
                stored_count = 0
                for entry in historical_data:
                    date_str = datetime.fromtimestamp(entry["open_time"] / 1000).strftime("%Y-%m-%d")
                    
                    success = db_manager.insert_price_data(
                        currency=currency,
                        date=date_str,
                        open_price=float(entry["open"]),
                        high=float(entry["high"]),
                        low=float(entry["low"]),
                        close=float(entry["close"]),
                        volume=float(entry["volume"])
                    )
                    
                    if success:
                        stored_count += 1
                
                results[currency] = stored_count
                logger.info(f"Stored {stored_count} price records for {currency}")
                
            except Exception as e:
                logger.error(f"Error fetching price data for {currency}: {str(e)}")
                results[currency] = 0
        
        return results
    
    def fetch_and_store_sentiment(self) -> Dict[str, Dict[str, int]]:
        """Fetch and store sentiment data for all currencies"""
        results = {}
        today = datetime.now().strftime("%Y-%m-%d")
        
        for currency in self.currencies:
            try:
                logger.info(f"Fetching sentiment data for {currency}")
                
                # Get Twitter sentiment
                twitter_sentiment = None
                try:
                    twitter_sentiment = self.twitter_scraper.get_crypto_sentiment(currency)
                    logger.info(f"Twitter sentiment for {currency}: {twitter_sentiment}")
                except Exception as e:
                    logger.warning(f"Error fetching Twitter sentiment for {currency}: {str(e)}")
                
                # Get Reddit sentiment
                reddit_sentiment = None
                try:
                    reddit_sentiment = self.reddit_scraper.get_crypto_sentiment(currency)
                    logger.info(f"Reddit sentiment for {currency}: {reddit_sentiment}")
                except Exception as e:
                    logger.warning(f"Error fetching Reddit sentiment for {currency}: {str(e)}")
                
                # Store in database
                success = db_manager.insert_sentiment_data(
                    currency=currency,
                    date=today,
                    twitter_sentiment=twitter_sentiment,
                    reddit_sentiment=reddit_sentiment
                )
                
                if success:
                    results[currency] = {
                        "twitter_sentiment": twitter_sentiment,
                        "reddit_sentiment": reddit_sentiment,
                        "stored": True
                    }
                    logger.info(f"Stored sentiment data for {currency}")
                else:
                    results[currency] = {
                        "twitter_sentiment": None,
                        "reddit_sentiment": None,
                        "stored": False
                    }
                    logger.error(f"Failed to store sentiment data for {currency}")
                
            except Exception as e:
                logger.error(f"Error processing sentiment for {currency}: {str(e)}")
                results[currency] = {
                    "twitter_sentiment": None,
                    "reddit_sentiment": None,
                    "stored": False
                }
        
        return results
    
    def run_full_ingestion(self, days_back: int = 30) -> Dict[str, Any]:
        """Run complete data ingestion process"""
        logger.info("Starting full data ingestion process")
        
        # Check database connection
        if not db_manager.is_connected():
            logger.error("Database connection not available")
            return {"error": "Database connection failed"}
        
        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection test failed")
            return {"error": "Database connection test failed"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "prices": {},
            "sentiment": {},
            "summary": {}
        }
        
        try:
            # Fetch and store price data
            logger.info("Fetching price data...")
            price_results = self.fetch_and_store_prices(days_back)
            results["prices"] = price_results
            
            # Fetch and store sentiment data
            logger.info("Fetching sentiment data...")
            sentiment_results = self.fetch_and_store_sentiment()
            results["sentiment"] = sentiment_results
            
            # Generate summary
            total_price_records = sum(price_results.values())
            successful_sentiment = sum(1 for r in sentiment_results.values() if r.get("stored", False))
            
            results["summary"] = {
                "total_price_records": total_price_records,
                "successful_sentiment_records": successful_sentiment,
                "currencies_processed": len(self.currencies),
                "status": "completed"
            }
            
            logger.info(f"Ingestion completed: {total_price_records} price records, {successful_sentiment} sentiment records")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            results["error"] = str(e)
            results["summary"]["status"] = "failed"
        
        return results

def main():
    """Main function for running data ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Price Prediction Data Ingestion")
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Number of days of historical data to fetch (default: 30)"
    )
    parser.add_argument(
        "--prices-only", 
        action="store_true", 
        help="Only fetch price data, skip sentiment"
    )
    parser.add_argument(
        "--sentiment-only", 
        action="store_true", 
        help="Only fetch sentiment data, skip prices"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    ingestion_manager = DataIngestionManager()
    
    if args.prices_only:
        logger.info("Running price-only ingestion")
        results = ingestion_manager.fetch_and_store_prices(args.days)
        print(f"Price ingestion results: {results}")
    elif args.sentiment_only:
        logger.info("Running sentiment-only ingestion")
        results = ingestion_manager.fetch_and_store_sentiment()
        print(f"Sentiment ingestion results: {results}")
    else:
        logger.info("Running full ingestion")
        results = ingestion_manager.run_full_ingestion(args.days)
        print(f"Full ingestion results: {results}")

if __name__ == "__main__":
    main() 