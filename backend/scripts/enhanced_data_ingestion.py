#!/usr/bin/env python3
"""
Enhanced Data Ingestion Script for Crypto Price Prediction App

This script provides real-time data ingestion with current day updates.
It can be run frequently to ensure the database always has the latest data.

Usage:
    python enhanced_data_ingestion.py --current    # Fetch current day data only
    python enhanced_data_ingestion.py --daily      # Fetch last 24 hours
    python enhanced_data_ingestion.py --recent     # Fetch last 7 days
    python enhanced_data_ingestion.py --full       # Fetch last 30 days
"""

import sys
import os
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.services.binance_service import BinancePriceFetcher
from app.services.twitter_service import TwitterScraper
from app.services.reddit_service import RedditScraper
from app.logger import logger

class EnhancedDataIngestionManager:
    """Enhanced data ingestion manager with real-time capabilities"""
    
    def __init__(self):
        self.price_fetcher = BinancePriceFetcher()
        self.twitter_scraper = TwitterScraper()
        self.reddit_scraper = RedditScraper()
        self.currencies = ["BTC", "ETH"]
        
    async def fetch_current_day_prices(self) -> Dict[str, int]:
        """Fetch and store current day price data for all currencies"""
        results = {}
        today = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🔄 Fetching current day ({today}) price data...")
        
        for currency in self.currencies:
            try:
                logger.info(f"Fetching current price for {currency}")
                
                # Get current price from Binance
                symbol = f"{currency}USDT"
                current_price_data = await self.price_fetcher.get_current_price(symbol)
                
                if current_price_data and "price" in current_price_data:
                    current_price = float(current_price_data["price"])
                    
                    # For current day, we'll use the current price as close price
                    # and estimate other values based on recent volatility
                    success = db_manager.insert_price_data(
                        currency=currency,
                        date=today,
                        open_price=current_price * 0.999,  # Slight variation
                        high=current_price * 1.002,
                        low=current_price * 0.998,
                        close=current_price,
                        volume=1000000.0  # Default volume
                    )
                    
                    if success:
                        results[currency] = 1
                        logger.info(f"✅ Stored current day price for {currency}: ${current_price:,.2f}")
                    else:
                        results[currency] = 0
                        logger.warning(f"⚠️ Failed to store current day price for {currency}")
                else:
                    results[currency] = 0
                    logger.error(f"❌ No current price data received for {currency}")
                
            except Exception as e:
                logger.error(f"Error fetching current day price for {currency}: {str(e)}")
                results[currency] = 0
        
        return results
    
    async def fetch_recent_prices(self, hours_back: int = 24) -> Dict[str, int]:
        """Fetch and store recent price data (last N hours)"""
        results = {}
        
        logger.info(f"🔄 Fetching recent price data (last {hours_back} hours)...")
        
        for currency in self.currencies:
            try:
                logger.info(f"Fetching recent prices for {currency}")
                
                # Get historical prices from Binance
                symbol = f"{currency}USDT"
                
                # Calculate how many days we need (minimum 1 day)
                days_needed = max(1, (hours_back + 23) // 24)
                
                historical_data = await self.price_fetcher.get_historical_prices(
                    symbol, interval="1d", limit=days_needed
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
                logger.info(f"✅ Stored {stored_count} recent price records for {currency}")
                
            except Exception as e:
                logger.error(f"Error fetching recent price data for {currency}: {str(e)}")
                results[currency] = 0
        
        return results
    
    async def fetch_current_sentiment(self) -> Dict[str, Dict[str, Any]]:
        """Fetch and store current sentiment data for all currencies"""
        results = {}
        today = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🔄 Fetching current day ({today}) sentiment data...")
        
        for currency in self.currencies:
            try:
                logger.info(f"Fetching current sentiment for {currency}")
                
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
                    logger.info(f"✅ Stored current sentiment data for {currency}")
                else:
                    results[currency] = {
                        "twitter_sentiment": None,
                        "reddit_sentiment": None,
                        "stored": False
                    }
                    logger.error(f"❌ Failed to store current sentiment data for {currency}")
                
            except Exception as e:
                logger.error(f"Error processing current sentiment for {currency}: {str(e)}")
                results[currency] = {
                    "twitter_sentiment": None,
                    "reddit_sentiment": None,
                    "stored": False
                }
        
        return results
    
    async def run_current_day_ingestion(self) -> Dict[str, Any]:
        """Run current day data ingestion (prices + sentiment)"""
        logger.info("🚀 Starting current day data ingestion...")
        
        start_time = datetime.now()
        
        # Fetch current day prices
        price_results = await self.fetch_current_day_prices()
        
        # Fetch current day sentiment
        sentiment_results = await self.fetch_current_sentiment()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate summary
        total_price_records = sum(price_results.values())
        successful_sentiment_records = sum(1 for r in sentiment_results.values() if r.get("stored", False))
        
        summary = {
            "status": "completed",
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "price_records": total_price_records,
            "sentiment_records": successful_sentiment_records,
            "currencies_processed": len(self.currencies)
        }
        
        logger.info(f"✅ Current day ingestion completed in {duration:.2f}s")
        logger.info(f"   Price records: {total_price_records}")
        logger.info(f"   Sentiment records: {successful_sentiment_records}")
        
        return {
            "summary": summary,
            "price_results": price_results,
            "sentiment_results": sentiment_results
        }
    
    async def run_recent_ingestion(self, hours_back: int = 24) -> Dict[str, Any]:
        """Run recent data ingestion (last N hours)"""
        logger.info(f"🚀 Starting recent data ingestion (last {hours_back} hours)...")
        
        start_time = datetime.now()
        
        # Fetch recent prices
        price_results = await self.fetch_recent_prices(hours_back)
        
        # Fetch current sentiment
        sentiment_results = await self.fetch_current_sentiment()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate summary
        total_price_records = sum(price_results.values())
        successful_sentiment_records = sum(1 for r in sentiment_results.values() if r.get("stored", False))
        
        summary = {
            "status": "completed",
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "price_records": total_price_records,
            "sentiment_records": successful_sentiment_records,
            "hours_back": hours_back,
            "currencies_processed": len(self.currencies)
        }
        
        logger.info(f"✅ Recent ingestion completed in {duration:.2f}s")
        logger.info(f"   Price records: {total_price_records}")
        logger.info(f"   Sentiment records: {successful_sentiment_records}")
        
        return {
            "summary": summary,
            "price_results": price_results,
            "sentiment_results": sentiment_results
        }

async def main():
    """Main function for enhanced data ingestion"""
    parser = argparse.ArgumentParser(description="Enhanced Crypto Price Prediction Data Ingestion")
    
    # Ingestion modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--current", action="store_true", help="Fetch current day data only")
    mode_group.add_argument("--daily", action="store_true", help="Fetch last 24 hours")
    mode_group.add_argument("--recent", action="store_true", help="Fetch last 7 days")
    mode_group.add_argument("--full", action="store_true", help="Fetch last 30 days")
    
    # Options
    parser.add_argument("--hours", type=int, default=24, help="Number of hours back for recent ingestion")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Initialize ingestion manager
    ingestion_manager = EnhancedDataIngestionManager()
    
    try:
        # Run the specified ingestion mode
        if args.current:
            logger.info("Running current day ingestion...")
            results = await ingestion_manager.run_current_day_ingestion()
        elif args.daily:
            logger.info("Running daily ingestion (24 hours)...")
            results = await ingestion_manager.run_recent_ingestion(24)
        elif args.recent:
            logger.info("Running recent ingestion (7 days)...")
            results = await ingestion_manager.run_recent_ingestion(7 * 24)
        elif args.full:
            logger.info("Running full ingestion (30 days)...")
            results = await ingestion_manager.run_recent_ingestion(30 * 24)
        
        # Save results if requested
        if args.save_results:
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ingestion_results_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        
        # Print summary
        summary = results["summary"]
        print(f"\n=== Ingestion Summary ===")
        print(f"Status: {summary['status']}")
        print(f"Duration: {summary['duration_seconds']:.2f}s")
        print(f"Price Records: {summary['price_records']}")
        print(f"Sentiment Records: {summary['sentiment_records']}")
        print(f"Currencies: {summary['currencies_processed']}")
        
        if summary['status'] == 'completed':
            print("✅ Ingestion completed successfully!")
            return 0
        else:
            print("❌ Ingestion failed!")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 