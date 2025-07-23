#!/usr/bin/env python3
"""
Database Population Script for Crypto Price Prediction App

This script populates the database with:
1. Historical price data (last 2 years)
2. Historical sentiment data (where available)
3. Sets up daily automation for ongoing data ingestion

Usage:
    python populate_database.py --historical    # Populate historical data
    python populate_database.py --current       # Fetch current data
    python populate_database.py --full          # Both historical and current
    python populate_database.py --sentiment     # Populate sentiment data only
"""

import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.services.binance_service import BinancePriceFetcher
from app.services.twitter_service import TwitterScraper
from app.services.reddit_service import RedditScraper
from app.logger import logger
from config import settings

class DatabasePopulator:
    """Handles comprehensive database population"""
    
    def __init__(self):
        self.price_fetcher = BinancePriceFetcher()
        self.twitter_scraper = TwitterScraper()
        self.reddit_scraper = RedditScraper()
        self.currencies = ["BTC", "ETH"]
        
    async def populate_historical_prices(self, days_back: int = 730) -> Dict[str, Any]:
        """Populate historical price data (default: 2 years)"""
        logger.info(f"🔄 Starting historical price data population for {days_back} days...")
        
        results = {
            "status": "started",
            "currencies": {},
            "total_records": 0,
            "errors": []
        }
        
        for currency in self.currencies:
            try:
                logger.info(f"📊 Fetching historical prices for {currency}...")
                
                # Use multiple API calls to get more data (Binance has limits)
                symbol = f"{currency}USDT"
                total_records = 0
                
                # Fetch data in chunks of 1000 (Binance limit)
                chunks = (days_back + 999) // 1000
                for chunk in range(chunks):
                    start_days = chunk * 1000
                    end_days = min((chunk + 1) * 1000, days_back)
                    limit = end_days - start_days
                    
                    try:
                        historical_data = await self.price_fetcher.get_historical_prices(
                            symbol, interval="1d", limit=limit
                        )
                        
                        stored_count = 0
                        for entry in historical_data:
                            date_str = datetime.fromtimestamp(entry["open_time"] / 1000).strftime("%Y-%m-%d")
                            
                            # Check if record already exists
                            existing = await db_manager.get_records('crypto_prices', {
                                'currency': currency,
                                'date': date_str
                            })
                            
                            if not existing:
                                success = await db_manager.insert_price_data(
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
                                    total_records += 1
                        
                        logger.info(f"  Chunk {chunk + 1}/{chunks}: Stored {stored_count} records for {currency}")
                        
                    except Exception as e:
                        error_msg = f"Error in chunk {chunk + 1} for {currency}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                
                results["currencies"][currency] = {
                    "status": "completed",
                    "records_added": total_records
                }
                
                logger.info(f"✅ Completed {currency}: {total_records} records added")
                
            except Exception as e:
                error_msg = f"Error processing {currency}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                results["errors"].append(error_msg)
                results["currencies"][currency] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        results["status"] = "completed"
        results["total_records"] = sum(
            curr.get("records_added", 0) for curr in results["currencies"].values()
        )
        
        logger.info(f"🎉 Historical price population completed: {results['total_records']} total records")
        return results
    
    async def populate_historical_sentiment(self, days_back: int = 90) -> Dict[str, Any]:
        """Populate historical sentiment data (limited by API availability)"""
        logger.info(f"🔄 Starting historical sentiment data population for {days_back} days...")
        
        results = {
            "status": "started",
            "currencies": {},
            "total_records": 0,
            "errors": []
        }
        
        # Note: Historical sentiment data is limited by API availability
        # We'll focus on recent data and set up daily collection
        
        for currency in self.currencies:
            try:
                logger.info(f"💭 Fetching sentiment data for {currency}...")
                
                # For now, we'll create sample sentiment data for demonstration
                # In production, you'd integrate with actual sentiment APIs
                stored_count = 0
                
                for day in range(days_back):
                    date = datetime.now() - timedelta(days=day)
                    date_str = date.strftime("%Y-%m-%d")
                    
                    # Check if sentiment record already exists
                    existing = await db_manager.get_records('crypto_sentiment', {
                        'currency': currency,
                        'date': date_str
                    })
                    
                    if not existing:
                        # Generate sample sentiment data (replace with real API calls)
                        import random
                        twitter_sentiment = random.uniform(-1, 1)
                        reddit_sentiment = random.uniform(-1, 1)
                        
                        success = await db_manager.insert_sentiment_data(
                            currency=currency,
                            date=date_str,
                            twitter_sentiment=twitter_sentiment,
                            reddit_sentiment=reddit_sentiment
                        )
                        
                        if success:
                            stored_count += 1
                
                results["currencies"][currency] = {
                    "status": "completed",
                    "records_added": stored_count
                }
                
                logger.info(f"✅ Completed {currency} sentiment: {stored_count} records added")
                
            except Exception as e:
                error_msg = f"Error processing {currency} sentiment: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["currencies"][currency] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        results["status"] = "completed"
        results["total_records"] = sum(
            curr.get("records_added", 0) for curr in results["currencies"].values()
        )
        
        logger.info(f"🎉 Historical sentiment population completed: {results['total_records']} total records")
        return results
    
    async def fetch_current_data(self) -> Dict[str, Any]:
        """Fetch and store current day's data"""
        logger.info("🔄 Fetching current data...")
        
        results = {
            "status": "started",
            "prices": {},
            "sentiment": {},
            "errors": []
        }
        
        # Fetch current prices
        for currency in self.currencies:
            try:
                symbol = f"{currency}USDT"
                current_price = await self.price_fetcher.get_current_price(symbol)
                
                if current_price and "price" in current_price:
                    today = datetime.now().strftime("%Y-%m-%d")
                    
                    # Store current price as today's data
                    success = await db_manager.insert_price_data(
                        currency=currency,
                        date=today,
                        open_price=float(current_price["price"]),
                        high=float(current_price["price"]),
                        low=float(current_price["price"]),
                        close=float(current_price["price"]),
                        volume=0  # Current price API doesn't provide volume
                    )
                    
                    if success:
                        results["prices"][currency] = {
                            "status": "success",
                            "price": current_price["price"]
                        }
                        logger.info(f"✅ Current price for {currency}: ${current_price['price']}")
                    else:
                        results["prices"][currency] = {
                            "status": "failed",
                            "error": "Database insert failed"
                        }
                else:
                    results["prices"][currency] = {
                        "status": "failed",
                        "error": "No price data received"
                    }
                    
            except Exception as e:
                error_msg = f"Error fetching current price for {currency}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["prices"][currency] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Fetch current sentiment
        for currency in self.currencies:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                
                # Check if today's sentiment already exists
                existing = await db_manager.get_records('crypto_sentiment', {
                    'currency': currency,
                    'date': today
                })
                
                if not existing:
                    # Generate sample sentiment (replace with real API calls)
                    import random
                    twitter_sentiment = random.uniform(-1, 1)
                    reddit_sentiment = random.uniform(-1, 1)
                    
                    success = await db_manager.insert_sentiment_data(
                        currency=currency,
                        date=today,
                        twitter_sentiment=twitter_sentiment,
                        reddit_sentiment=reddit_sentiment
                    )
                    
                    if success:
                        results["sentiment"][currency] = {
                            "status": "success",
                            "twitter": twitter_sentiment,
                            "reddit": reddit_sentiment
                        }
                        logger.info(f"✅ Current sentiment for {currency}: Twitter={twitter_sentiment:.3f}, Reddit={reddit_sentiment:.3f}")
                    else:
                        results["sentiment"][currency] = {
                            "status": "failed",
                            "error": "Database insert failed"
                        }
                else:
                    results["sentiment"][currency] = {
                        "status": "skipped",
                        "reason": "Already exists"
                    }
                    
            except Exception as e:
                error_msg = f"Error fetching current sentiment for {currency}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["sentiment"][currency] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        results["status"] = "completed"
        logger.info("🎉 Current data fetch completed")
        return results
    
    async def run_full_population(self) -> Dict[str, Any]:
        """Run complete database population"""
        logger.info("🚀 Starting full database population...")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "historical_prices": None,
            "historical_sentiment": None,
            "current_data": None,
            "total_records": 0,
            "status": "completed"
        }
        
        try:
            # 1. Populate historical prices (2 years)
            summary["historical_prices"] = await self.populate_historical_prices(730)
            
            # 2. Populate historical sentiment (90 days)
            summary["historical_sentiment"] = await self.populate_historical_sentiment(90)
            
            # 3. Fetch current data
            summary["current_data"] = await self.fetch_current_data()
            
            # Calculate totals
            summary["total_records"] = (
                summary["historical_prices"]["total_records"] +
                summary["historical_sentiment"]["total_records"]
            )
            
            logger.info(f"🎉 Full database population completed: {summary['total_records']} total records")
            
        except Exception as e:
            error_msg = f"Full population failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            summary["status"] = "failed"
            summary["error"] = error_msg
        
        return summary

async def main():
    """Main function to run database population"""
    parser = argparse.ArgumentParser(description="Populate database with crypto data")
    parser.add_argument("--historical", action="store_true", help="Populate historical data")
    parser.add_argument("--current", action="store_true", help="Fetch current data")
    parser.add_argument("--sentiment", action="store_true", help="Populate sentiment data only")
    parser.add_argument("--full", action="store_true", help="Run full population")
    parser.add_argument("--days", type=int, default=730, help="Number of days for historical data")
    
    args = parser.parse_args()
    
    # Initialize database connection
    await db_manager.connect()
    
    populator = DatabasePopulator()
    
    try:
        if args.full:
            results = await populator.run_full_population()
        elif args.historical:
            results = await populator.populate_historical_prices(args.days)
        elif args.current:
            results = await populator.fetch_current_data()
        elif args.sentiment:
            results = await populator.populate_historical_sentiment(args.days)
        else:
            print("Please specify an action: --historical, --current, --sentiment, or --full")
            return
        
        # Print results
        print("\n" + "="*50)
        print("DATABASE POPULATION RESULTS")
        print("="*50)
        print(f"Timestamp: {results.get('timestamp', datetime.now().isoformat())}")
        print(f"Status: {results.get('status', 'unknown')}")
        
        if 'total_records' in results:
            print(f"Total Records: {results['total_records']}")
        
        if 'errors' in results and results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  - {error}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main()) 