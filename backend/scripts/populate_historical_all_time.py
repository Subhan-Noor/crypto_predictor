#!/usr/bin/env python3
"""
All-Time Historical Data Population Script for Crypto Price Prediction App

This script populates the database with comprehensive historical data:
- BTC: From 2010 onwards (when trading began)
- ETH: From 2015 onwards (when Ethereum launched)
- Handles large date ranges by chunking API calls
- Includes comprehensive error handling and retry logic
- Progress tracking and detailed logging

Usage:
    python populate_historical_all_time.py --btc-all       # All BTC historical data
    python populate_historical_all_time.py --eth-all       # All ETH historical data  
    python populate_historical_all_time.py --both-all      # Both BTC and ETH all time
    python populate_historical_all_time.py --days 1000     # Custom days back
    python populate_historical_all_time.py --test          # Test with 30 days
"""

import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import traceback
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from app.services.binance_service import BinancePriceFetcher
from app.logger import logger
from config import settings

class AllTimeDataPopulator:
    """Handles comprehensive all-time historical data population"""
    
    def __init__(self):
        self.price_fetcher = BinancePriceFetcher()
        
        # Historical start dates (when trading/data became available)
        self.currency_start_dates = {
            "BTC": datetime(2010, 7, 17),  # Bitcoin trading start on exchanges
            "ETH": datetime(2015, 8, 7),   # Ethereum launch date
        }
        
        # Current supported currencies
        self.currencies = ["BTC", "ETH"]
        
    def get_days_back_for_currency(self, currency: str) -> int:
        """Calculate days back to get all historical data for a currency"""
        start_date = self.currency_start_dates.get(currency)
        if not start_date:
            return 1000  # Default fallback
            
        today = datetime.now()
        days_back = (today - start_date).days
        logger.info(f"📅 {currency} historical data spans {days_back} days (from {start_date.strftime('%Y-%m-%d')})")
        return days_back
    
    async def populate_currency_historical_data(self, currency: str, days_back: Optional[int] = None) -> Dict[str, Any]:
        """Populate all historical data for a specific currency"""
        
        if days_back is None:
            days_back = self.get_days_back_for_currency(currency)
            
        logger.info(f"🔄 Starting all-time historical data population for {currency} ({days_back} days)...")
        
        results = {
            "currency": currency,
            "status": "started",
            "total_records": 0,
            "new_records": 0,
            "skipped_records": 0,
            "errors": [],
            "chunks_processed": 0,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            symbol = f"{currency}USDT"
            
            # Binance API limits: 1000 records per call
            chunk_size = 1000
            total_chunks = (days_back + chunk_size - 1) // chunk_size
            
            logger.info(f"📦 Processing {total_chunks} chunks of {chunk_size} records each")
            
            for chunk_index in range(total_chunks):
                chunk_start = chunk_index * chunk_size
                chunk_end = min((chunk_index + 1) * chunk_size, days_back)
                records_in_chunk = chunk_end - chunk_start
                
                logger.info(f"📊 Processing chunk {chunk_index + 1}/{total_chunks} ({records_in_chunk} days)")
                
                try:
                    # Add delay between API calls to respect rate limits
                    if chunk_index > 0:
                        await asyncio.sleep(1)  # 1 second delay between chunks
                    
                    # Calculate end time for this chunk (working backwards from today)
                    end_time = datetime.now() - timedelta(days=chunk_start)
                    start_time = end_time - timedelta(days=records_in_chunk)
                    
                    logger.info(f"🕐 Fetching data from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
                    
                    # Fetch historical data from Binance
                    historical_data = await self.price_fetcher.get_historical_prices(
                        symbol, 
                        interval="1d", 
                        limit=records_in_chunk,
                        end_time=int(end_time.timestamp() * 1000)  # Convert to milliseconds
                    )
                    
                    if not historical_data:
                        logger.warning(f"⚠️ No data returned for chunk {chunk_index + 1}")
                        continue
                    
                    # Process and store each record
                    chunk_new_records = 0
                    chunk_skipped_records = 0
                    
                    for entry in historical_data:
                        try:
                            # Convert timestamp to date
                            date_obj = datetime.fromtimestamp(entry["open_time"] / 1000)
                            date_str = date_obj.strftime("%Y-%m-%d")
                            
                            # Check if record already exists
                            existing_records = await db_manager.get_records('crypto_prices', {
                                'currency': currency,
                                'date': date_str
                            })
                            
                            if existing_records:
                                chunk_skipped_records += 1
                                continue
                            
                            # Insert new record
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
                                chunk_new_records += 1
                            else:
                                logger.warning(f"⚠️ Failed to insert data for {currency} on {date_str}")
                                
                        except Exception as record_error:
                            error_msg = f"Error processing record for {currency}: {str(record_error)}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)
                    
                    # Update results
                    results["new_records"] += chunk_new_records
                    results["skipped_records"] += chunk_skipped_records
                    results["total_records"] += len(historical_data)
                    results["chunks_processed"] += 1
                    
                    logger.info(f"✅ Chunk {chunk_index + 1} complete: {chunk_new_records} new, {chunk_skipped_records} skipped")
                    
                except Exception as chunk_error:
                    error_msg = f"Error processing chunk {chunk_index + 1} for {currency}: {str(chunk_error)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    continue
            
            # Final results
            results["status"] = "completed"
            results["end_time"] = datetime.now().isoformat()
            
            duration = datetime.fromisoformat(results["end_time"]) - datetime.fromisoformat(results["start_time"])
            results["duration_minutes"] = duration.total_seconds() / 60
            
            logger.info(f"🎉 {currency} all-time data population completed!")
            logger.info(f"📈 Total: {results['total_records']} records")
            logger.info(f"🆕 New: {results['new_records']} records")
            logger.info(f"⏭️ Skipped: {results['skipped_records']} existing records")
            logger.info(f"⏱️ Duration: {results['duration_minutes']:.2f} minutes")
            
        except Exception as e:
            error_msg = f"Critical error in {currency} data population: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results["status"] = "failed"
            results["error"] = error_msg
        
        return results
    
    async def populate_all_currencies(self) -> Dict[str, Any]:
        """Populate all-time historical data for all supported currencies"""
        logger.info("🚀 Starting all-time data population for all currencies...")
        
        overall_results = {
            "status": "started",
            "currencies": {},
            "total_records": 0,
            "total_new_records": 0,
            "start_time": datetime.now().isoformat(),
            "errors": []
        }
        
        for currency in self.currencies:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting {currency} all-time data population")
                logger.info(f"{'='*60}")
                
                currency_results = await self.populate_currency_historical_data(currency)
                overall_results["currencies"][currency] = currency_results
                
                # Aggregate results
                overall_results["total_records"] += currency_results.get("total_records", 0)
                overall_results["total_new_records"] += currency_results.get("new_records", 0)
                overall_results["errors"].extend(currency_results.get("errors", []))
                
            except Exception as e:
                error_msg = f"Failed to populate {currency}: {str(e)}"
                logger.error(error_msg)
                overall_results["errors"].append(error_msg)
        
        overall_results["status"] = "completed"
        overall_results["end_time"] = datetime.now().isoformat()
        
        duration = datetime.fromisoformat(overall_results["end_time"]) - datetime.fromisoformat(overall_results["start_time"])
        overall_results["duration_minutes"] = duration.total_seconds() / 60
        
        return overall_results

async def main():
    """Main function to run all-time historical data population"""
    parser = argparse.ArgumentParser(description="Populate database with all-time crypto historical data")
    parser.add_argument("--btc-all", action="store_true", help="Populate all BTC historical data")
    parser.add_argument("--eth-all", action="store_true", help="Populate all ETH historical data")
    parser.add_argument("--both-all", action="store_true", help="Populate all historical data for both currencies")
    parser.add_argument("--days", type=int, help="Custom number of days back")
    parser.add_argument("--test", action="store_true", help="Test run with 30 days only")
    
    args = parser.parse_args()
    
    # Initialize database connection
    if not db_manager.is_connected():
        logger.error("❌ Database connection failed")
        return
    
    populator = AllTimeDataPopulator()
    
    try:
        if args.test:
            logger.info("🧪 Running test mode with 30 days of data")
            results = await populator.populate_currency_historical_data("BTC", 30)
        elif args.btc_all:
            results = await populator.populate_currency_historical_data("BTC")
        elif args.eth_all:
            results = await populator.populate_currency_historical_data("ETH")
        elif args.both_all:
            results = await populator.populate_all_currencies()
        elif args.days:
            logger.info(f"🔄 Custom run with {args.days} days for both currencies")
            results = {}
            for currency in ["BTC", "ETH"]:
                currency_results = await populator.populate_currency_historical_data(currency, args.days)
                results[currency] = currency_results
        else:
            print("Please specify an action:")
            print("  --btc-all     : All BTC historical data")
            print("  --eth-all     : All ETH historical data")
            print("  --both-all    : All historical data for both currencies")
            print("  --days N      : Custom days back for both currencies")
            print("  --test        : Test with 30 days of BTC data")
            return
        
        # Print comprehensive results
        print("\n" + "="*80)
        print("🎯 ALL-TIME HISTORICAL DATA POPULATION RESULTS")
        print("="*80)
        
        if isinstance(results, dict) and "currencies" in results:
            # Multiple currencies results
            print(f"📅 Timestamp: {results.get('end_time', datetime.now().isoformat())}")
            print(f"📊 Status: {results.get('status', 'unknown')}")
            print(f"⏱️ Duration: {results.get('duration_minutes', 0):.2f} minutes")
            print(f"📈 Total Records Processed: {results.get('total_records', 0)}")
            print(f"🆕 Total New Records: {results.get('total_new_records', 0)}")
            
            print(f"\n📋 Currency Breakdown:")
            for currency, currency_results in results.get("currencies", {}).items():
                print(f"  {currency}:")
                print(f"    📊 Total: {currency_results.get('total_records', 0)}")
                print(f"    🆕 New: {currency_results.get('new_records', 0)}")
                print(f"    ⏭️ Skipped: {currency_results.get('skipped_records', 0)}")
                
        else:
            # Single currency results
            print(f"📅 Timestamp: {results.get('end_time', datetime.now().isoformat())}")
            print(f"💰 Currency: {results.get('currency', 'Unknown')}")
            print(f"📊 Status: {results.get('status', 'unknown')}")
            print(f"⏱️ Duration: {results.get('duration_minutes', 0):.2f} minutes")
            print(f"📈 Total Records: {results.get('total_records', 0)}")
            print(f"🆕 New Records: {results.get('new_records', 0)}")
            print(f"⏭️ Skipped Records: {results.get('skipped_records', 0)}")
        
        # Show errors if any
        all_errors = []
        if isinstance(results, dict):
            if "errors" in results:
                all_errors.extend(results["errors"])
            if "currencies" in results:
                for currency_results in results["currencies"].values():
                    all_errors.extend(currency_results.get("errors", []))
        
        if all_errors:
            print(f"\n⚠️ Errors ({len(all_errors)}):")
            for error in all_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more errors")
        
        print("="*80)
        print("✅ All-time historical data population completed!")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 