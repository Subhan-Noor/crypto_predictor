#!/usr/bin/env python3
"""
Clean Mock Data Script

This script identifies and removes mock/fallback data from the database
that was generated when the Binance API failed. It looks for unrealistic
price values and removes them.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import db_manager
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDataCleaner:
    """Cleaner for identifying and removing mock data from the database"""
    
    def __init__(self):
        self.mock_price_thresholds = {
            'BTC': {
                'min_realistic': 20000,  # Minimum realistic BTC price
                'max_realistic': 150000, # Maximum realistic BTC price
                'fallback_base': 45000   # Fallback base price used in mock data
            },
            'ETH': {
                'min_realistic': 1000,   # Minimum realistic ETH price
                'max_realistic': 8000,   # Maximum realistic ETH price
                'fallback_base': 2500    # Fallback base price used in mock data
            }
        }
    
    async def identify_mock_data(self, currency: str) -> list:
        """Identify records that are likely mock data"""
        logger.info(f"🔍 Identifying mock data for {currency}")
        
        try:
            # Get all price records for the currency
            records = await db_manager.get_crypto_prices(currency, limit=10000)
            
            if not records:
                logger.info(f"No price records found for {currency}")
                return []
            
            mock_records = []
            thresholds = self.mock_price_thresholds[currency]
            
            for record in records:
                close_price = float(record.get('close', 0))
                
                # Check if price is outside realistic bounds
                if (close_price < thresholds['min_realistic'] or 
                    close_price > thresholds['max_realistic']):
                    
                    # Additional check: if price is close to fallback base price
                    # with very low volume, it's likely mock data
                    volume = float(record.get('volume', 0))
                    price_diff_from_fallback = abs(close_price - thresholds['fallback_base'])
                    
                    if (price_diff_from_fallback < 5000 or  # Close to fallback price
                        volume < 100000):  # Very low volume
                        
                        mock_records.append({
                            'id': record.get('id'),
                            'date': record.get('date'),
                            'close': close_price,
                            'volume': volume,
                            'reason': 'unrealistic_price_or_volume'
                        })
            
            logger.info(f"Found {len(mock_records)} potential mock records for {currency}")
            return mock_records
            
        except Exception as e:
            logger.error(f"Error identifying mock data for {currency}: {e}")
            return []
    
    async def remove_mock_records(self, currency: str, mock_records: list) -> int:
        """Remove mock records from the database"""
        if not mock_records:
            logger.info(f"No mock records to remove for {currency}")
            return 0
        
        logger.info(f"🗑️ Removing {len(mock_records)} mock records for {currency}")
        
        try:
            client = db_manager.get_client()
            if not client:
                logger.error("Database client not available")
                return 0
            
            removed_count = 0
            
            for record in mock_records:
                try:
                    # Delete the record by ID
                    result = client.table("crypto_prices").delete().eq("id", record['id']).execute()
                    
                    if result.data:
                        removed_count += 1
                        logger.info(f"Removed mock record: {record['date']} - ${record['close']}")
                    else:
                        logger.warning(f"Failed to remove record {record['id']}")
                        
                except Exception as e:
                    logger.error(f"Error removing record {record['id']}: {e}")
            
            logger.info(f"✅ Successfully removed {removed_count} mock records for {currency}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error removing mock records for {currency}: {e}")
            return 0
    
    async def clean_currency_data(self, currency: str) -> dict:
        """Clean mock data for a specific currency"""
        logger.info(f"🧹 Starting mock data cleanup for {currency}")
        
        # Identify mock data
        mock_records = await self.identify_mock_data(currency)
        
        if not mock_records:
            return {
                'currency': currency,
                'mock_records_found': 0,
                'records_removed': 0,
                'status': 'no_mock_data_found'
            }
        
        # Show summary of mock data found
        logger.info(f"📊 Mock data summary for {currency}:")
        for record in mock_records[:5]:  # Show first 5
            logger.info(f"  - {record['date']}: ${record['close']} (vol: {record['volume']})")
        
        if len(mock_records) > 5:
            logger.info(f"  ... and {len(mock_records) - 5} more records")
        
        # Ask for confirmation (in production, you might want to make this automatic)
        logger.info(f"⚠️ Found {len(mock_records)} potential mock records for {currency}")
        logger.info("Proceeding with removal...")
        
        # Remove mock records
        removed_count = await self.remove_mock_records(currency, mock_records)
        
        return {
            'currency': currency,
            'mock_records_found': len(mock_records),
            'records_removed': removed_count,
            'status': 'completed'
        }
    
    async def clean_all_data(self) -> dict:
        """Clean mock data for all currencies"""
        logger.info("🚀 Starting comprehensive mock data cleanup")
        
        results = {}
        total_removed = 0
        
        for currency in ['BTC', 'ETH']:
            result = await self.clean_currency_data(currency)
            results[currency] = result
            total_removed += result['records_removed']
        
        summary = {
            'total_records_removed': total_removed,
            'currency_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Mock data cleanup completed. Total records removed: {total_removed}")
        return summary

async def main():
    """Main function to run the mock data cleanup"""
    logger.info("🧹 Crypto Price Mock Data Cleanup Tool")
    logger.info("=" * 50)
    
    # Test database connection
    try:
        await db_manager.test_connection()
        logger.info("✅ Database connection successful")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # Create cleaner instance
    cleaner = MockDataCleaner()
    
    # Run cleanup
    try:
        summary = await cleaner.clean_all_data()
        
        # Print summary
        logger.info("\n📋 Cleanup Summary:")
        logger.info("=" * 30)
        for currency, result in summary['currency_results'].items():
            logger.info(f"{currency}: {result['mock_records_found']} found, {result['records_removed']} removed")
        
        logger.info(f"\nTotal records removed: {summary['total_records_removed']}")
        
        if summary['total_records_removed'] > 0:
            logger.info("\n🎉 Mock data cleanup completed successfully!")
            logger.info("💡 Your charts should now show real price data.")
        else:
            logger.info("\n✅ No mock data found. Database appears to be clean.")
            
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 