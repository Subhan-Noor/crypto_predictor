#!/usr/bin/env python3
"""
Data ingestion script for cryptocurrency price and sentiment data.
This script is designed to be run daily via cron job or GitHub Actions.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_fetcher import CryptoPriceService
from app.services.sentiment_fetcher import SentimentService
from app.database import db_manager
from app.models.crypto_models import CryptoPriceCreate, CryptoSentimentCreate
from app.logger import logger


class DataIngestionOrchestrator:
    """Orchestrates the data ingestion process"""
    
    def __init__(self):
        self.price_service = CryptoPriceService()
        self.sentiment_service = SentimentService()
        self.db_client = db_manager.get_client()
        self.currencies = ["BTC", "ETH"]
    
    def store_price_data(self, price_data: List[Dict]) -> bool:
        """Store price data in Supabase"""
        if not self.db_client:
            logger.error("Database connection not available")
            return False
        
        try:
            for data in price_data:
                # Check if record already exists
                existing = self.db_client.table("crypto_prices").select("id").eq("currency", data["currency"]).eq("date", data["date"].isoformat()).execute()
                
                if not existing.data:
                    # Insert new record
                    result = self.db_client.table("crypto_prices").insert({
                        "currency": data["currency"],
                        "date": data["date"].isoformat(),
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"]
                    }).execute()
                    logger.info(f"Stored price data for {data['currency']} on {data['date'].date()}")
                else:
                    logger.info(f"Price data for {data['currency']} on {data['date'].date()} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            return False
    
    def store_sentiment_data(self, sentiment_data: Dict) -> bool:
        """Store sentiment data in Supabase"""
        if not self.db_client:
            logger.error("Database connection not available")
            return False
        
        try:
            # Check if record already exists for today
            today = sentiment_data["date"].date()
            existing = self.db_client.table("crypto_sentiment").select("id").eq("currency", sentiment_data["currency"]).eq("date", today.isoformat()).execute()
            
            if not existing.data:
                # Insert new record
                result = self.db_client.table("crypto_sentiment").insert({
                    "currency": sentiment_data["currency"],
                    "date": sentiment_data["date"].isoformat(),
                    "fear_greed_index": sentiment_data["fear_greed_index"],
                    "twitter_sentiment": sentiment_data["twitter_sentiment"],
                    "reddit_sentiment": sentiment_data["reddit_sentiment"]
                }).execute()
                logger.info(f"Stored sentiment data for {sentiment_data['currency']} on {today}")
            else:
                # Update existing record
                result = self.db_client.table("crypto_sentiment").update({
                    "fear_greed_index": sentiment_data["fear_greed_index"],
                    "twitter_sentiment": sentiment_data["twitter_sentiment"],
                    "reddit_sentiment": sentiment_data["reddit_sentiment"]
                }).eq("currency", sentiment_data["currency"]).eq("date", today.isoformat()).execute()
                logger.info(f"Updated sentiment data for {sentiment_data['currency']} on {today}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
            return False
    
    def ingest_historical_price_data(self, days: int = 30) -> bool:
        """Ingest historical price data for the last N days"""
        logger.info(f"Starting historical price data ingestion for last {days} days...")
        
        success = True
        for currency in self.currencies:
            logger.info(f"Fetching price data for {currency}...")
            
            price_data = self.price_service.fetch_and_format_prices(currency, days=days)
            if price_data:
                if self.store_price_data(price_data):
                    logger.info(f"Successfully stored {len(price_data)} price records for {currency}")
                else:
                    logger.error(f"Failed to store price data for {currency}")
                    success = False
            else:
                logger.error(f"Failed to fetch price data for {currency}")
                success = False
        
        return success
    
    def ingest_current_sentiment_data(self) -> bool:
        """Ingest current sentiment data"""
        logger.info("Starting sentiment data ingestion...")
        
        success = True
        for currency in self.currencies:
            logger.info(f"Fetching sentiment data for {currency}...")
            
            sentiment_data = self.sentiment_service.get_sentiment_data(currency)
            if self.store_sentiment_data(sentiment_data):
                logger.info(f"Successfully stored sentiment data for {currency}")
            else:
                logger.error(f"Failed to store sentiment data for {currency}")
                success = False
        
        return success
    
    def run_daily_ingestion(self) -> bool:
        """Run daily data ingestion (prices + sentiment)"""
        logger.info(f"Starting daily data ingestion at {datetime.now()}")
        
        # Ingest latest price data (last 2 days to ensure we get today's data)
        price_success = self.ingest_historical_price_data(days=2)
        
        # Ingest current sentiment data
        sentiment_success = self.ingest_current_sentiment_data()
        
        overall_success = price_success and sentiment_success
        logger.info(f"Daily ingestion completed. Success: {overall_success}")
        
        return overall_success
    
    def run_initial_setup(self, days: int = 365) -> bool:
        """Run initial setup to populate historical data"""
        logger.info(f"Starting initial data setup for last {days} days...")
        
        # Ingest historical price data
        price_success = self.ingest_historical_price_data(days=days)
        
        # Ingest current sentiment data
        sentiment_success = self.ingest_current_sentiment_data()
        
        overall_success = price_success and sentiment_success
        logger.info(f"Initial setup completed. Success: {overall_success}")
        
        return overall_success


def main():
    """Main function to run data ingestion"""
    orchestrator = DataIngestionOrchestrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "daily":
            success = orchestrator.run_daily_ingestion()
        elif command == "initial":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
            success = orchestrator.run_initial_setup(days)
        elif command == "prices":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            success = orchestrator.ingest_historical_price_data(days)
        elif command == "sentiment":
            success = orchestrator.ingest_current_sentiment_data()
        else:
            logger.error("Usage: python data_ingestion.py [daily|initial|prices|sentiment] [days]")
            return False
    else:
        # Default to daily ingestion
        success = orchestrator.run_daily_ingestion()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 