import os
import sys
from typing import Optional
from supabase import create_client, Client
import logging

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages Supabase database connections and operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self):
        """Initialize Supabase client connection"""
        try:
            if not settings.supabase_url or not settings.supabase_key:
                logger.warning("Supabase credentials not found in environment variables")
                return
            
            self.client = create_client(settings.supabase_url, settings.supabase_key)
            logger.info("Successfully connected to Supabase")
            
        except Exception as e:
            logger.error(f"Error connecting to Supabase: {str(e)}")
            self.client = None
    
    def get_client(self) -> Optional[Client]:
        """Get the Supabase client instance"""
        if not self.client:
            self._connect()
        return self.client
    
    def is_connected(self) -> bool:
        """Check if database connection is available"""
        return self.client is not None
    
    def test_connection(self) -> bool:
        """Test database connection by making a simple query"""
        try:
            if not self.client:
                return False
            
            # Try a simple query to test connection
            result = self.client.table("crypto_prices").select("id").limit(1).execute()
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def insert_price_data(self, currency: str, date: str, open_price: float, 
                         high: float, low: float, close: float, volume: float) -> bool:
        """Insert price data into crypto_prices table"""
        try:
            if not self.client:
                logger.error("Database client not available")
                return False
            
            data = {
                "currency": currency.upper(),
                "date": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            }
            
            result = self.client.table("crypto_prices").upsert(data).execute()
            logger.info(f"Successfully inserted price data for {currency} on {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting price data: {str(e)}")
            return False
    
    def insert_sentiment_data(self, currency: str, date: str, 
                            twitter_sentiment: Optional[float] = None,
                            reddit_sentiment: Optional[float] = None) -> bool:
        """Insert sentiment data into crypto_sentiment table"""
        try:
            if not self.client:
                logger.error("Database client not available")
                return False
            
            data = {
                "currency": currency.upper(),
                "date": date
            }
            
            if twitter_sentiment is not None:
                data["twitter_sentiment"] = twitter_sentiment
            if reddit_sentiment is not None:
                data["reddit_sentiment"] = reddit_sentiment
            
            result = self.client.table("crypto_sentiment").upsert(data).execute()
            logger.info(f"Successfully inserted sentiment data for {currency} on {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting sentiment data: {str(e)}")
            return False
    
    def get_latest_prices(self, currency: str, limit: int = 30):
        """Get latest price data for a currency"""
        try:
            if not self.client:
                return None
            
            result = self.client.table("crypto_prices")\
                .select("*")\
                .eq("currency", currency.upper())\
                .order("date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return None
    
    def get_latest_sentiment(self, currency: str, limit: int = 30):
        """Get latest sentiment data for a currency"""
        try:
            if not self.client:
                return None
            
            result = self.client.table("crypto_sentiment")\
                .select("*")\
                .eq("currency", currency.upper())\
                .order("date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            return None

    async def get_records(self, table: str, filters: dict = None):
        """Get records from a table with optional filters"""
        try:
            if not self.client:
                return []
            
            query = self.client.table(table).select("*")
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            result = query.execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching records from {table}: {str(e)}")
            return []

# Global database manager instance
db_manager = DatabaseManager() 