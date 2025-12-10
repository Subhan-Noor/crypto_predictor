import os
import sys
from typing import Optional, List, Dict
from supabase import create_client, Client
import logging
from datetime import datetime, timedelta

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
    
    async def test_connection(self) -> bool:
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

    async def get_crypto_prices(self, currency: str, limit: int = 1000):
        """Get crypto price data for a currency"""
        try:
            if not self.client:
                return []
            
            result = self.client.table("crypto_prices")\
                .select("*")\
                .eq("currency", currency.upper())\
                .order("date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching crypto prices for {currency}: {str(e)}")
            return []

    async def get_crypto_sentiment(self, currency: str, limit: int = 1000):
        """Get crypto sentiment data for a currency"""
        try:
            if not self.client:
                return []
            
            result = self.client.table("crypto_sentiment")\
                .select("*")\
                .eq("currency", currency.upper())\
                .order("date", desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error fetching crypto sentiment for {currency}: {str(e)}")
            return []

    async def insert_prediction(self, prediction_data: dict) -> str:
        """Insert a prediction record into the predictions table and return the record ID."""
        try:
            if not self.client:
                logger.error("Database client not available")
                return None
            result = self.client.table("predictions").insert(prediction_data).execute()
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully inserted prediction record: {result.data[0].get('id')}")
                return result.data[0].get('id')
            logger.warning("Prediction insert returned no data")
            return None
        except Exception as e:
            logger.error(f"Error inserting prediction: {str(e)}")
            return None

    async def get_predictions(self, currency: str, days: int = 30, limit: int = 100) -> List[Dict]:
        """Get historical predictions for a currency"""
        try:
            if not self.client:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query predictions table
            result = self.client.table("predictions").select("*").eq("currency", currency).gte("prediction_date", start_date.isoformat()).order("prediction_date", desc=True).limit(limit).execute()
            
            predictions = result.data if result.data else []
            
            # Sort predictions to prioritize random_forest
            def sort_key(pred):
                model_version = pred.get('model_version', '')
                # Prioritize random_forest over logistic_regression
                if 'random_forest' in model_version:
                    return (pred.get('prediction_date', ''), 0)  # random_forest first
                else:
                    return (pred.get('prediction_date', ''), 1)  # others second
            
            predictions.sort(key=sort_key, reverse=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error fetching predictions for {currency}: {str(e)}")
            return []
    
    async def get_best_predictions(self, currency: str, days: int = 30, limit: int = 100) -> List[Dict]:
        """
        Get the best prediction for each date, prioritizing random_forest
        
        Args:
            currency: Currency to get predictions for
            days: Number of days to look back
            limit: Maximum number of predictions to return
            
        Returns:
            List of best predictions (one per date)
        """
        try:
            if not self.client:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query predictions table
            result = self.client.table("predictions").select("*").eq("currency", currency).gte("prediction_date", start_date.isoformat()).order("prediction_date", desc=True).limit(limit * 2).execute()
            
            predictions = result.data if result.data else []
            
            # Group predictions by date and select the best one for each date
            date_groups = {}
            for pred in predictions:
                pred_date = pred.get('prediction_date', '')
                if pred_date not in date_groups:
                    date_groups[pred_date] = []
                date_groups[pred_date].append(pred)
            
            # Select best prediction for each date (prioritize random_forest)
            best_predictions = []
            for date, preds in date_groups.items():
                # Sort predictions for this date: random_forest first, then by confidence
                def sort_key(pred):
                    model_version = pred.get('model_version', '')
                    confidence = pred.get('confidence_score', 0)
                    if 'random_forest' in model_version:
                        return (0, confidence)  # random_forest first
                    else:
                        return (1, confidence)  # others second
                
                preds.sort(key=sort_key, reverse=True)
                best_predictions.append(preds[0])  # Take the best one
            
            # Sort by date (newest first) and limit
            best_predictions.sort(key=lambda x: x.get('prediction_date', ''), reverse=True)
            return best_predictions[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching best predictions for {currency}: {str(e)}")
            return []

    async def update_prediction(self, prediction_id: str, update_data: dict) -> bool:
        """Update a prediction record with validation results"""
        try:
            if not self.client:
                logger.error("Database client not available")
                return False
            
            # Update the prediction record
            result = self.client.table("predictions").update(update_data).eq("id", prediction_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully updated prediction {prediction_id}")
                return True
            else:
                logger.warning(f"No prediction found with ID {prediction_id} or update failed")
                return False
                
        except Exception as e:
            logger.error(f"Error updating prediction {prediction_id}: {str(e)}")
            return False

    async def get_recent_predictions_for_monitoring(self, currency: str, days: int = 30) -> List[Dict]:
        """Get recent predictions with actual direction for performance monitoring"""
        try:
            if not self.client:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use prediction_date instead of date (based on other methods in this class)
            result = self.client.table("predictions")\
                .select("*")\
                .eq("currency", currency)\
                .gte("prediction_date", start_date.date().isoformat())\
                .lte("prediction_date", end_date.date().isoformat())\
                .order("prediction_date", desc=True)\
                .execute()
            
            # Filter out records without actual_direction in Python instead of SQL
            filtered_data = []
            for record in result.data:
                if record.get('actual_direction') is not None:
                    filtered_data.append({
                        'date': record.get('prediction_date'),  # Map prediction_date to date for compatibility
                        'predicted_direction': record.get('predicted_direction'),
                        'actual_direction': record.get('actual_direction'),
                        'confidence': record.get('confidence'),
                        'model_version': record.get('model_version'),
                        'is_correct': record.get('is_correct')
                    })
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error fetching predictions for monitoring {currency}: {str(e)}")
            return []

# Global database manager instance
db_manager = DatabaseManager() 