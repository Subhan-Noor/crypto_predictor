from supabase import create_client, Client
from typing import Optional
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from .logger import logger


class DatabaseManager:
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Supabase connection"""
        try:
            if settings.supabase_url and settings.supabase_key:
                self.supabase = create_client(
                    settings.supabase_url,
                    settings.supabase_key
                )
                logger.info("Successfully connected to Supabase")
            else:
                logger.warning("Supabase credentials not found. Database operations will not work.")
        except Exception as e:
            logger.error(f"Error connecting to Supabase: {e}")
            self.supabase = None
    
    def get_client(self) -> Optional[Client]:
        """Get Supabase client instance"""
        return self.supabase
    
    def is_connected(self) -> bool:
        """Check if database connection is established"""
        return self.supabase is not None


# Global database manager instance
db_manager = DatabaseManager() 