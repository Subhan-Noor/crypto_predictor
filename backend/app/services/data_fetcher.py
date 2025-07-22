import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings
from app.logger import logger


class CoinGeckoAPI:
    """Service to fetch cryptocurrency data from CoinGecko API"""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.headers = {}
        if settings.coingecko_api_key:
            self.headers["x-cg-demo-api-key"] = settings.coingecko_api_key
    
    def get_historical_prices(
        self, 
        coin_id: str, 
        vs_currency: str = "usd", 
        days: int = 365
    ) -> Optional[List[Dict]]:
        """
        Fetch historical price data for a cryptocurrency
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            vs_currency: Currency to get prices in (default: 'usd')
            days: Number of days of historical data to fetch
            
        Returns:
            List of price data dictionaries or None if error
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": "daily"
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to list of dictionaries with proper structure
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            historical_data = []
            for i, (timestamp, price) in enumerate(prices):
                date = datetime.fromtimestamp(timestamp / 1000)
                volume = volumes[i][1] if i < len(volumes) else 0
                
                historical_data.append({
                    "date": date,
                    "price": price,
                    "volume": volume
                })
            
            return historical_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching data from CoinGecko: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_ohlcv_data(
        self, 
        coin_id: str, 
        vs_currency: str = "usd", 
        days: int = 365
    ) -> Optional[List[Dict]]:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to get prices in
            days: Number of days of historical data
            
        Returns:
            List of OHLCV data dictionaries
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                "vs_currency": vs_currency,
                "days": days
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            ohlcv_data = []
            for entry in data:
                timestamp, open_price, high_price, low_price, close_price = entry
                date = datetime.fromtimestamp(timestamp / 1000)
                
                ohlcv_data.append({
                    "date": date,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": 0  # OHLC endpoint doesn't provide volume, use market_chart for volume
                })
            
            return ohlcv_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None
    
    def get_current_price(self, coin_id: str, vs_currency: str = "usd") -> Optional[Dict]:
        """
        Get current price for a cryptocurrency
        
        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Currency to get price in
            
        Returns:
            Dictionary with current price data
        """
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": vs_currency,
                "include_24hr_change": True,
                "include_24hr_vol": True
            }
            
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get(coin_id, {})
            
        except requests.RequestException as e:
            logger.error(f"Error fetching current price: {e}")
            return None


class CryptoPriceService:
    """Service to manage crypto price data operations"""
    
    def __init__(self):
        self.coingecko = CoinGeckoAPI()
        self.coin_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum"
        }
    
    def fetch_and_format_prices(self, currency: str, days: int = 365) -> Optional[List[Dict]]:
        """
        Fetch and format price data for storage
        
        Args:
            currency: Currency symbol (BTC or ETH)
            days: Number of days to fetch
            
        Returns:
            List of formatted price dictionaries
        """
        coin_id = self.coin_mapping.get(currency.upper())
        if not coin_id:
            logger.error(f"Unsupported currency: {currency}")
            return None
        
        # Get both OHLCV and volume data
        ohlcv_data = self.coingecko.get_ohlcv_data(coin_id, days=days)
        price_volume_data = self.coingecko.get_historical_prices(coin_id, days=days)
        
        if not ohlcv_data or not price_volume_data:
            logger.error(f"Failed to fetch OHLCV or price/volume data for {currency}")
            return None
        
        # Create volume lookup by date
        volume_lookup = {
            data["date"].date(): data["volume"] 
            for data in price_volume_data
        }
        
        # Combine OHLCV with volume data
        formatted_data = []
        for ohlcv in ohlcv_data:
            date_key = ohlcv["date"].date()
            volume = volume_lookup.get(date_key, 0)
            
            formatted_data.append({
                "currency": currency.upper(),
                "date": ohlcv["date"],
                "open": ohlcv["open"],
                "high": ohlcv["high"],
                "low": ohlcv["low"],
                "close": ohlcv["close"],
                "volume": volume
            })
        
        return formatted_data 