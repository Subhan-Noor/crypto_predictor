import requests
from typing import Dict

class BinancePriceFetcher:
    """Service to fetch cryptocurrency price data using Binance Public REST API"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    def get_current_price(self, symbol: str) -> Dict:
        """Get current price for a cryptocurrency symbol"""
        url = f"{self.base_url}/ticker/price"
        params = {
            "symbol": symbol
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def get_historical_prices(self, symbol: str, interval: str = "1d", limit: int = 30) -> Dict:
        """Get historical price data for a cryptocurrency symbol"""
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return [{
            "open_time": entry[0],
            "open": entry[1],
            "high": entry[2],
            "low": entry[3],
            "close": entry[4],
            "volume": entry[5],
            "close_time": entry[6]
        } for entry in data] 