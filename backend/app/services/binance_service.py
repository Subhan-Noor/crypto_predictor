# Enhanced Binance Service with Fallback Mechanisms
import requests
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class BinancePriceFetcher:
    """Service to fetch cryptocurrency price data using Binance Public REST API with fallbacks"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.fallback_enabled = True
        
    async def get_current_price(self, symbol: str) -> Dict:
        """Get current price for a cryptocurrency symbol - NO FALLBACK DATA"""
        try:
            # Try Binance API first
            url = f"{self.base_url}/ticker/price"
            params = {"symbol": symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✅ Binance current price API success for {symbol}: ${data['price']}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                logger.error(f"❌ Binance current price API blocked (451) for {symbol} - NO FALLBACK DATA")
                raise Exception(f"Binance API blocked for {symbol}. No fallback data will be generated.")
            else:
                logger.error(f"❌ Binance current price API HTTP error for {symbol}: {e}")
                raise Exception(f"Binance API HTTP error for {symbol}: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Binance current price API request error for {symbol}: {e}")
            raise Exception(f"Binance API request error for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching current price for {symbol}: {e}")
            raise Exception(f"Unexpected error fetching current price for {symbol}: {e}")

    async def get_historical_prices(self, symbol: str, interval: str = "1d", limit: int = 30) -> List[Dict]:
        """Get historical price data for a cryptocurrency symbol - NO FALLBACK DATA"""
        try:
            # Try Binance API first
            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            result = [{
                "open_time": entry[0],
                "open": entry[1],
                "high": entry[2],
                "low": entry[3],
                "close": entry[4],
                "volume": entry[5],
                "close_time": entry[6]
            } for entry in data]
            
            logger.info(f"✅ Binance historical API success for {symbol}: {len(result)} records")
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 451:
                logger.error(f"❌ Binance historical API blocked (451) for {symbol} - NO FALLBACK DATA")
                raise Exception(f"Binance API blocked for {symbol}. No fallback data will be generated.")
            else:
                logger.error(f"❌ Binance historical API HTTP error for {symbol}: {e}")
                raise Exception(f"Binance API HTTP error for {symbol}: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Binance historical API request error for {symbol}: {e}")
            raise Exception(f"Binance API request error for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching historical prices for {symbol}: {e}")
            raise Exception(f"Unexpected error fetching historical prices for {symbol}: {e}")

    async def _get_fallback_current_price(self, symbol: str) -> Dict:
        """DEPRECATED: Fallback method for current price when Binance API fails - DO NOT USE"""
        logger.warning(f"⚠️ DEPRECATED: Using fallback price for {symbol} - This should not be used!")
        
        # Try alternative APIs
        try:
            # Try CoinGecko API as fallback
            coingecko_id = self._get_coingecko_id(symbol)
            if coingecko_id:
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {
                    "ids": coingecko_id,
                    "vs_currencies": "usd"
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if coingecko_id in data and "usd" in data[coingecko_id]:
                        price = data[coingecko_id]["usd"]
                        logger.info(f"✅ CoinGecko fallback success for {symbol}: ${price}")
                        return {"symbol": symbol, "price": str(price)}
        except Exception as e:
            logger.warning(f"CoinGecko fallback failed for {symbol}: {e}")
        
        # Generate realistic fallback price
        fallback_price = self._generate_realistic_price(symbol)
        logger.warning(f"⚠️ DEPRECATED: Generated fallback price for {symbol}: ${fallback_price}")
        
        return {
            "symbol": symbol,
            "price": str(fallback_price),
            "source": "fallback"
        }

    async def _get_fallback_historical_prices(self, symbol: str, limit: int) -> List[Dict]:
        """DEPRECATED: Fallback method for historical prices when Binance API fails - DO NOT USE"""
        logger.warning(f"⚠️ DEPRECATED: Using fallback historical prices for {symbol} - This should not be used!")
        
        # Generate realistic historical data
        base_price = self._get_base_price(symbol)
        historical_data = []
        
        for i in range(limit):
            # Generate date
            date = datetime.now() - timedelta(days=i)
            timestamp = int(date.timestamp() * 1000)
            
            # Generate realistic price movement
            price_change = random.uniform(-0.1, 0.1)  # ±10% daily change
            current_price = base_price * (1 + price_change)
            
            # Generate OHLCV data
            open_price = current_price
            high_price = current_price * random.uniform(1.0, 1.05)
            low_price = current_price * random.uniform(0.95, 1.0)
            close_price = current_price * random.uniform(0.98, 1.02)
            volume = random.uniform(1000000, 10000000)
            
            historical_data.append({
                "open_time": timestamp,
                "open": str(open_price),
                "high": str(high_price),
                "low": str(low_price),
                "close": str(close_price),
                "volume": str(volume),
                "close_time": timestamp + 86400000  # 24 hours in milliseconds
            })
            
            # Update base price for next iteration
            base_price = close_price
        
        # Reverse to get chronological order (oldest first)
        historical_data.reverse()
        
        logger.warning(f"⚠️ DEPRECATED: Generated {len(historical_data)} fallback historical records for {symbol}")
        return historical_data

    def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko ID for a symbol"""
        coingecko_mapping = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "BTC": "bitcoin",
            "ETH": "ethereum"
        }
        return coingecko_mapping.get(symbol)

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for generating realistic fallback data"""
        base_prices = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 2500.0,
            "BTC": 45000.0,
            "ETH": 2500.0
        }
        return base_prices.get(symbol, 100.0)

    def _generate_realistic_price(self, symbol: str) -> float:
        """Generate a realistic current price for fallback"""
        base_price = self._get_base_price(symbol)
        # Add some random variation (±5%)
        variation = random.uniform(-0.05, 0.05)
        return base_price * (1 + variation)

    def disable_fallback(self):
        """Disable fallback mechanisms (for testing)"""
        self.fallback_enabled = False

    def enable_fallback(self):
        """Enable fallback mechanisms"""
        self.fallback_enabled = True 