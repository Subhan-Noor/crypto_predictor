"""
Broader market indicators (BTC dominance, total market cap)
Trading volume across exchanges
Correlation data between coins
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Optional, Dict, List
import time


class MarketDataFetcher:
    """
    Fetches broader market data using free APIs.
    Includes market caps, trading volumes, and correlations.
    """
    
    def __init__(self, cache_dir='data/market'):
        """
        Initialize the market data fetcher.
        
        Args:
            cache_dir (str): Directory to store cached market data
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_global_market_data(self) -> Optional[Dict]:
        """
        Fetch global crypto market data from CoinGecko
        
        Returns:
            Optional[Dict]: Dictionary with market data or None if failed
        """
        try:
            # Cache file path
            cache_file = os.path.join(self.cache_dir, 'global_market.json')
            
            # Check if we have cached data less than 1 hour old
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                if (datetime.now() - datetime.fromtimestamp(cached_data['timestamp'])).total_seconds() < 3600:
                    return cached_data
            
            # Fetch new data
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"CoinGecko API error: {response.status_code}")
                return None
            
            data = response.json()['data']
            
            # Add timestamp
            data['timestamp'] = int(datetime.now().timestamp())
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
            
        except Exception as e:
            print(f"Error fetching global market data: {e}")
            return None
    
    def get_trading_volume(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch 24h trading volume across major exchanges
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with exchange volumes or None if failed
        """
        try:
            # Cache file path
            cache_file = os.path.join(self.cache_dir, f'volume_{symbol}.csv')
            
            # Check if we have cached data less than 1 hour old
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                if (datetime.now() - pd.to_datetime(df['timestamp'].max())).total_seconds() < 3600:
                    return df.set_index('exchange')
            
            # Fetch new data from CryptoCompare
            url = f"https://min-api.cryptocompare.com/data/top/exchanges"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 30
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"CryptoCompare API error: {response.status_code}")
                return None
            
            data = response.json()
            
            if data['Response'] != 'Success':
                print(f"CryptoCompare API error: {data.get('Message')}")
                return None
            
            # Process exchange data
            exchanges_data = []
            timestamp = datetime.now()
            
            for exchange in data['Data']:
                exchanges_data.append({
                    'exchange': exchange['exchange'],
                    'volume_24h': exchange['volume24h'],
                    'volume_24h_to': exchange['volume24hTo'],
                    'market_share': exchange['volume24h'] / sum(e['volume24h'] for e in data['Data']),
                    'timestamp': timestamp
                })
            
            df = pd.DataFrame(exchanges_data)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            
            return df.set_index('exchange')
            
        except Exception as e:
            print(f"Error fetching trading volume: {e}")
            return None
    
    def calculate_correlations(self, symbols: List[str], days: int = 30) -> Optional[pd.DataFrame]:
        """
        Calculate price correlations between different cryptocurrencies
        
        Args:
            symbols (List[str]): List of cryptocurrency symbols
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: Correlation matrix or None if failed
        """
        try:
            # Cache file path
            symbols_key = '_'.join(sorted(symbols))
            cache_file = os.path.join(self.cache_dir, f'correlations_{symbols_key}_{days}d.csv')
            
            # Check if we have cached data less than 1 day old
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0)
                if (datetime.now() - pd.to_datetime(df.index.max())).days < 1:
                    return df
            
            # Fetch historical prices for all symbols
            prices_data = {}
            
            for symbol in symbols:
                # Use CryptoCompare API
                url = "https://min-api.cryptocompare.com/data/v2/histoday"
                params = {
                    'fsym': symbol,
                    'tsym': 'USD',
                    'limit': days
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code != 200:
                    print(f"CryptoCompare API error for {symbol}: {response.status_code}")
                    continue
                
                data = response.json()
                
                if data['Response'] != 'Success':
                    print(f"CryptoCompare API error for {symbol}: {data.get('Message')}")
                    continue
                
                # Extract closing prices
                prices = pd.DataFrame(data['Data']['Data'])
                prices['time'] = pd.to_datetime(prices['time'], unit='s')
                prices = prices.set_index('time')
                prices_data[symbol] = prices['close']
                
                # Rate limiting
                time.sleep(0.1)
            
            if not prices_data:
                return None
            
            # Create price DataFrame
            price_df = pd.DataFrame(prices_data)
            
            # Calculate correlations
            correlations = price_df.corr()
            
            # Cache the data
            correlations.to_csv(cache_file)
            
            return correlations
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return None
    
    def get_btc_dominance(self) -> Optional[float]:
        """
        Calculate Bitcoin's market dominance
        
        Returns:
            Optional[float]: BTC dominance percentage or None if failed
        """
        try:
            market_data = self.get_global_market_data()
            if market_data:
                return market_data['market_cap_percentage']['btc']
            return None
        except Exception as e:
            print(f"Error calculating BTC dominance: {e}")
            return None
    
    def get_market_summary(self) -> Optional[Dict]:
        """
        Get a summary of key market metrics
        
        Returns:
            Optional[Dict]: Dictionary with market summary or None if failed
        """
        try:
            market_data = self.get_global_market_data()
            if not market_data:
                return None
            
            return {
                'total_market_cap_usd': market_data['total_market_cap']['usd'],
                'total_volume_24h_usd': market_data['total_volume']['usd'],
                'btc_dominance': market_data['market_cap_percentage']['btc'],
                'active_cryptocurrencies': market_data['active_cryptocurrencies'],
                'markets': market_data['markets'],
                'timestamp': datetime.fromtimestamp(market_data['timestamp'])
            }
            
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return None