"""
Support for multiple cryptocurrencies
Multiple API integrations (CoinGecko, CryptoCompare, Alpha Vantage)
Cache management for efficient data retrieval
Configurable time periods and granularity
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import warnings
import requests
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """
    Enhanced class to fetch historical cryptocurrency price data.
    Supports multiple cryptocurrencies and data sources with fallbacks.
    """
    
    # Default supported cryptocurrencies and their circulating supplies
    SUPPORTED_CRYPTOS = {
        'BTC': 19_600_000,  # Approximate Bitcoin supply
        'ETH': 120_000_000,  # Approximate Ethereum supply
        'XRP': 46_000_000_000,  # XRP supply
        'ADA': 35_000_000_000,  # Cardano supply
        'SOL': 400_000_000,  # Solana supply
        'DOT': 1_200_000_000,  # Polkadot supply
    }
    
    def __init__(self, cache_dir='data', use_sample_data=False):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir (str): Directory to store cached data
            use_sample_data (bool): Whether to use sample data instead of API calls
        """
        self.cache_dir = cache_dir
        self.use_sample_data = use_sample_data
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Create subdirectory for each cryptocurrency
        for crypto in self.SUPPORTED_CRYPTOS:
            crypto_cache_dir = os.path.join(cache_dir, crypto.lower())
            if not os.path.exists(crypto_cache_dir):
                os.makedirs(crypto_cache_dir)
        
        self.base_urls = {
            'cryptocompare': 'https://min-api.cryptocompare.com/data/v2/histoday',
            'alternative': 'https://api.alternative.me/v2/historical',
        }
        
        # Standard column names mapping
        self.column_mapping = {
            'volumefrom': 'volume',
            'volumeto': 'volume_to',
            'vol_24h': 'volume',
            'price_usd': 'close',
            'market_cap_usd': 'market_cap'
        }
    
    def fetch_historical_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for any supported cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH', 'XRP')
            days (int): Number of days of historical data to fetch
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical price data or None if failed
        """
        symbol = symbol.upper()
        if symbol not in self.SUPPORTED_CRYPTOS:
            raise ValueError(f"Unsupported cryptocurrency: {symbol}. Supported options: {list(self.SUPPORTED_CRYPTOS.keys())}")
        
        # If sample data is explicitly requested, skip API calls
        if self.use_sample_data:
            print(f"Using sample data for {symbol} as requested.")
            return self._generate_sample_data(symbol, days)
        
        # Try different data sources in order
        try:
            print(f"Fetching {symbol} historical data for the last {days} days...")
            
            # Try CryptoCompare API first
            df = self._fetch_from_cryptocompare(symbol, days)
            if df is not None and not df.empty:
                print(f"Successfully fetched {symbol} data from CryptoCompare.")
                return df
            
            # Try Alphavantage API next
            df = self._fetch_from_alphavantage(symbol, days)
            if df is not None and not df.empty:
                print(f"Successfully fetched {symbol} data from Alpha Vantage.")
                return df
            
            # Try Historic-Crypto as a fallback
            df = self._fetch_from_historic_crypto(symbol, days)
            if df is not None and not df.empty:
                print(f"Successfully fetched {symbol} data from Historic-Crypto.")
                return df
            
            # If all APIs fail, use sample data
            print("All API attempts failed, generating sample data instead.")
            return self._generate_sample_data(symbol, days)
            
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            print("Generating sample data instead.")
            return self._generate_sample_data(symbol, days)
    
    def _fetch_from_cryptocompare(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch data from CryptoCompare API for any cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if failed
        """
        try:
            print(f"Trying CryptoCompare API for {symbol} ({days} days)...")
            
            # CryptoCompare has a limit of 2000 days per request
            max_days_per_request = 2000
            all_data = []
            
            # Calculate number of requests needed
            remaining_days = days
            current_timestamp = int(datetime.now().timestamp())
            
            while remaining_days > 0:
                # Determine days for this request
                request_days = min(remaining_days, max_days_per_request)
                
                # Construct the API URL
                url = "https://min-api.cryptocompare.com/data/v2/histoday"
                params = {
                    "fsym": symbol,
                    "tsym": "USD",
                    "limit": request_days,
                    "toTs": current_timestamp
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code != 200:
                    print(f"CryptoCompare API error: {response.status_code}")
                    return None
                    
                data = response.json()
                
                if data.get('Response') == 'Error':
                    print(f"CryptoCompare API error: {data.get('Message')}")
                    return None
                    
                # Process the data
                crypto_data = data['Data']['Data']
                
                if not crypto_data:
                    print("No data returned from CryptoCompare API.")
                    break
                    
                # Add data to our collection
                all_data.extend(crypto_data)
                
                # Update for next request
                remaining_days -= request_days
                if remaining_days > 0:
                    current_timestamp = crypto_data[0]['time']
                    time.sleep(0.5)  # Rate limiting
            
            if not all_data:
                return None
                
            # Create DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('date')
            
            # Rename columns to match our standard
            df = df.rename(columns={
                'low': 'low',
                'high': 'high',
                'open': 'open',
                'close': 'close',
                'volumefrom': 'volume'
            })
            
            # Add market cap using circulating supply
            circulating_supply = self.SUPPORTED_CRYPTOS[symbol]
            df['market_cap'] = df['close'] * circulating_supply
            
            # Select and sort columns
            columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
            df = df[columns].sort_index()
            
            return df
            
        except Exception as e:
            print(f"Error with CryptoCompare API: {e}")
            return None
    
    def _fetch_from_alphavantage(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage API for any cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if failed
        """
        try:
            print(f"Trying Alpha Vantage API for {symbol}...")
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": symbol,
                "market": "USD",
                "apikey": api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Alpha Vantage API error: {response.status_code}")
                return None
                
            data = response.json()
            
            if 'Error Message' in data:
                print(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            
            if 'Time Series (Digital Currency Daily)' not in data:
                print("Alpha Vantage data format error or limit reached")
                return None
                
            # Process the data
            time_series = data['Time Series (Digital Currency Daily)']
            daily_data = []
            
            for date, values in time_series.items():
                daily_data.append({
                    'date': date,
                    'open': float(values['1a. open (USD)']),
                    'high': float(values['2a. high (USD)']),
                    'low': float(values['3a. low (USD)']),
                    'close': float(values['4a. close (USD)']),
                    'volume': float(values['5. volume']),
                    'market_cap': float(values['4a. close (USD)']) * self.SUPPORTED_CRYPTOS[symbol]
                })
            
            # Create DataFrame
            df = pd.DataFrame(daily_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Sort and limit to requested days
            df = df.sort_index()
            if len(df) > days:
                df = df.iloc[-days:]
                
            return df
            
        except Exception as e:
            print(f"Error with Alpha Vantage API: {e}")
            return None
    
    def _fetch_from_historic_crypto(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch data using Historic-Crypto library for any cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical data or None if failed
        """
        try:
            print(f"Trying Historic-Crypto API for {symbol} (Coinbase Pro)...")
            
            try:
                from Historic_Crypto import HistoricalData
            except ImportError:
                print("Historic-Crypto library not found. Install with: pip install Historic-Crypto")
                return None
            
            # Calculate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates
            start_date_str = start_date.strftime('%Y-%m-%d-%H-%M')
            end_date_str = end_date.strftime('%Y-%m-%d-%H-%M')
            
            # Fetch data
            data = HistoricalData(
                ticker=f'{symbol}-USD',
                start_date=start_date_str,
                end_date=end_date_str,
                granularity=86400
            ).retrieve_data()
            
            if data.empty:
                print(f"No data returned from Coinbase Pro API for {symbol}.")
                return None
            
            # Add market cap
            data['market_cap'] = data['close'] * self.SUPPORTED_CRYPTOS[symbol]
            
            # Ensure date is the index
            if data.index.name != 'date':
                data.index.name = 'date'
            
            return data
            
        except Exception as e:
            print(f"Error with Historic-Crypto/Coinbase Pro API: {e}")
            return None
    
    def _generate_sample_data(self, symbol: str, days: int = 1825) -> pd.DataFrame:
        """
        Generate sample data for testing when APIs are unavailable.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of data to generate
            
        Returns:
            pd.DataFrame: DataFrame with sample price data
        """
        print(f"Generating synthetic data for {symbol} ({days} days)...")
        
        # Set different base parameters for different cryptocurrencies
        params = {
            'BTC': {'base_price': 5000, 'price_range': 50000, 'volatility': 0.03},
            'ETH': {'base_price': 200, 'price_range': 3000, 'volatility': 0.04},
            'XRP': {'base_price': 0.30, 'price_range': 3.0, 'volatility': 0.05},
            'ADA': {'base_price': 0.10, 'price_range': 2.0, 'volatility': 0.06},
            'SOL': {'base_price': 20, 'price_range': 200, 'volatility': 0.07},
            'DOT': {'base_price': 5, 'price_range': 40, 'volatility': 0.06}
        }
        
        crypto_params = params.get(symbol, {'base_price': 1.0, 'price_range': 10.0, 'volatility': 0.05})
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        np.random.seed(42)  # For reproducibility
        n_days = len(date_range)
        
        # Generate price movement with multiple cycles
        t = np.linspace(0, 4*np.pi, n_days)
        trend1 = 0.5 * np.sin(t)
        trend2 = 0.2 * np.sin(2.5*t)
        trend3 = 0.1 * np.sin(5*t)
        
        upward_drift = np.linspace(0, 0.7, n_days)
        combined_trend = trend1 + trend2 + trend3 + upward_drift
        
        # Normalize and scale to price range
        combined_trend = (combined_trend - combined_trend.min()) / (combined_trend.max() - combined_trend.min())
        prices = crypto_params['base_price'] + combined_trend * crypto_params['price_range']
        
        # Add daily noise
        daily_noise = np.random.normal(0, crypto_params['volatility'], n_days)
        for i in range(1, n_days):
            prices[i] = prices[i] * (1 + daily_noise[i])
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Add OHLC data
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * 0.99
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.008, len(df))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.008, len(df))))
        
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Generate volume
        volatility = np.abs(df['high'] - df['low']) / df['low']
        df['volume'] = np.random.lognormal(mean=18, sigma=1, size=len(date_range)) * 1000
        df['volume'] = df['volume'] * (1 + 5 * volatility)
        
        df['market_cap'] = df['close'] * self.SUPPORTED_CRYPTOS[symbol]
        
        return df
    
    def get_daily_ohlc(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch and cache daily OHLC data for any cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with daily OHLC data
        """
        symbol = symbol.upper()
        cache_file = os.path.join(self.cache_dir, symbol.lower(), f"{symbol.lower()}_daily_ohlc_{days}d.csv")
        
        # Check cache
        if os.path.exists(cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_mod_time < timedelta(hours=24):
                print(f"Loading cached {symbol} data from {cache_file}")
                return pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        
        # Fetch new data
        df = self.fetch_historical_data(symbol, days)
        
        if df is None or df.empty:
            warnings.warn(f"Failed to retrieve {symbol} data. Generating sample data.")
            df = self._generate_sample_data(symbol, days)
        
        # Save to cache
        df.to_csv(cache_file)
        
        return df
    
    def get_weekly_ohlc(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch and cache weekly OHLC data for any cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days (int): Number of days of historical data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with weekly OHLC data
        """
        symbol = symbol.upper()
        cache_file = os.path.join(self.cache_dir, symbol.lower(), f"{symbol.lower()}_weekly_ohlc_{days}d.csv")
        
        # Check cache
        if os.path.exists(cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_mod_time < timedelta(hours=24):
                print(f"Loading cached {symbol} weekly data from {cache_file}")
                return pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        
        # Get daily data first
        daily_df = self.get_daily_ohlc(symbol, days)
        
        if daily_df is None or daily_df.empty:
            return None
        
        # Resample to weekly OHLC
        weekly_df = daily_df.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'market_cap': 'mean'
        })
        
        # Save to cache
        weekly_df.to_csv(cache_file)
        
        return weekly_df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and ensure required columns exist."""
        df = df.copy()
        
        # Rename columns based on mapping
        df.rename(columns={k: v for k, v in self.column_mapping.items() if k in df.columns}, 
                 inplace=True)
        
        # Ensure all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # If we only have close prices, estimate others
        if 'close' in df.columns and not all(col in df.columns for col in ['open', 'high', 'low']):
            df['open'] = df['close'].shift(1)
            df['high'] = df['close']
            df['low'] = df['close']
            
        # If volume is missing, add a placeholder
        if 'volume' not in df.columns:
            df['volume'] = 0
            logger.warning("Volume data not available, using placeholder values")
        
        return df
    
    def _fetch_from_cryptocompare(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from CryptoCompare."""
        try:
            # Calculate number of days
            days = (end_date - start_date).days + 1
            
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': days,
                'toTs': int(end_date.timestamp())
            }
            
            response = requests.get(self.base_urls['cryptocompare'], params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['Response'] == 'Success':
                df = pd.DataFrame(data['Data']['Data'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return self._standardize_columns(df)
            
        except Exception as e:
            logger.error(f"Error fetching from CryptoCompare: {str(e)}")
        
        return None
    
    def _fetch_from_alternative(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from Alternative.me."""
        try:
            params = {
                'coin': symbol.lower(),
                'start': int(start_date.timestamp()),
                'end': int(end_date.timestamp())
            }
            
            response = requests.get(self.base_urls['alternative'], params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['status']['error_code'] == 0:
                df = pd.DataFrame(data['data'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return self._standardize_columns(df)
            
        except Exception as e:
            logger.error(f"Error fetching from Alternative.me: {str(e)}")
        
        return None

def fetch_crypto_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch cryptocurrency data with fallback mechanisms.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        start_date: Start date for data collection
        end_date: End date for data collection
    
    Returns:
        DataFrame with cryptocurrency data
    """
    fetcher = CryptoDataFetcher()
    
    # Try CryptoCompare first
    logger.info(f"Fetching {symbol} data from CryptoCompare...")
    df = fetcher._fetch_from_cryptocompare(symbol, start_date, end_date)
    
    if df is not None:
        logger.info("Successfully fetched data from CryptoCompare")
        return df
    
    # Fallback to Alternative.me
    logger.info("Falling back to Alternative.me...")
    df = fetcher._fetch_from_alternative(symbol, start_date, end_date)
    
    if df is not None:
        logger.info("Successfully fetched data from Alternative.me")
        return df
    
    raise ValueError("Failed to fetch data from all available sources")

# Example usage
if __name__ == "__main__":
    # Create fetcher instance
    fetcher = CryptoDataFetcher()
    
    # Test with multiple cryptocurrencies
    cryptos = ['BTC', 'ETH', 'XRP']
    days = 730  # 2 years of data
    
    for symbol in cryptos:
        print(f"\nFetching data for {symbol}...")
        
        # Get daily OHLC data
        daily_data = fetcher.get_daily_ohlc(symbol, days)
        if daily_data is not None:
            print(f"{symbol} Daily OHLC Data:")
            print(daily_data.head())
            print(f"Total days: {len(daily_data)}")
        
        # Get weekly OHLC data
        weekly_data = fetcher.get_weekly_ohlc(symbol, days)
        if weekly_data is not None:
            print(f"\n{symbol} Weekly OHLC Data:")
            print(weekly_data.head())
            print(f"Total weeks: {len(weekly_data)}")