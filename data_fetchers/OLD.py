import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import warnings
import requests

class XRPDataFetcher:
    """
    Class to fetch historical XRP price data.
    Supports multiple data sources with fallbacks.
    """
    
    def __init__(self, cache_dir='data', use_sample_data=False):
        self.cache_dir = cache_dir
        self.use_sample_data = use_sample_data
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def fetch_historical_data(self, days=365):
        """
        Fetch historical XRP price data with multiple fallback options.
        
        Args:
            days (int): Number of days of historical data to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with historical price data
        """
        # If sample data is explicitly requested, skip API calls
        if self.use_sample_data:
            print("Using sample data as requested.")
            return self._generate_sample_data(days)
        
        # Try different data sources in order
        try:
            print(f"Fetching XRP historical data for the last {days} days...")
            
            # Try CryptoCompare API first
            df = self._fetch_from_cryptocompare(days)
            if df is not None and not df.empty:
                print("Successfully fetched data from CryptoCompare.")
                return df
            
            # Try Alphavantage API next
            df = self._fetch_from_alphavantage(days)
            if df is not None and not df.empty:
                print("Successfully fetched data from Alpha Vantage.")
                return df
            
            # Try Historic-Crypto as a fallback
            df = self._fetch_from_historic_crypto(days)
            if df is not None and not df.empty:
                print("Successfully fetched data from Historic-Crypto.")
                return df
            
            # If all APIs fail, use sample data
            print("All API attempts failed, generating sample data instead.")
            return self._generate_sample_data(days)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Generating sample data instead.")
            return self._generate_sample_data(days)
    
    def _fetch_from_cryptocompare(self, days):
        """
        Fetch data from CryptoCompare API, handling requests for long time periods
        by making multiple API calls if necessary.
        
        Args:
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame or None: DataFrame with historical data or None if failed
        """
        try:
            print(f"Trying CryptoCompare API for {days} days of data...")
            
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
                url = f"https://min-api.cryptocompare.com/data/v2/histoday"
                params = {
                    "fsym": "XRP",
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
                    # Set toTs to the timestamp of the oldest day we've received
                    current_timestamp = crypto_data[0]['time']
                    # Wait a bit to avoid rate limiting
                    time.sleep(0.5)
            
            if not all_data:
                print("Failed to retrieve any data from CryptoCompare API.")
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
            
            # Add market cap (approximated)
            circulating_supply = 46_000_000_000  # 46 billion XRP
            df['market_cap'] = df['close'] * circulating_supply
            
            # Select only the columns we need
            columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
            df = df[columns]
            
            # Sort chronologically
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"Error with CryptoCompare API: {e}")
            return None
    
    def _fetch_from_alphavantage(self, days):
        """
        Fetch data from Alpha Vantage API.
        
        Args:
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame or None: DataFrame with historical data or None if failed
        """
        try:
            print("Trying Alpha Vantage API...")
            # Alpha Vantage free tier doesn't require an API key for limited use,
            # but it's better to get one for consistent access
            api_key = "demo"  # Replace with your API key if you have one
            
            # Construct the API URL
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": "XRP",
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
            
            # Create a list to store daily data
            daily_data = []
            
            # Process each day
            for date, values in time_series.items():
                daily_data.append({
                    'date': date,
                    'open': float(values['1a. open (USD)']),
                    'high': float(values['2a. high (USD)']),
                    'low': float(values['3a. low (USD)']),
                    'close': float(values['4a. close (USD)']),
                    'volume': float(values['5. volume']),
                    'market_cap': float(values['4a. close (USD)']) * 46_000_000_000  # Estimated
                })
            
            # Create DataFrame
            df = pd.DataFrame(daily_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Sort chronologically and limit to requested days
            df = df.sort_index()
            if len(df) > days:
                df = df.iloc[-days:]
                
            return df
            
        except Exception as e:
            print(f"Error with Alpha Vantage API: {e}")
            return None
            
    def _fetch_from_historic_crypto(self, days):
        """
        Fetch data using Historic-Crypto library.
        
        Args:
            days (int): Number of days of historical data
            
        Returns:
            pandas.DataFrame or None: DataFrame with historical data or None if failed
        """
        try:
            print("Trying Historic-Crypto API (Coinbase Pro)...")
            
            # Try importing the library
            try:
                from Historic_Crypto import HistoricalData
            except ImportError:
                print("Historic-Crypto library not found. Install with: pip install Historic-Crypto")
                return None
            
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Historic-Crypto
            start_date_str = start_date.strftime('%Y-%m-%d-%H-%M')
            end_date_str = end_date.strftime('%Y-%m-%d-%H-%M')
            
            # Fetch data using Historic-Crypto
            # Note: granularity options are 60, 300, 900, 3600, 21600, 86400 seconds
            # 86400 = daily data
            data = HistoricalData(
                ticker='XRP-USD',
                start_date=start_date_str,
                end_date=end_date_str,
                granularity=86400
            ).retrieve_data()
            
            if data.empty:
                print("No data returned from Coinbase Pro API.")
                return None
            
            # Add market cap (estimated using circulating supply)
            circulating_supply = 46_000_000_000
            data['market_cap'] = data['close'] * circulating_supply
            
            # Make sure date is the index
            if data.index.name != 'date':
                data.index.name = 'date'
            
            return data
            
        except Exception as e:
            print(f"Error with Historic-Crypto/Coinbase Pro API: {e}")
            return None
    
    def _generate_sample_data(self, days=1825):  # Default to 5 years
        """
        Generate sample data for testing when APIs are unavailable.
        
        Args:
            days (int): Number of days of data to generate
            
        Returns:
            pandas.DataFrame: DataFrame with sample price data
        """
        print(f"Generating synthetic data for {days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random price data with more realistic long-term trends for 5 years
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price 
        base_price = 0.30  # Start lower for 5 years ago
        
        # Create several trend periods to simulate bull and bear markets
        n_days = len(date_range)
        
        # Generate more complex price movement with multiple cycles
        # Create sine waves with different frequencies and amplitudes
        t = np.linspace(0, 4*np.pi, n_days)  # 2 full cycles over 5 years
        trend1 = 0.5 * np.sin(t) # Main cycle
        trend2 = 0.2 * np.sin(2.5*t) # Secondary cycle
        trend3 = 0.1 * np.sin(5*t) # Faster cycle
        
        # Combine trends with a general upward drift
        upward_drift = np.linspace(0, 0.7, n_days)  # General upward drift over time
        combined_trend = trend1 + trend2 + trend3 + upward_drift
        
        # Normalize to 0-1 range and scale to price range
        combined_trend = (combined_trend - combined_trend.min()) / (combined_trend.max() - combined_trend.min())
        price_range = 3.0  # Max price around $3
        prices = base_price + combined_trend * price_range
        
        # Add daily noise
        daily_noise = np.random.normal(0, 0.02, n_days)
        for i in range(1, n_days):
            prices[i] = prices[i] * (1 + daily_noise[i])
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Add OHLC data
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, len(df)))
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * 0.99  # Set first open price
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.008, len(df))))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.008, len(df))))
        
        # Ensure high is always >= open and close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is always <= open and close
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Generate volume with correlation to price volatility
        volatility = np.abs(df['high'] - df['low']) / df['low']
        df['volume'] = np.random.lognormal(mean=18, sigma=1, size=len(date_range)) * 1000
        # Increase volume during more volatile periods
        df['volume'] = df['volume'] * (1 + 5 * volatility)
        
        df['market_cap'] = df['close'] * 46000000000  # 46 billion XRP supply
        
        return df
    
    def get_daily_ohlc(self, days=365, vs_currency='usd'):
        """
        Fetch and convert historical data to daily OHLC format.
        
        Args:
            days (int): Number of days of historical data to fetch
            vs_currency (str): Currency to compare against (default: 'usd')
            
        Returns:
            pandas.DataFrame: DataFrame with daily OHLC data
        """
        # Check if we have cached data
        cache_file = os.path.join(self.cache_dir, f"xrp_daily_ohlc_{days}d.csv")
        
        # Check if cache file exists and is recent (less than 24 hours old)
        if os.path.exists(cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_mod_time < timedelta(hours=24):
                print(f"Loading cached data from {cache_file}")
                return pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        
        # Fetch raw historical data
        df = self.fetch_historical_data(days)
        
        if df is None or df.empty:
            warnings.warn("Failed to retrieve data. Generating sample data.")
            df = self._generate_sample_data(days)
        
        # Save to cache
        df.to_csv(cache_file)
        
        return df
    
    def get_weekly_ohlc(self, days=365, vs_currency='usd'):
        """
        Fetch and convert historical data to weekly OHLC format.
        
        Args:
            days (int): Number of days of historical data to fetch
            vs_currency (str): Currency to compare against (default: 'usd')
            
        Returns:
            pandas.DataFrame: DataFrame with weekly OHLC data
        """
        # Check if we have cached data
        cache_file = os.path.join(self.cache_dir, f"xrp_weekly_ohlc_{days}d.csv")
        
        # Check if cache file exists and is recent (less than 24 hours old)
        if os.path.exists(cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_mod_time < timedelta(hours=24):
                print(f"Loading cached data from {cache_file}")
                return pd.read_csv(cache_file, index_col='date', parse_dates=['date'])
        
        # Get daily data first
        daily_df = self.get_daily_ohlc(days, vs_currency)
        
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

# Example usage
if __name__ == "__main__":
    # Create fetcher with automatic fallback to sample data
    fetcher = XRPDataFetcher()
    
    # Get daily OHLC data for the past 2 years
    daily_data = fetcher.get_daily_ohlc(days=730)
    if daily_data is not None:
        print("Daily OHLC Data:")
        print(daily_data.head())
        print(f"Total days: {len(daily_data)}")
    
    # Get weekly OHLC data for the past 2 years
    weekly_data = fetcher.get_weekly_ohlc(days=730)
    if weekly_data is not None:
        print("\nWeekly OHLC Data:")
        print(weekly_data.head())
        print(f"Total weeks: {len(weekly_data)}")