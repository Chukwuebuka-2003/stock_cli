import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from twelvedata import TDClient
import yfinance as yf

from stock_cli.file_paths import CACHE_PATH

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, twelvedata_api_key=None, cache_path=CACHE_PATH, cache_duration=900):
        """
        Initializes the DataFetcher with Twelve Data.
        Args:
            twelvedata_api_key (str): The Twelve Data API key.
            cache_path (str): The path to the cache file.
            cache_duration (int): Cache duration in seconds. Defaults to 900 (15 minutes).
        """
        self.td_client = None
        
        # Initialize Twelve Data
        if twelvedata_api_key:
            self.td_client = TDClient(apikey=twelvedata_api_key)
            logger.info("Initialized Twelve Data client")
        
        if not self.td_client:
            # Check if key is in env as fallback
            key = os.getenv("TWELVE_DATA_API_KEY")
            if key:
                self.td_client = TDClient(apikey=key)
                logger.info("Initialized Twelve Data client from env")
            else:
                logger.warning("Twelve Data API key not provided. Some features may fail.")
        
        self.cache_path = cache_path
        self.cache_duration = cache_duration
        self.cache = self._load_cache()

    def _load_cache(self):
        """Loads the cache from a JSON file."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    logger.info(f"Loading cache from {self.cache_path}")
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                logger.warning("Could not read cache file. Starting fresh.")
                return {}
        return {}

    def _save_cache(self):
        """Saves the current cache to a JSON file."""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=4)
        except IOError:
            logger.error(f"Could not save cache to {self.cache_path}")

    def get_stock_data(self, symbol):
        """
        Get stock data for a given symbol using Twelve Data.
        """
        now = time.time()
        symbol = symbol.upper()

        # Check cache first
        if (
            symbol in self.cache
            and now - self.cache[symbol].get("timestamp", 0) < self.cache_duration
        ):
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[symbol]["data"]

        # Fetch from Twelve Data
        if self.td_client:
            try:
                logger.info(f"Fetching fresh data for {symbol} from Twelve Data.")
                quote = self.td_client.quote(symbol=symbol).as_json()
                
                if quote and 'close' in quote:
                    formatted_data = {
                        "symbol": symbol,
                        "currentPrice": float(quote['close']),
                        "previousClose": float(quote['previous_close']),
                        "change": float(quote['change']),
                        "changePercent": f"{quote['percent_change']}%",
                    }

                    self.cache[symbol] = {"data": formatted_data, "timestamp": now}
                    self._save_cache()
                    return formatted_data
                else:
                    logger.warning(f"No valid data received from Twelve Data for {symbol}: {quote}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} from Twelve Data: {e}")
        
        # Return stale cache if available
        if symbol in self.cache:
            logger.warning(f"Returning stale data for {symbol} due to fetch error.")
            return self.cache[symbol]["data"]
        
        return None


    def get_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data using Twelve Data.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            or None if error occurs
        """
        if not self.td_client:
            logger.error("Twelve Data client not initialized")
            return None
            
        try:
            logger.info(f"Fetching historical data for {symbol} from Twelve Data")
            
            # Map period to outputsize or date range
            # Twelve Data uses 'outputsize' for number of data points
            # 1 year ~ 252 trading days
            outputsize = 30 # Default
            
            if period == "1d": outputsize = 1
            elif period == "5d": outputsize = 5
            elif period == "1mo": outputsize = 22
            elif period == "3mo": outputsize = 66
            elif period == "6mo": outputsize = 132
            elif period == "1y": outputsize = 252
            elif period == "2y": outputsize = 504
            elif period == "5y": outputsize = 1260
            elif period == "10y" or period == "max": outputsize = 2520
            else: outputsize = 252
            
            # Map interval to Twelve Data format
            td_interval = interval
            if interval == "1d":
                td_interval = "1day"
            
            # Fetch time series
            ts = self.td_client.time_series(
                symbol=symbol,
                interval=td_interval,
                outputsize=outputsize
            ).as_pandas()
            
            if ts is not None and not ts.empty:
                # Rename columns to match expected format (Title Case)
                ts = ts.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Ensure index is datetime and sorted
                ts.index = pd.to_datetime(ts.index)
                ts = ts.sort_index()
                
                ts.index.name = "Date"
                ts = ts.reset_index()
                
                logger.info(f"Successfully fetched {len(ts)} rows of historical data for {symbol}")
                return ts
            else:
                logger.warning(f"No historical data found for {symbol} from Twelve Data")
                return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol} from Twelve Data: {e}")
            return None


    def get_historical_data_range(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific date range using Twelve Data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format), defaults to today
            interval: Data interval (only '1d' supported)

        Returns:
            DataFrame with OHLCV data or None if error occurs
        """
        if not self.td_client:
            return None

        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Map interval to Twelve Data format
            td_interval = interval
            if interval == "1d":
                td_interval = "1day"

            # Twelve Data supports start_date and end_date parameters
            ts = self.td_client.time_series(
                symbol=symbol,
                interval=td_interval,
                start_date=start_date,
                end_date=end_date
            ).as_pandas()
            
            if ts is not None and not ts.empty:
                # Rename columns
                ts = ts.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Ensure index is datetime and sorted
                ts.index = pd.to_datetime(ts.index)
                ts = ts.sort_index()
                
                ts.index.name = "Date"
                ts = ts.reset_index()

                logger.info(f"Successfully fetched {len(ts)} rows of historical data for {symbol}")
                return ts
            else:
                logger.warning(f"No historical data found for {symbol} in specified range")
                return None

        except Exception as e:
            logger.error(f"Error fetching historical data range for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[dict]:
        """
        Get detailed stock information including company details, financials, etc.
        Using yfinance for detailed info as Twelve Data free tier is limited for this.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with stock information or None if error occurs
        """
        try:
            logger.info(f"Fetching detailed info for {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key information
            return {
                "symbol": symbol,
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "marketCap": info.get("marketCap", 0),
                "peRatio": info.get("trailingPE", 0),
                "dividendYield": info.get("dividendYield", 0),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
                "averageVolume": info.get("averageVolume", 0),
                "beta": info.get("beta", 0),
            }

        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None
