import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

from stock_cli.file_paths import CACHE_PATH

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, api_key, cache_path=CACHE_PATH, cache_duration=900):
        """
        Initializes the DataFetcher.
        Args:
            api_key (str): The Alpha Vantage API key.
            cache_path (str): The path to the cache file. Defaults to path from file_paths.py.
            cache_duration (int): Cache duration in seconds. Defaults to 900 (15 minutes).
        """
        if not api_key:
            raise ValueError("API key for Alpha Vantage is required.")
        self.ts = TimeSeries(key=api_key, output_format="json")
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
        Get stock data for a given symbol using Alpha Vantage, with persistent caching.
        """
        now = time.time()
        symbol = symbol.upper()

        if (
            symbol in self.cache
            and now - self.cache[symbol].get("timestamp", 0) < self.cache_duration
        ):
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[symbol]["data"]

        try:
            logger.info(f"Fetching fresh data for {symbol} from Alpha Vantage.")
            data, _ = self.ts.get_quote_endpoint(symbol=symbol)

            if not data or "01. symbol" not in data:
                logger.warning(
                    f"No valid data received from Alpha Vantage for {symbol}."
                )
                return None

            formatted_data = {
                "symbol": data.get("01. symbol"),
                "currentPrice": float(data.get("05. price")),
                "previousClose": float(data.get("08. previous close")),
                "change": float(data.get("09. change")),
                "changePercent": data.get("10. change percent"),
            }

            self.cache[symbol] = {"data": formatted_data, "timestamp": now}
            self._save_cache()
            return formatted_data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Alpha Vantage: {e}")
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
        Get historical OHLCV data for a stock using yfinance.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            or None if error occurs
        """
        try:
            logger.info(f"Fetching historical data for {symbol} (period={period}, interval={interval})")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None

            # Ensure standard column names
            df.index.name = "Date"
            df = df.reset_index()

            logger.info(f"Successfully fetched {len(df)} rows of historical data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def get_historical_data_range(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific date range.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format), defaults to today
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data or None if error occurs
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No historical data found for {symbol} in specified range")
                return None

            df.index.name = "Date"
            df = df.reset_index()

            logger.info(f"Successfully fetched {len(df)} rows of historical data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data range for {symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> Optional[dict]:
        """
        Get detailed stock information including company details, financials, etc.

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
