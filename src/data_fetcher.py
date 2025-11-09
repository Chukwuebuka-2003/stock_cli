import json
import logging
import os
import time

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
