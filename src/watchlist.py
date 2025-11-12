import json
import logging
from datetime import datetime

from stock_cli.file_paths import WATCHLIST_PATH

logger = logging.getLogger(__name__)


class Watchlist:
    def __init__(self, watchlist_path=WATCHLIST_PATH):
        """
        Initializes the Watchlist manager.
        Args:
            watchlist_path (str): The path to the watchlist file.
        """
        self.watchlist_path = watchlist_path
        self.stocks = self.load_watchlist()

    def load_watchlist(self):
        """Load watchlist from JSON file."""
        try:
            with open(self.watchlist_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("Watchlist file not found. Starting with empty watchlist.")
            return []
        except json.JSONDecodeError:
            logger.error("Error decoding watchlist.json. Starting with empty watchlist.")
            return []

    def save_watchlist(self):
        """Save watchlist to JSON file."""
        try:
            with open(self.watchlist_path, "w") as f:
                json.dump(self.stocks, f, indent=4)
            logger.info(f"Watchlist saved successfully to {self.watchlist_path}")
        except IOError as e:
            logger.error(f"Error saving watchlist file: {e}")

    def add_stock(self, symbol, note=None):
        """
        Add a stock to the watchlist.
        Args:
            symbol (str): Stock symbol
            note (str): Optional note about the stock
        Returns:
            bool: True if added, False if already exists
        """
        symbol = symbol.strip().upper()

        # Validate symbol format: must be non-empty and alphanumeric
        if not symbol or not symbol.replace('.', '').replace('-', '').isalnum():
            logger.warning(f"Invalid stock symbol: '{symbol}'")
            return False

        # Check if stock already exists (normalize for comparison)
        if any(stock["symbol"].strip().upper() == symbol for stock in self.stocks):
            logger.warning(f"{symbol} is already in the watchlist")
            return False

        stock_entry = {
            "symbol": symbol,
            "added_at": datetime.now().isoformat(),
            "note": note
        }

        self.stocks.append(stock_entry)
        self.save_watchlist()
        logger.info(f"Added {symbol} to watchlist")

        return True

    def remove_stock(self, symbol):
        """
        Remove a stock from the watchlist.
        Args:
            symbol (str): Stock symbol
        Returns:
            bool: True if removed, False if not found
        """
        symbol = symbol.strip().upper()
        original_count = len(self.stocks)
        self.stocks = [s for s in self.stocks if s["symbol"] != symbol]

        if len(self.stocks) < original_count:
            self.save_watchlist()
            logger.info(f"Removed {symbol} from watchlist")
            return True
        else:
            logger.warning(f"{symbol} not found in watchlist")
            return False

    def get_stocks(self):
        """Get all stocks in the watchlist."""
        return self.stocks

    def is_in_watchlist(self, symbol):
        """
        Check if a stock is in the watchlist.
        Args:
            symbol (str): Stock symbol
        Returns:
            bool: True if in watchlist, False otherwise
        """
        symbol = symbol.strip().upper()
        return any(stock["symbol"] == symbol for stock in self.stocks)

    def update_note(self, symbol, note):
        """
        Update the note for a stock.
        Args:
            symbol (str): Stock symbol
            note (str): New note
        Returns:
            bool: True if updated, False if not found
        """
        symbol = symbol.strip().upper()

        for stock in self.stocks:
            if stock["symbol"] == symbol:
                stock["note"] = note
                self.save_watchlist()
                logger.info(f"Updated note for {symbol}")
                return True

        logger.warning(f"{symbol} not found in watchlist")
        return False
