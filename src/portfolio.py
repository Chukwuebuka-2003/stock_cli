import json
import logging
import os

from stock_cli.file_paths import POSITIONS_PATH

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, positions_path=POSITIONS_PATH):
        """
        Initializes the Portfolio manager.
        Args:
            positions_path (str): The path to the positions file.
                                  Defaults to the path from file_paths.py.
        """
        self.positions_path = positions_path
        self.positions = self.load_positions()

    def load_positions(self):
        """
        Load positions from environment variable or JSON file.
        Priority: PORTFOLIO_POSITIONS env var -> positions.json file
        """
        # First, check if positions are provided via environment variable
        if env_positions := os.getenv("PORTFOLIO_POSITIONS"):
            try:
                positions = json.loads(env_positions)
                # Validate structure of positions
                if not isinstance(positions, list):
                    logger.error("PORTFOLIO_POSITIONS must be a JSON array")
                    logger.info("Falling back to positions file...")
                else:
                    # Validate each position has required fields
                    valid_positions = []
                    for pos in positions:
                        if not isinstance(pos, dict):
                            logger.warning(f"Skipping invalid position (not a dict): {pos}")
                            continue
                        if not all(key in pos for key in ["symbol", "quantity", "purchase_price"]):
                            logger.warning(f"Skipping position with missing fields: {pos}")
                            continue
                        valid_positions.append(pos)

                    if valid_positions:
                        logger.info(f"Loaded {len(valid_positions)} valid portfolio positions from PORTFOLIO_POSITIONS environment variable")
                        return valid_positions
                    else:
                        logger.warning("No valid positions found in PORTFOLIO_POSITIONS environment variable")
                        logger.info("Falling back to positions file...")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding PORTFOLIO_POSITIONS environment variable: {e}")
                logger.info("Falling back to positions file...")

        # Fall back to loading from file
        try:
            with open(self.positions_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                "Positions file not found. Starting with an empty portfolio."
            )
            return []
        except json.JSONDecodeError:
            logger.error(
                "Error decoding positions.json. Starting with an empty portfolio."
            )
            return []

    def save_positions(self):
        """Save positions to a JSON file."""
        try:
            with open(self.positions_path, "w") as f:
                json.dump(self.positions, f, indent=4)
            logger.info(f"Positions saved successfully to {self.positions_path}")
        except IOError as e:
            logger.error(f"Error saving positions file: {e}")

    def add_position(self, symbol, quantity, purchase_price):
        """Add a new position to the portfolio."""
        self.positions.append(
            {
                "symbol": symbol.upper(),
                "quantity": quantity,
                "purchase_price": purchase_price,
            }
        )
        self.save_positions()
        logger.info(f"Added position: {symbol} - {quantity} shares @ ${purchase_price}")

    def remove_position(self, symbol):
        """Remove a position from the portfolio."""
        original_count = len(self.positions)
        self.positions = [p for p in self.positions if p["symbol"] != symbol.upper()]
        if len(self.positions) < original_count:
            self.save_positions()
            logger.info(f"Removed position: {symbol}")
            return True
        else:
            logger.warning(f"Symbol not found: {symbol}")
            return False

    def get_positions(self):
        """Get all positions."""
        return self.positions
