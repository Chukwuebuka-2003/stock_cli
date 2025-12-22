import json
import logging
import os
import stat

from stock_cli.file_paths import POSITIONS_PATH

logger = logging.getLogger(__name__)

# Secure file permission mode (owner read/write only)
SECURE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600


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
        """Save positions to a JSON file with secure permissions.

        The positions file is created with restrictive permissions (0o600 - owner read/write only)
        since it may contain sensitive portfolio data.
        """
        try:
            # Ensure parent directory exists
            positions_dir = os.path.dirname(self.positions_path)
            if positions_dir:
                os.makedirs(positions_dir, exist_ok=True)

            # Write to file with secure permissions
            # Use os.open with secure mode to create file with proper permissions from the start
            fd = os.open(
                self.positions_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                SECURE_FILE_MODE
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self.positions, f, indent=4)
            except Exception:
                os.close(fd)
                raise

            # Also ensure permissions are correct for existing files
            os.chmod(self.positions_path, SECURE_FILE_MODE)
            logger.info(f"Positions saved securely to {self.positions_path}")
        except IOError as e:
            logger.error(f"Error saving positions file: {e}")

    def add_position(self, symbol, quantity, purchase_price):
        """Add a new position to the portfolio.

        Args:
            symbol: Stock ticker symbol (1-10 alphanumeric characters)
            quantity: Number of shares (must be positive)
            purchase_price: Price per share (must be non-negative)

        Raises:
            ValueError: If any input validation fails
        """
        # Validate symbol format
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        # Clean and validate symbol (allow only alphanumeric, max 10 chars)
        clean_symbol = symbol.strip().upper()
        if not clean_symbol.isalnum() or len(clean_symbol) > 10:
            raise ValueError("Symbol must be 1-10 alphanumeric characters")

        # Validate quantity (must be positive number)
        try:
            quantity = float(quantity)
        except (TypeError, ValueError):
            raise ValueError("Quantity must be a valid number")

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        # Validate purchase price (must be non-negative)
        try:
            purchase_price = float(purchase_price)
        except (TypeError, ValueError):
            raise ValueError("Purchase price must be a valid number")

        if purchase_price < 0:
            raise ValueError("Purchase price cannot be negative")

        self.positions.append(
            {
                "symbol": clean_symbol,
                "quantity": quantity,
                "purchase_price": purchase_price,
            }
        )
        self.save_positions()
        logger.info(f"Added position: {clean_symbol} - {quantity} shares @ ${purchase_price}")

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

# Alias for compatibility
PortfolioManager = Portfolio
