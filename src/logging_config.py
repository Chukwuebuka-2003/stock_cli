import logging
import logging.handlers
import os
import sys


def setup_logging():
    """Configure logging with rotation and structured format"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("stock_tracker")
    logger.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler with rotation (10MB max, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "stock_tracker.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
