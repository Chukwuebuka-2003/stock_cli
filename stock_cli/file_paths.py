import os
from appdirs import user_config_dir, user_data_dir

# Define the application name
APP_NAME = "StockTrackerCLI"
APP_AUTHOR = "Chukwuebuka Ezeokeke"

# Get the platform-specific config and data directories
# We've removed APP_AUTHOR to standardize paths, but we check for old paths to migrate data.
NEW_CONFIG_DIR = user_config_dir(APP_NAME)
NEW_DATA_DIR = user_data_dir(APP_NAME)
OLD_CONFIG_DIR = user_config_dir(APP_NAME, APP_AUTHOR)
OLD_DATA_DIR = user_data_dir(APP_NAME, APP_AUTHOR)

def migrate_directory(old_dir, new_dir):
    """Migrates files from old_dir to new_dir if new_dir is empty/missing."""
    if os.path.exists(old_dir) and old_dir != new_dir:
        # If new dir doesn't exist or is empty, move contents from old dir
        if not os.path.exists(new_dir) or not os.listdir(new_dir):
            import shutil
            os.makedirs(os.path.dirname(new_dir), exist_ok=True)
            if os.path.exists(new_dir):
                os.rmdir(new_dir) # Remove empty dir if it exists to allow move
            shutil.move(old_dir, new_dir)
            return True
    return False

# Perform migration if needed
migrate_directory(OLD_CONFIG_DIR, NEW_CONFIG_DIR)
migrate_directory(OLD_DATA_DIR, NEW_DATA_DIR)

# Final resolved directories
CONFIG_DIR = NEW_CONFIG_DIR
DATA_DIR = NEW_DATA_DIR

# Ensure the directories exist
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Define the full paths to your files
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
POSITIONS_PATH = os.path.join(DATA_DIR, "positions.json")
CACHE_PATH = os.path.join(DATA_DIR, "cache.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
ALERTS_PATH = os.path.join(DATA_DIR, "alerts.json")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
