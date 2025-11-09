import os
from appdirs import user_config_dir, user_data_dir

# Define the application name
APP_NAME = "StockTrackerCLI"
APP_AUTHOR = "Chukwuebuka Ezeokeke"

# Get the platform-specific config directory
# Linux: ~/.config/StockTrackerCLI
# macOS: ~/Library/Application Support/StockTrackerCLI
# Windows: C:\\Users\\<user>\\AppData\\Local\\YourAppName\\StockTrackerCLI
CONFIG_DIR = user_config_dir(APP_NAME, APP_AUTHOR)
DATA_DIR = user_data_dir(APP_NAME, APP_AUTHOR)

# Ensure the directories exist
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Define the full paths to your files
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
POSITIONS_PATH = os.path.join(DATA_DIR, "positions.json")
CACHE_PATH = os.path.join(DATA_DIR, "cache.json")
