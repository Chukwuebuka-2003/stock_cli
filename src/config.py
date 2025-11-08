import json
import logging
import os

from .file_paths import CONFIG_PATH

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path=CONFIG_PATH):
        """
        Initializes the Config manager.
        Args:
            config_path (str): The path to the configuration file.
                               Defaults to the path from file_paths.py.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from a JSON file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found. Creating a default config.")
            return self.create_default_config()
        except json.JSONDecodeError:
            logger.error("Error decoding config.json. Starting with a default config.")
            return self.create_default_config()

    def create_default_config(self):
        """Create a default configuration."""
        default_config = {
            "email_settings": {
                "smtp_server": "",
                "smtp_port": 587,
                "email": "",
                "password": "",
                "recipient": "",
            },
            "groq_api_key": None,
            "alpha_vantage_api_key": None,
        }
        self.save_config(default_config)
        return default_config

    def save_config(self, config_data=None):
        """
        Save configuration to a JSON file.
        Args:
            config_data (dict, optional): The configuration data to save.
                                          If None, saves the current self.config.
        """
        if config_data is None:
            config_data = self.config
        try:
            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Configuration saved successfully to {self.config_path}")
        except IOError as e:
            logger.error(f"Error saving config file: {e}")

    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value and save it."""
        self.config[key] = value
        self.save_config()
