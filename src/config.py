import json
import logging
import os

from stock_cli.file_paths import CONFIG_PATH

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
        """
        Load configuration from environment variables or JSON file.
        Priority: Environment variables -> config.json file
        """
        config = {}

        # First, try to load from file
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                logger.info("Loaded configuration from file")
        except FileNotFoundError:
            logger.warning("Config file not found. Will use environment variables or defaults.")
            config = self.create_default_config()
        except json.JSONDecodeError:
            logger.error("Error decoding config.json. Will use environment variables or defaults.")
            config = self.create_default_config()

        # Override with environment variables if present
        if os.getenv("GROQ_API_KEY"):
            config["groq_api_key"] = os.getenv("GROQ_API_KEY")
            logger.info("Using GROQ_API_KEY from environment variable")

        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            config["alpha_vantage_api_key"] = os.getenv("ALPHA_VANTAGE_API_KEY")
            logger.info("Using ALPHA_VANTAGE_API_KEY from environment variable")

        if os.getenv("TAVILY_API_KEY"):
            config["tavily_api_key"] = os.getenv("TAVILY_API_KEY")
            logger.info("Using TAVILY_API_KEY from environment variable")

        # Override email settings with environment variables if present
        email_settings = config.get("email_settings", {})
        if os.getenv("EMAIL_SMTP_SERVER"):
            email_settings["smtp_server"] = os.getenv("EMAIL_SMTP_SERVER")
        if os.getenv("EMAIL_SMTP_PORT"):
            try:
                email_settings["smtp_port"] = int(os.getenv("EMAIL_SMTP_PORT"))
            except ValueError:
                logger.error(f"Invalid EMAIL_SMTP_PORT value: {os.getenv('EMAIL_SMTP_PORT')}. Using default 587.")
                email_settings["smtp_port"] = 587
        if os.getenv("EMAIL_ADDRESS"):
            email_settings["email"] = os.getenv("EMAIL_ADDRESS")
        if os.getenv("EMAIL_PASSWORD"):
            email_settings["password"] = os.getenv("EMAIL_PASSWORD")
        if os.getenv("EMAIL_RECIPIENT"):
            email_settings["recipient"] = os.getenv("EMAIL_RECIPIENT")

        if any(email_settings.values()):
            config["email_settings"] = email_settings
            logger.info("Using email settings from environment variables")

        return config

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
