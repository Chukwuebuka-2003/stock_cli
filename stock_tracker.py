from src.cli import cli
from src.logging_config import setup_logging

# Initialize logger
setup_logging()

if __name__ == "__main__":
    cli()
