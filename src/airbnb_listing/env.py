import os

from dotenv import load_dotenv

from airbnb_listing.config import ENV_DIR
from airbnb_listing.logging import logger

# Load the .env file
try:
    load_dotenv(ENV_DIR)
except Exception as e:
    logger.error(f"Error loading .env file: {e}")


# Get the environment variables
DB_HOST = os.getenv("DB_HOST")
DB_TOKEN = os.getenv("DB_TOKEN")

if not all([DB_HOST, DB_TOKEN]):
    raise EnvironmentError("One or more Databricks environment variables are not set!")
