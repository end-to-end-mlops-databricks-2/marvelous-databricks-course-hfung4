import argparse
import os

from databricks.connect import DatabricksSession
from sklearn.model_selection import train_test_split

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.data_processor import DataProcessor, generate_synthetic_data
from airbnb_listing.logging import logger

spark = DatabricksSession.builder.getOrCreate()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env",
    action="store",
    default="dev",
    type=str,
    required=False,
)

parser.add_argument(
    "--root_path",
    action="store",
    default="/Workspace/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/.bundle",
    type=str,
    required=False,
)

args = parser.parse_args()


# Get configuration

# NOTE: root path is: /Workspace/Users/<user email>/.bundle/
config_path = f"{args.root_path}/{args.env}/airbnb_listing/files/project_config.yml"

# If running locally, change the root path
if not os.path.exists(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_root_path = os.path.dirname(current_dir)  # Move one level up
    config_path = f"{new_root_path}/project_config.yml"

config = get_config(config_path)

catalog_name = get_env_catalog(env=args.env, config=config)

# Import bronze data (always from the production catalog)
bronze_table_name = f"{config.general.PROD_CATALOG}.{config.general.BRONZE_SCHEMA}.airbnb_listing_price"
bronze = spark.table(bronze_table_name).toPandas()

if config.general.GENERATE_AND_APPEND_SYN_DATA:
    # Generate synthetic data
    # This is mimicking a new data arrival. In real world, this would be a new batch of data.
    # bronze is passed to infer schema
    synthetic_df = generate_synthetic_data(bronze, num_rows=100)
    logger.info("Synthetic data generated.")
    data_processor = DataProcessor(synthetic_df)
else:
    data_processor = DataProcessor(bronze)

silver = data_processor.preprocess()

# Split the dataset into training and test sets (e.g., 80% training, 20% testing)
train_df, test_df = train_test_split(silver, test_size=config.model.TEST_SIZE, random_state=config.general.RANDOM_STATE)
logger.info(f"Training set shape: {train_df.shape}")
logger.info(f"Test set shape: {test_df.shape}")

# Load data to silver table
train_silver_table_name = f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_train"
test_silver_table_name = f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test"

data_processor.write_processed_data(train_df, table_name=train_silver_table_name)
logger.info(f"Training data written to {train_silver_table_name} in Unity Catalog.")

data_processor.write_processed_data(test_df, table_name=test_silver_table_name)
logger.info(f"Training data written to {test_silver_table_name} in Unity Catalog.")
