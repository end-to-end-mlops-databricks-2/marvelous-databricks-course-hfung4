from databricks.connect import DatabricksSession
from sklearn.model_selection import train_test_split

from airbnb_listing.config import config
from airbnb_listing.data_processor import DataProcessor, generate_synthetic_data
from airbnb_listing.logging import logger

spark = DatabricksSession.builder.getOrCreate()

# Import bronze data
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
train_silver_table_name = f"{config.general.DEV_CATALOG}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_train"
test_silver_table_name = f"{config.general.DEV_CATALOG}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test"

data_processor.write_processed_data(train_df, table_name=train_silver_table_name)
data_processor.write_processed_data(test_df, table_name=test_silver_table_name)
