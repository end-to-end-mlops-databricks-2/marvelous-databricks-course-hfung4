from databricks.connect import DatabricksSession

from airbnb_listing.config import config
from airbnb_listing.data_processor import DataProcessor

spark = DatabricksSession.builder.getOrCreate()

# Import bronze data
bronze_table_name = f"{config.general.PROD_CATALOG}.{config.general.BRONZE_SCHEMA}.airbnb_listing_price"
bronze = spark.table(bronze_table_name).toPandas()

# Process data
data_processor = DataProcessor(bronze)
silver = data_processor.preprocess()

# Load data to silver table
silver_table_name = f"{config.general.DEV_CATALOG}.{config.general.SILVER_SCHEMA}.airbnb_listing_price"
data_processor.write_processed_data(silver, table_name=silver_table_name)
