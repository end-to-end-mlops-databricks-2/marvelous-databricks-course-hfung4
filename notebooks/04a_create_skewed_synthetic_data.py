# Databricks notebook source

# COMMAND ----------
import os
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from databricks.connect import DatabricksSession
from airbnb_listing.config import get_config
from airbnb_listing.data_manager import (
    get_env_catalog,
    get_env_pipeline_id,
)
from airbnb_listing.data_processor import DataProcessor, generate_synthetic_data
from airbnb_listing.logging import logger
import time
from databricks.sdk import WorkspaceClient

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
# NOTE: hardcoded in notebooks, get env from DAB in scripts
env = "dev"  # hardcoded

# COMMAND ----------
# Load the configuration

# NOTE: Hardcoded path in notebook, get root path from DAB in scripts
root_path = "/Workspace/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/.bundle"
env_root_path = f"{root_path}/{env}/marvelous-databricks-course-hfung4/files"
config_path = f"{env_root_path}/project_config.yml"

# If running locally, change the root path
if not os.path.exists(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_root_path = os.path.dirname(current_dir)  # Move one level up
    config_path = f"{new_root_path}/project_config.yml"

config = get_config(config_path)

# Get the catalog name and pipeline_id based on the environment
catalog_name = get_env_catalog(env, config)
pipeline_id = get_env_pipeline_id(env, config)


# COMMAND ----------
# Get test set
test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test"
).toPandas()

# COMMAND ----------
# Create skewed inference data

bronze_table_name = (
    f"{config.general.PROD_CATALOG}.{config.general.BRONZE_SCHEMA}.airbnb_listing_price"
)
bronze = spark.table(bronze_table_name).toPandas()

skewed_synthetic_df = generate_synthetic_data(
    df=bronze, config=config, num_rows=200, drift=True
)
logger.info("Skewed synthetic data generated.")
skewed_data_processor = DataProcessor(skewed_synthetic_df, config)

# COMMAND ----------
inference_data_skewed = skewed_data_processor.preprocess()

# COMMAND ----------
inference_data_skewed.info()
# COMMAND ----------
inference_data_skewed.is_manhattan.value_counts(normalize=True)

# COMMAND ----------
inference_data_skewed.room_type.value_counts(normalize=True)
# COMMAND ----------
# Add update_timestamp_utc column to the skewed inference data
# and convert it back to spark DataFrame
inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)
# COMMAND ----------
# Persist the table in Unity Catalog
inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}.inference_data_skewed"
)

# COMMAND ----------
config.general.FEATURE_TABLE_NAME

# COMMAND ----------
# Since one of the columns (is_manhattan) in the skewed inference data
# is a feature in the feature table, I will need to insert
# rows from the skewed inference data into the feature table.
# I then need to update the online feature table.

workspace = WorkspaceClient()

# Write is_manhattan the skewed inference data to the feature table
spark.sql(
    f"""
    INSERT INTO {catalog_name}.{config.general.GOLD_SCHEMA}.{config.general.FEATURE_TABLE_NAME}
    SELECT id, latitude, longitude, is_manhattan
    FROM {catalog_name}.{config.general.ML_ASSET_SCHEMA}.inference_data_skewed
    """
)

# COMMAND ----------
# Update the online table

update_response = workspace.pipelines.start_update(
    pipeline_id=pipeline_id, full_refresh=False
)
while True:
    update_info = workspace.pipelines.get_update(
        pipeline_id=pipeline_id, update_id=update_response.update_id
    )
    state = update_info.update.state.value
    if state == "COMPLETED":
        break
    elif state in ["FAILED", "CANCELED"]:
        raise SystemError("Online table failed to update.")
    elif state == "WAITING_FOR_RESOURCES":
        print("Pipeline is waiting for resources.")
    else:
        print(f"Pipeline is in {state} state.")
    time.sleep(30)
