# Databricks notebook source

# COMMAND ----------
import os
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from databricks.connect import DatabricksSession
from airbnb_listing.config import get_config
from airbnb_listing.data_manager import (
    get_env_catalog,
    get_env_pipeline_id,
    table_exists,
)

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


# COMMAND ----------
# Get train and test set
train_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA_NAME}.airbnb_listing_price_train"
).toPandas()
test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA_NAME}.airbnb_listing_price_test"
).toPandas()

# COMMAND ----------
