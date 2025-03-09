# Databricks notebook source
%pip install ../../artifacts/.internal/airbnb_listing-0.0.12-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.connect import DatabricksSession

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.serving.feature_serving import FeatureServing
from airbnb_listing.logging import logger

# COMMAND ----------
# Load the configuration
# NOTE: Hardcoded path in notebook, get root path from DAB in scripts
root_path = "/Workspace/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/.bundle/dev/airbnb-listing/" # hardcoded
config_path = f"{root_path}/files/project_config.yml"

# If running locally, change the root path
if not os.path.exists(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_root_path = os.path.dirname(current_dir)  # Move one level up
    config_path = f"{new_root_path}/project_config.yml"

config = get_config(config_path)

# COMMAND ----------
# NOTE: hardcoded in notebooks, get env from DAB in scripts
env = "dev"  # hardcoded

# Get the catalog name based on the environment
catalog_name = get_env_catalog(env)

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
# Accessing secrets: 

# If running as a task in a Databricks workflow, then I can get the Databricks Host and Token from DAB
# which in turns gets it from e.g., the Databricks secret scope or Github secrets from CI/CD

# If running outside of a Databricks Workflow, 
# get the environment variables from dbutils (if running from Databricks workspace), 
# or the .env file if running locally via Databricks Connect

if config.general.RUN_ON_DATABRICKS_WORKSPACE:
    from pyspark.dbutils import DBUtils

    dbutils = DBUtils(spark)
    os.environ["DBR_TOKEN"] = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
    os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
else:
    # If running locally, get the environment variables from the .env file
    from airbnb_listing.env import DB_HOST, DB_TOKEN

    os.environ["DBR_HOST"] = DB_HOST
    os.environ["DBR_TOKEN"] = DB_TOKEN

# COMMAND ----------

# Define catalog, schema, and feature table, feature spec, and endpoint names
model_asset_schema_name = config.general.ML_ASSET_SCHEMA
feature_table_name = f"{catalog_name}.{model_asset_schema_name}.airbnb_listing_preds"
feature_spec_name = f"{catalog_name}.{model_asset_schema_name}.return_predictions"
endpoint_name = "airbnb-listing-feature-serving"

silver_schema_name = config.general.SILVER_SCHEMA

# COMMAND ----------

# Get the train and test set, and then combine them into a single dataframe
train_set = spark.table(
    f"{catalog_name}.{silver_schema_name}.silver_airbnb_listing_price_train"
).toPandas()
test_set = spark.table(
    f"{catalog_name}.{silver_schema_name}.silver_airbnb_listing_price_test"
).toPandas()
full_df = pd.concat([train_set, test_set])
full_df.head()

# COMMAND ----------

model_name = config.model.MODEL_NAME

# COMMAND ----------

# Load the latest basic model
model = mlflow.sklearn.load_model(
    f"models:/{catalog_name}.{model_asset_schema_name}.{model_name}_basic@latest-model"
)

# COMMAND ----------

# Create inference (pred) table
preds_df = full_df[[config.model.ID_COLUMN, "latitude", "longitude"]].copy()
# Add predicted_listing_price to the preds_df by performing inference with full_df and the trained model
preds_df["predicted_listing_price"] = model.predict(
    full_df[
        config.model.SELECTED_NUMERIC_FEATURES
        + config.model.SELECTED_CATEGORICAL_FEATURES
    ]
)
# Convert to a spark dataframe
preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# Create a feature table from preds_df (the inference table in spark)
fe.create_table(
    name=feature_table_name,
    primary_keys=[config.model.ID_COLUMN],
    df=preds_df,
    description="Airbnb listing prices predictions feature table",
)

# COMMAND ----------

# In order for the predictions (offline) predictions table to be served, I need to create a
# read-copy low-latency copy of it (the online table). If I don't want to copy the full offline
# table to the online table at each trigger, I need to enable ChangeDataFeed for the offline feature table
spark.sql(
    f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
)


# COMMAND ----------

# Initialize feature store manager
feature_serving = FeatureServing(
    feature_table_name=feature_table_name,
    feature_spec_name=feature_spec_name,
    endpoint_name=endpoint_name,
)

# COMMAND ----------

# Create online table
feature_serving.create_online_table()

# COMMAND ----------

# Create feature spec
try:
    feature_serving.create_feature_spec()
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        logger.info("feature spec already exists")
    else:
        raise

# COMMAND ----------

# Deploy feature serving endpoint
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Feature Serving Endpoint

# COMMAND ----------

start_time = time.time()

# Serving endpoint URL
serving_endpoint = (
    f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
)

# Send request to get response, ise datafraome-records payload structure
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": [{"id": "6261696"}, {"id": "6619589"}]},
)

end_time = time.time()
execution_time = end_time - start_time


print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# Post data in the dataframe split payload structure
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_split": {"columns": ["id"], "data": [["16936036"], ["31562978"]]}},
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
