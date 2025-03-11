# Databricks notebook source
%pip install ../../artifacts/.internal/airbnb_listing-0.0.12-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import time
from typing import Dict, List
import requests
from databricks.connect import DatabricksSession
import mlflow
from pyspark.dbutils import DBUtils

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import (
    get_env_catalog,
    get_env_pipeline_id,
    table_exists,
)
from airbnb_listing.logging import logger
from airbnb_listing.serving.fe_model_serving import FeatureLookupServing


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
catalog_name = get_env_catalog(env)
pipeline_id = get_env_pipeline_id(env)


# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# NOTE: If running as a task in a Databricks workflow, then I can get the Databricks Host and Token from DAB
# which in turns gets it from e.g., the Databricks secret scope or Github secrets from CI/CD

dbutils = DBUtils(spark)
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Define catalog, schema, and feature table, feature spec, and endpoint names
model_asset_schema_name = config.general.ML_ASSET_SCHEMA
silver_schema_name = config.general.SILVER_SCHEMA
gold_schema_name = config.general.GOLD_SCHEMA
endpoint_name = f"airbnb-listing-model-serving-fe-{env}"
model_name = f"{config.model.MODEL_NAME}_fe"
feature_table_name = config.general.FEATURE_TABLE_NAME

# COMMAND ----------

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{model_asset_schema_name}.{model_name}",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{gold_schema_name}.{feature_table_name}",
    config=config,
    env = env
)

# COMMAND ----------

# Create the online table
if not table_exists(
    catalog=catalog_name,
    schema=gold_schema_name,
    table=f"{feature_table_name}_online",
):
    feature_model_server.create_online_table()
    logger.info("✅ Online Feature Table created")
else:
    feature_model_server.update_online_table(pipeline_id=pipeline_id)
    logger.info("✅ Online Feature Table updated")

# COMMAND ----------

feature_model_server.model_name

# COMMAND ----------

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the serving endpoint

# COMMAND ----------

# Create a sample request body
train_set = spark.table(f"{catalog_name}.{silver_schema_name}.airbnb_listing_price_train").drop(
            "latitude", "longitude", "is_manhattan","update_timestamp_utc"
        )

train_set = train_set.toPandas()

# Need to change to nullable float64 to be able to have NA rather than NaN
# JSON does not support NaN
for col in ["minimum_nights",
            "estimated_listed_months",
            "availability_365",
            "number_of_reviews",
            "calculated_host_listings_count"]:
    train_set[col] = train_set[col].astype('Float64')

train_set.head()

# COMMAND ----------

sampled_records = train_set.sample(n=1000, replace=True).to_dict(orient="records")

# COMMAND ----------

dataframe_records = [[record] for record in sampled_records]
dataframe_records

# COMMAND ----------

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])

# COMMAND ----------

# Call the endpoint with one sample record
def call_endpoint(record: List[Dict]):
    """
    Calls the model serving endpoint with a given input record.
    """
    # Define the endpoint URL
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# COMMAND ----------

# Get response from the endpoint, using the first dataframe record as the input
status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Look at model log (inference table) of the endpoint

inference_table_df= spark.table("dev.airbnb_listing_ml_assets.airbnb_listing_price_model_payload")
inference_table_df.display()