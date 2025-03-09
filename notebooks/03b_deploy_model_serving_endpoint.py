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

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.logging import logger
from airbnb_listing.serving.model_serving import ModelServing

# COMMAND ----------
# NOTE: hardcoded in notebooks, get env from DAB in scripts
env = "dev"  # hardcoded

# Get the catalog name based on the environment
catalog_name = get_env_catalog(env)

# COMMAND ----------
# Load the configuration
# NOTE: Hardcoded path in notebook, get root path from DAB in scripts
root_path = "/Workspace/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/.bundle/dev/airbnb-listing/"  # hardcoded
config_path = f"{root_path}/files/project_config.yml"

# If running locally, change the root path
if not os.path.exists(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_root_path = os.path.dirname(current_dir)  # Move one level up
    config_path = f"{new_root_path}/project_config.yml"

config = get_config(config_path)

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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
silver_schema_name = config.general.SILVER_SCHEMA
endpoint_name = "airbnb-listing-model-serving"
model_name = f"{config.model.MODEL_NAME}_basic"

# COMMAND ----------

# Initialize Feature Lookup Serving Manager
feature_model_server = ModelServing(
    model_name=f"{catalog_name}.{model_asset_schema_name}.{model_name}",
    endpoint_name=endpoint_name,
    config=config,
)


# COMMAND ----------

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()

# COMMAND ----------
# Test the serving endpoint

# Create a sample request body
train_set = spark.table(
    f"{catalog_name}.{silver_schema_name}.silver_airbnb_listing_price_train"
).drop("update_timestamp_utc")

train_set = train_set.toPandas()

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
