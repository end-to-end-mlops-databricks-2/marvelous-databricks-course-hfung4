# Databricks notebook source
# MAGIC %pip install ../../artifacts/.internal/airbnb_listing-0.0.13-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
import datetime
import itertools
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from databricks.connect import DatabricksSession
from airbnb_listing.config import get_config
from airbnb_listing.data_manager import (
    get_env_catalog,
    get_env_pipeline_id,
)
from pyspark.dbutils import DBUtils
from airbnb_listing.logging import logger
import time
from databricks.sdk import WorkspaceClient
from typing import Dict, List

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

# Endpoint name
endpoint_name = f"airbnb-listing-model-serving-fe-{env}"

# COMMAND ----------

# NOTE: If running as a task in a Databricks workflow, then I can get the Databricks Host and Token from DAB
# which in turns gets it from e.g., the Databricks secret scope or Github secrets from CI/CD

dbutils = DBUtils(spark)
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Get test set
test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test"
).toPandas()

# Get the skewed inference data
inference_data_skewed = (
    spark.table(f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}.inference_data_skewed").toPandas())

# COMMAND ----------

# Drop columns that will be retrieved from the feature table, and the update_timestamp_utc
columns_to_drop = ["latitude","longitude","is_manhattan", "update_timestamp_utc", config.model.TARGET]
test_set = test_set.drop(columns=columns_to_drop)
inference_data_skewed = inference_data_skewed.drop(columns=columns_to_drop)

# COMMAND ----------

# Need to change to nullable float64 to be able to have NA rather than NaN
# JSON does not support NaN
for col in ["minimum_nights",
            "estimated_listed_months",
            "availability_365",
            "number_of_reviews",
            "calculated_host_listings_count"]:
    test_set[col] = test_set[col].astype('Float64')
    inference_data_skewed[col] = inference_data_skewed[col].astype('Float64')

# COMMAND ----------

inference_data_skewed.display()

# COMMAND ----------

test_set.display()

# COMMAND ----------

# Sample records from skewed inference datasets
sampled_skewed_records = inference_data_skewed.to_dict(orient="records")
skewed_dataframe_records = [[record] for record in sampled_skewed_records]
skewed_dataframe_records

# COMMAND ----------

# Sample records from test dataset
sampled_test_records = test_set.to_dict(orient="records")
test_dataframe_records = [[record] for record in sampled_test_records]
test_dataframe_records

# COMMAND ----------

workspace = WorkspaceClient()

# COMMAND ----------

# Two different way to send request to the endpoint

# 1. Using https endpoint (we used this method previously in 03c_deploy_fe_model_serving_endpoint.py)
def send_request_https(dataframe_record: List[Dict], endpoint_name:str):
    # Define the endpoint URL
    serving_endpoint = (
        f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
    )
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": dataframe_record},
    )
    return response


# 2. Using workspace client
def send_request_workspace(dataframe_record: List[Dict], endpoint_name:str):
    response = workspace.serving_endpoints.query(
        name=endpoint_name, dataframe_records=dataframe_record
    )
    return response

# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(test_dataframe_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record, endpoint_name)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)


# COMMAND ----------

# Loop over test records and send requests for 10 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(skewed_dataframe_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record, endpoint_name)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)