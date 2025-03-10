import argparse

import mlflow
from databricks.connect import DatabricksSession
from pyspark.dbutils import DBUtils

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import (
    get_env_catalog,
    get_env_pipeline_id,
    table_exists,
)
from airbnb_listing.logging import logger
from airbnb_listing.serving.fe_model_serving import FeatureLookupServing

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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
spark = DatabricksSession.builder.getOrCreate()

# Get configuration
args = parser.parse_args()

# NOTE: root path is: /Workspace/Users/<user email>/.bundle
config_path = f"{args.root_path}/{args.env}/airbnb_listing/files/project_config.yml"
config = get_config(config_path)
catalog_name = get_env_catalog(env=args.env, config=config)
pipeline_id = get_env_pipeline_id(env=args.env, config=config)
dbutils = DBUtils(spark)

# Get model version from the task with the task key "train_model"
# This task points to train_register_fe_model.py
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Define catalog, schema, and feature table, feature spec, and endpoint names
model_asset_schema_name = config.general.ML_ASSET_SCHEMA
silver_schema_name = config.general.SILVER_SCHEMA
gold_schema_name = config.general.GOLD_SCHEMA
endpoint_name = f"airbnb-listing-model-serving-fe-{args.env}"
model_name = f"{config.model.MODEL_NAME}_fe"
feature_table_name = config.general.FEATURE_TABLE_NAME

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{model_asset_schema_name}.{model_name}",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{gold_schema_name}.{feature_table_name}",
)


# Create the online table
if not table_exists(
    catalog=catalog_name,
    schema=gold_schema_name,
    table=f"{feature_table_name}_online",
):
    feature_model_server.create_online_table()
    logger.info("Online Feature Table created")
else:
    feature_model_server.update_online_table(pipeline_id=pipeline_id)
    logger.info("Online Feature Table updated")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint(version=model_version)
logger.info("Started deployment/update of the serving endpoint")
