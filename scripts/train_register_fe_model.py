import argparse
import os

import mlflow
from databricks.connect import DatabricksSession
from pyspark.dbutils import DBUtils

from airbnb_listing.config import Tags, get_config
from airbnb_listing.data_manager import get_env_catalog, table_exists
from airbnb_listing.logging import logger
from airbnb_listing.models.feature_lookup_model import FeatureLookUpModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)

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

# NOTE: root path is: /Workspace/Users/<user email>/.bundle/
config_path = f"{args.root_path}/{args.env}/airbnb_listing/files/project_config.yml"

# If running locally, change the root path
if not os.path.exists(config_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    new_root_path = os.path.dirname(current_dir)  # Move one level up
    config_path = f"{new_root_path}/project_config.yml"

config = get_config(config_path)

catalog_name = get_env_catalog(env=args.env, config=config)
dbutils = DBUtils(spark)

# raw tags
tags_dict = {
    "git_sha": args.git_sha,
    "branch": args.branch,
    "job_run_id": args.job_run_id,
}
# validated tags
tags = Tags(**tags_dict)

# Initialize the FeatureLookUpModel
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark, env=args.env)

# Create the feature table
if not table_exists(
    catalog=catalog_name,
    schema=config.general.GOLD_SCHEMA,
    table=config.general.FEATURE_TABLE_NAME,
):
    # Feature table does not exist, create it
    fe_model.create_feature_table()
    logger.info("Feature table created.")
else:
    # Feature table already exists, update it
    fe_model.update_feature_table()
    logger.info("Feature table updated.")

# Define the `days_since_last_review` feature function
fe_model.create_feature_function()

# Load silver data
fe_model.load_silver_data()

# Perform feature engineering and create training set
fe_model.feature_engineering()

# Train the model
fe_model.train()
logger.info("✅ Model training complete.")

# Evaluate the model
test_set = spark.table(f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test").limit(100)

# Drop the columns in the feature table
test_set = test_set.drop("latitude", "longitude", "is_manhattan")

# Check if the model already exist (i.e. there is at least one model version in the registry)
model_exists = fe_model.check_model_exists()

if model_exists:
    # Get the "model_improved" flag
    model_improved = fe_model.model_improved(test_set)
    logger.info(f"Model evaluation completed. Model improved: {model_improved}")

# Register the model if there is a model performance improvement, or I have a new model
if model_improved or not model_exists:
    latest_version = fe_model.register_model()
    logger.info(f"New model registered with version:{latest_version}")
    # Log the model version and update flag to be passed to the next task
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)  # set to 1 if model_updated is True
else:
    # We don't register the model if it didn't improved
    dbutils.jobs.taskValues.set(key="model_updated", value=0)  # set to 0 if model_updated is False
