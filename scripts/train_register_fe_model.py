import argparse

import mlflow
from databricks.connect import DatabricksSession
from pyspark.dbutils import DBUtils

from airbnb_listing.config import Tags, config
from airbnb_listing.data_manager import get_env_catalog, table_exists
from airbnb_listing.logging import logger
from airbnb_listing.models.feature_lookup_model import FeatureLookUpModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

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

spark = DatabricksSession.builder.getOrCreate()
args = parser.parse_args()

catalog_name = get_env_catalog(env=args.env)
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
logger.info("Model training complete.")

# Evaluate the model
test_set = spark.table(f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test").limit(100)

# Drop the columns in the feature table
test_set = test_set.drop("latitude", "longitude", "is_manhattan")

# Get the "model_improved" flag
model_improved = fe_model.model_improved(test_set)
logger.info(f"Model evaluation completed. Model improved: {model_improved}")

# Register the model
if model_improved:
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    # Log the model version and update flag to be passed to the next task
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)  # set to 1 if model_updated is True
else:
    # We don't register the model if it didn't improve
    dbutils.jobs.taskValues.set(key="model_updated", value=0)  # set to 0 if model_updated is False
