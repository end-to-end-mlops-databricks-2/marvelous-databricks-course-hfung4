# Databricks notebook source
%pip install ../../artifacts/.internal/airbnb_listing-0.0.9-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.connect import DatabricksSession

# COMMAND ----------
from airbnb_listing.config import Tags, get_config
from airbnb_listing.data_manager import get_env_catalog, table_exists
from airbnb_listing.models.feature_lookup_model import FeatureLookUpModel
from airbnb_listing.logging import logger

# COMMAND ----------
# NOTE: hardcoded in notebooks, get env from DAB in scripts
env = "dev"  # hardcoded

# COMMAND ----------
# Load the configuration

# NOTE: Hardcoded path in notebook, get root path from DAB in scripts
root_path = "/Workspace/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/.bundle"
env_root_path = f"{root_path}/{env}/marvelous-databricks-course-hfung4/files"
config_path = f"{env_root_path}/project_config.yml"
config = get_config(config_path)

# Get the catalog name based on the environment
catalog_name = get_env_catalog(env, config)

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# raw tags
# NOTE: hardcoded in notebook, get git_sha and branch from DAB in scripts
tags_dict = {"git_sha": "abcd12345", "branch": "model_training"}
# validated tags
tags = Tags(**tags_dict)
tags.git_sha

# COMMAND ----------

# Initialize the FeatureLookUpModel
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark, env=env)

# COMMAND ----------

fe_model.tags

# COMMAND ----------

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

# COMMAND ----------

# Define the `days_since_last_review` feature function
fe_model.create_feature_function()

# COMMAND ----------

# Load silver data
fe_model.load_silver_data()

# COMMAND ----------
fe_model.train_set.display()


# COMMAND ----------
fe_model.test_set.display()

# COMMAND ----------

# Perform feature engineering and create training set
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()
logger.info("✅ Model training complete.")

# COMMAND ----------
# Evaluate the model
test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test"
).limit(100)

# Drop the columns in the feature table
test_set = test_set.drop("latitude", "longitude", "is_manhattan")

test_set.display()

# COMMAND ----------
# Check if the model already exists, or I have a new model
model_exists = fe_model.check_model_exists()
model_exists

# COMMAND ----------
# If the model already exist (at least one model version in the Model Registry), I will check if the new model is better than the existing model
if model_exists:
    model_improved = fe_model.model_improved(test_set)
    logger.info(f"Model evaluation completed. Model improved: {model_improved}")

# COMMAND ----------

# Register the model if there is a model performance improvement, or I have a new model
if model_improved or not model_exists:
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
else:
    logger.info("Model did not improve, no new model registered.")

# COMMAND ----------
# Make predictions
predictions = fe_model.load_latest_model_and_predict(test_set)

# COMMAND ----------

predictions.display()
