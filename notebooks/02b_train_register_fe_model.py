# Databricks notebook source
%pip install ../../artifacts/.internal/airbnb_listing-0.0.10-py3-none-any.whl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.connect import DatabricksSession

# COMMAND ----------
import os
from airbnb_listing.config import Tags, get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.models.feature_lookup_model import FeatureLookUpModel

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
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

fe_model.tags

# COMMAND ----------

# Create the feature table
fe_model.create_feature_table()

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

# COMMAND ----------

# Register the model
fe_model.register_model()

# COMMAND ----------

# Testing: load the model and make predictions
# Lets run prediction on the last production model

test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA}.silver_airbnb_listing_price_test"
).limit(10)
X_test = test_set.drop("latitude", "longitude", "is_manhattan", config.model.TARGET)

# COMMAND ----------

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# COMMAND ----------

predictions.display()
