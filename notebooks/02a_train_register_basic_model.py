# Databricks notebook source
import mlflow
import os
from databricks.connect import DatabricksSession
from airbnb_listing.config import Tags, get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.models.basic_model import BasicModel

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
tags_dict = {"git_sha": "abcd12345", "branch": "model_training"}  # hardcoded
# validated tags
tags = Tags(**tags_dict)
tags.git_sha

# COMMAND ----------

# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

basic_model.load_data()


# COMMAND ----------

# Define the train pipeline
basic_model.prepare_features()


# COMMAND ----------

# Train and log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------

# Get experiment run id
run_id = mlflow.search_runs(
    experiment_names=[
        "/Users/henryhfung4_gmail.com#ext#@henryhfung4gmail.onmicrosoft.com/airbnb_listing_price_basic"
    ],
    filter_string="tags.branch='model_training'",
).run_id[0]
print(run_id)

# COMMAND ----------

# Load the model from the current experiment run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------

# Retrieve dataset from the current experiment run
display(basic_model.retrieve_current_run_dataset().limit(5))

# COMMAND ----------

basic_model.retrieve_current_run_metadata()

# COMMAND ----------

# Register model to the Unity Catalog Model Registry
basic_model.register_model()

# COMMAND ----------

# Perform inference with the registered model using the test set
test_set = spark.table(
    f"{catalog_name}.{config.general.SILVER_SCHEMA}.silver_airbnb_listing_price_test"
)

# COMMAND ----------

X_test = test_set.drop(config.model.TARGET).limit(10).toPandas()
X_test

# COMMAND ----------

predictions_df = basic_model.load_latest_model_and_predict(X_test)

# COMMAND ----------

predictions_df
