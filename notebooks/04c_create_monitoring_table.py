# Databricks notebook source

# COMMAND ----------
import os
from databricks.connect import DatabricksSession
from airbnb_listing.config import get_config
from databricks.sdk import WorkspaceClient
from airbnb_listing.monitoring import create_or_refresh_monitoring

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

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

# COMMAND ----------
create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
