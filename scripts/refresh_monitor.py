import argparse
import os

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from airbnb_listing.config import get_config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.monitoring import create_or_refresh_monitoring

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

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Create or refresh the monitoring
create_or_refresh_monitoring(
    processed_inference_table_name=f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}.model_monitoring",
    config=config,
    env=args.env,
    spark=spark,
)
