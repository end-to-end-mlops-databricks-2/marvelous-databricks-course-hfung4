import mlflow

from airbnb_listing.config import config
from airbnb_listing.logging import logger
from airbnb_listing.serving.fe_model_serving import FeatureLookupServing

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Define catalog, schema, and feature table, feature spec, and endpoint names
catalog_name = config.general.DEV_CATALOG
model_asset_schema_name = config.general.ML_ASSET_SCHEMA
silver_schema_name = config.general.SILVER_SCHEMA
gold_schema_name = config.general.GOLD_SCHEMA
endpoint_name = "airbnb-listing-model-serving-fe"
model_name = f"{config.model.MODEL_NAME}_fe"
feature_table_name = config.general.FEATURE_TABLE_NAME

# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{model_asset_schema_name}.{model_name}",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{gold_schema_name}.{feature_table_name}",
)

# Create the online table
feature_model_server.create_online_table()
logger.info("Created online table")

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()
logger.info("Started deployment/update of the serving endpoint")
