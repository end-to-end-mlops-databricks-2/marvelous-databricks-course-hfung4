from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

from airbnb_listing.config import config
from airbnb_listing.logging import logger


# Feature Serving Manager
class FeatureServing:
    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str):
        """Initializes the prediction serving manager

        Args:
            feature_table_name (str): user specified name of the feature table
            feature_spec_name (str): user specified name of the feature spec
            endpoint_name (str): user specifed name of the endpoint
        """
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_online_table(self):
        """Create an online table based on a feature table"""
        spec = OnlineTableSpec(
            primary_key_columns=[config.model.ID_COLUMN],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        try:
            self.workspace.online_tables.get(name=self.online_table_name)
            logger.info(f"Online table {self.online_table_name} already exists.")
        except NotFound:
            self.workspace.online_tables.create(name=self.online_table_name, spec=spec)
            logger.info(f"Online table {self.online_table_name} created.")

    def create_feature_spec(self):
        """Create a feature spec to be served with Feature Serving"""
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key=config.model.ID_COLUMN,
                feature_names=[
                    "longitude",
                    "latitude",
                    "is_manhattan",
                    "predicted_listing_price",
                ],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features, exclude_columns=None)

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True):
        """Deploys the feature serving endpoint in Databricks

        Args:
            workload_size (str, optional): Workload size for the Serving Endpoint. Defaults to "Small".
            scale_to_zero (bool, optional): Scale to zero option. Defaults to True.
        """
        # True if the endpoint already exists
        endpoint_exists = any(item.name == self.endpoint_name for item in self.worksapce.serving_endpoints.list())

        # A list of served entities behind the endpoint, we will serve a feature spec for this endpoint
        served_entities = [
            ServedEntityInput(
                # Serve the feature spec we created
                entity_name=self.feature_spec_name,
                scale_to_zero_eanbled=scale_to_zero,
                workload_size=workload_size,
            )
        ]

        # If the endpoint does not exist, create it; otherwise, if it already exists, update it
        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entites=served_entities)
