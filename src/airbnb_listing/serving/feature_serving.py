from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

from airbnb_listing.config import config


# Feature Serving Manager that serves precomputed features and/or predictions
class FeatureServing:
    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str):
        """Initializes the prediction serving manager"""
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = (f"{self.feature_table_name}_online",)
        self.fe = feature_engineering.FeatureEngineeringClient()

    def create_online_table(self):
        """Create an online table based on a feature table"""
        spec = OnlineTableSpec(
            primary_key_columns=[config.model.ID_COLUMN],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

    def create_feature_spec(self):
        """Create a feature spec to be served with Feature Serving"""
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key=config.model.ID_COLUMN,
                feature_names=["longitude", "latitude", "predicted_listing_price"],
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name, features=features)
