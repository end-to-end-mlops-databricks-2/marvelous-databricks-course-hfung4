from databricks import feature_engineering
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from airbnb_listing.config import Config, Tags


# Feature Lookup Model
class FeatureLookUpModel:
    def __init__(self, config: Config, tags: Tags, spark: DatabricksSession):
        """initialize the FeatureLookUpModel class

        Args:
            config (Config): configuration object
            tags (Tags): tag object
            spark (DatabricksSession): spark session
        """
        self.config = config
        self.tags = tags
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Get configuration variables
