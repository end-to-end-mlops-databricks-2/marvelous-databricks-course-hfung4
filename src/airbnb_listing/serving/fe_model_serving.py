import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

from airbnb_listing.config import config
from airbnb_listing.logging import logger


class FeatureLookupServing:
    def __init__(self, model_name: str, endpoint_name: str, feature_table_name: str):
        """Initializes the Feature Lookup Serving Manager.

        Args:
            model_name (str): name of the model
            endpoint_name (str): name of the endpoint
            feature_table_name (str): name of the feature table that will be used to retrieve some features for the model at inference
        """
        self.workspace = WorkspaceClient()
        self.feature_table_name = feature_table_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.model_name = model_name
        self.endpoint_name = endpoint_name

    def create_online_table(self):
        """Creates an online table based on a feature table."""
        spec = OnlineTableSpec(
            primary_key_columns=config.model.ID_COLUMN,
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

    def get_latest_model_version(self):
        """Gets the latest version of the model."""
        client = mlflow.MlflowClient()
        # Get the latest version of the model (a string)
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ):
        """Deploys the model serving endpoint in Databricks

        Args:
            version (str, optional): Version of the model to deploy. Defaults to "latest", in this case, I will retrieve the latest version of the model.
            workload_size (str, optional): Workload size (number of concurrent requests). Default is Small = 4 concurrent requests.
            scale_to_zero (bool, optional): If True, endpoint scales to 0 when unused. Defaults to True.
        """

        # Check if endpoint already exists
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        if version == "latest":
            entity_version = self.get_latest_model_version()
        else:
            # Use the version provided by the user
            entity_version = version

        # Specify the entities to be served
        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            # create the endpoint
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
        else:
            # update the endpoint
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
