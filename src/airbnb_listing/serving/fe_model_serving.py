import time

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

    def update_online_table(self, pipeline_id: str):
        """Triggers a Databricks pipeline update and monitors its state.

        Args:
            pipeline_id (str): Pipeline id of the DLT pipeline that is used to update the online table
        """
        # Since the (offline) feature table is updated, we need to update the online table also
        # we will trigger the DLT pipeline to update the online tabe
        update_response = self.workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=pipeline_id, update_id=update_response.update_id
            )
            # Get the state of the pipeline update
            state = update_info.update.state.value

            if state == "COMPLETED":
                logger.info("Pipline update completed successfully.")
                break
            elif state in ["FAILED", "CANCELED"]:
                logger.error("Pipeline update failed.")
                raise SystemError("Online table failed to update.")
            elif state == "WAITING_FOR_RESOURCES":
                logger.warning("Pipeline update is waiting for resources.")
            else:
                logger.info(f"Pipeline update state: {state}")
            time.sleep(30)

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
