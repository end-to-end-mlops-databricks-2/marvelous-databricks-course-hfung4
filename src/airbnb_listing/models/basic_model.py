import mlflow
import pandas as pd
from databricks.connect import DatabricksSession
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airbnb_listing.config import Config, Tags
from airbnb_listing.logging import logger


class BasicModel:
    def __init__(self, config: Config, tags: Tags, spark: DatabricksSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Get configuration variables
        self.num_features = self.config.model.SELECTED_NUMERIC_FEATURES
        self.cat_features = self.config.model.SELECTED_CATEGORICAL_FEATURES
        self.ID_COLUMN = self.config.model.ID_COLUMN
        self.target = self.config.model.TARGET
        self.parameters = self.config.model.MODEL_PARAMS
        self.catalog_name = (
            self.config.general.DEV_CATALOG
        )  # hardcoded for now, later it will be dependent on the target environment
        self.silver_schema = self.config.general.SILVER_SCHEMA
        self.gold_schema = self.config.general.GOLD_SCHEMA
        self.ml_asset_schema = self.config.general.ML_ASSET_SCHEMA

        # Mlflow configuration
        self.experiment_name = self.config.general.EXPERIMENT_NAME_BASIC
        self.model_name = f"{self.catalog_name}.{self.ml_asset_schema}.house_prices_model_basic"
        # self.tags = tags.model_dump()
        self.tags = tags.dict()

    def load_data(self):
        """
        Load train and test data from the silver layer.
        """
        logger.info("🔄 Loading data from silver layer...")
        # Train set in spark
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train")
        # Train set in pandas
        self.train_set = self.train_set_spark.toPandas()

        # Test set in pandas
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_test"
        ).toPandas()

        # Data version (hard coded, later will be retrived from DAB)
        # NOTE: with databricks-connect, we cannot use DESCRIBE HISTORY to get delta table version
        self.data_version = "0"

        # Get X_train, y_train, X_test, and y_test
        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]

        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Data loaded successfully.")

    def prepare_features(self):
        """
        Create train pipeline to process features and train model
        """
        logger.info("🔄 Defining train pipeline...")

        # Define the preprocessor step
        cat_pipeline = Pipeline(
            [
                (
                    "cat_imputer",
                    SimpleImputer(strategy="most_frequent", missing_values=None),
                ),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        num_pipeline = Pipeline(
            [
                ("num_imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, self.num_features),
                ("cat", cat_pipeline, self.cat_features),
            ],
            remainder="passthrough",
        )

        # Define the final pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )
        logger.info("✅ Train pipeline defined.")

    def train(self):
        """Train the pipeline"""
        logger.info("🚂 Training the model...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self):
        """Log the model"""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # Get the predictions from the model with test data
            y_pred = self.pipeline.predict(self.X_test)

            # Compute the test metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)

            # Log the test metrics
            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")

            # Log the test metrics in mlflow
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)

            # Log parameters in mlflow
            mlflow.log_param("model_type", "LGBMRegressor with preprocessing")
            mlflow.log_params(self.parameters)

            # Create a mlflow dataset object
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train",
                version=self.data_version,
            )
            # Log the train set in mlflow
            mlflow.log_input(dataset, context="training")

            # Log the model in mlflow
            signature = infer_signature(self.X_train, y_pred)
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
            )

    def register_model(self):
        """Register the model in the Unity Catalog model registry"""
        logger.info("📦 Registering model in Unity Catalog...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}")

        latest_version = registered_model.version

        # Set alias of the model
        client = MlflowClient()
        client.set_registered_model_alias(name=self.model_name, alias="latest-model", version=latest_version)

    def retrieve_current_run_dataset(self):
        """Retrieve MLflow dataset from current run"""
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """Retrieve Mlflow run metadata"""
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Args:
            input_data (pd.DataFrame): Input dataframe containing features for prediction
        """
        logger.info("🔄 Loading model from MLflow alias 'latest-model'...")
        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        return predictions
