from databricks import feature_engineering
from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient

from airbnb_listing.config import Config, Tags
from airbnb_listing.logging import logger


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

        # Define the feature table name and feature function name
        self.feature_table_name = f"{self.catalog_name}.{self.gold_schema}.{self.config.general.FEATURE_TABLE_NAME}"
        self.function_name = f"{self.catalog_name}.{self.gold_schema}.calculate_date_since_last_review"

        # Mlflow configuration
        self.experiment_name = self.config.general.EXPERIMENT_NAME_FE
        self.tags = tags.model_dump()

    def create_feature_table(self):
        """Create the feature table in the gold layer."""

        query = f"""
            CREATE OR REPLACE TABLE {self.feature_table_name} (
            {self.ID_COLUMN} STRING NOT NULL,
            latitude DOUBLE,
            longitude DOUBLE,
            is_manhattan BOOLEAN
        );
        """
        self.spark.sql(query)

        # Set primary key
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT listing_pk PRIMARY KEY ({self.ID_COLUMN});"
        )
        # Set table properties to enable change data feed (needed for creating online tables)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # Insert train and test set into the feature table to ensure we can perform feature lookup for
        # ALL listings (either used for training and testing). If there is an unknonw/new listing, then
        # we cannot perform feature lookup, and therefore we cannot make predictions for it.

        # Insert train set
        query = f""" INSERT INTO {self.feature_table_name}
         SELECT {self.ID_COLUMN}, latitude, longitude, is_manhattan FROM
            {self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train"""
        self.spark.sql(query)

        # Insert test set
        query = f""" INSERT INTO {self.feature_table_name}
         SELECT {self.ID_COLUMN}, latitude, longitude, is_manhattan FROM
            {self.catalog_name}.{self.silver_schema}.airbnb_listing_price_test"""
        self.spark.sql(query)

        logger.info("✅ Feature table created and populated.")

    # Feature function definition
    def create_feature_function(self):
        """Define a function to compute date since last review"""
        query = f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(last_review TIMESTAMP)
        RETURNS DOUBLE
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        return (datetime.now() - last_review).dt.days
        $$
        """
        self.spark.sql(query)

    # Load silver train and test data
    def load_silver_data(self):
        """Load train and test data from the silver layer"""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_train").drop(
            "latitude", "longitude", "is_manhattan"
        )
        self.train_set = self.train_set.withColumn("last_review", self.train_set["last_review"].cast("timestamp"))

        # Since I need the test set to evaluate the trained model and compute performance metrics, I need
        # all features (including the ones that will be retrieved from the feature table)
        self.test_set = (
            self.spark.table(f"{self.catalog_name}.{self.silver_schema}.airbnb_listing_price_test")
            .drop("last_review")
            .toPandas()
        )

        logger.info("✅ Data loaded successfully.")

    # Create training set with features from 1) silver train set, 2) feature table, and 3) feature function
    def feature_engineering(self):
        """Perform feature engineering by linking silver ata with feature tables"""
        # Create the specification for the training set
        self.training_set_spec = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                # If I want to use multiple feature tables, I can add additional FeatureLookup objects
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["latitude", "longitude", "is_manhattan"],
                    lookup_key=self.ID_COLUMN,
                ),
                # FeatureFunction(
                #    udf_name=self.function_name,
                #    output_name="days_since_last_review",
                #    # key is input argument of the function, value is the column name in input dataframe
                #    input_bindings={"last_review": "last_review"},
                # ),
            ],
            exclude_columns=["last_review", "update_timestamp_utc"],
        )

        # Create the training set (in Pandas)
        # self.training_df = self.training_set_spec.load_df().toPandas()

        # Create the days_since_last_review feature for the test set that is used
        # to evaluate the trained model
        # self.test_set["days_since_last_review"] = (
        #    datetime.now() - self.test_set["last_review"]
        # ).dt.days

        # Create X_train, y_train, X_test, y_test for model training and evaluation
        # self.X_train = self.training_df[
        #    self.num_features + self.cat_features + ["days_since_last_review"]
        # ]
        # self.y_train = self.training_df[self.target]
        # self.X_test = self.test_set[
        #    self.num_features + self.cat_features + ["days_since_last_review"]
        # ]
        # self.y_test = self.test_set[self.target]

        # logger.info("✅ Feature engineering completed.")
