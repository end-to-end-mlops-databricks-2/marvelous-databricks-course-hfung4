from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from airbnb_listing.config import Config
from airbnb_listing.data_manager import get_env_catalog
from airbnb_listing.logging import logger


def process_inference_table(config: Config, env: str, spark: object) -> None:
    """Process the raw inference table and create a model monitoring table

    Args:
        config (Config): configuration object
        env (str): target environment
        spark (object): spark session
    """

    # Get the catalog name
    catalog_name = get_env_catalog(env, config)

    # raw inference table
    inf_table = spark.sql(
        f"SELECT * FROM {catalog_name}.{config.general.ML_ASSET_SCHEMA}.airbnb_listing_price_model_payload"
    )

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("id", StringType(), True),
                            StructField("neighbourhood", StringType(), True),
                            StructField("room_type", StringType(), True),
                            StructField("minimum_nights", DoubleType(), True),
                            StructField("estimated_listed_months", DoubleType(), True),
                            StructField("availability_365", DoubleType(), True),
                            StructField("number_of_reviews", DoubleType(), True),
                            StructField("calculated_host_listings_count", DoubleType(), True),
                            StructField("last_review", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [
                        StructField("trace", StringType(), True),
                        StructField("databricks_request_id", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    # Process the raw inference table

    # Convert json-like string to array for the columns "request" and "repsonse"
    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    # Explode the parsed_request.dataframe_records array
    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    # Rename columns
    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.id").alias("id"),
        F.col("record.neighbourhood").alias("neighbourhood"),
        F.col("record.room_type").alias("room_type"),
        F.col("record.minimum_nights").alias("minimum_nights"),
        F.col("record.estimated_listed_months").alias("estimated_listed_months"),
        F.col("record.availability_365").alias("availability_365"),
        F.col("record.number_of_reviews").alias("number_of_reviews"),
        F.col("record.calculated_host_listings_count").alias("calculated_host_listings_count"),
        F.col("record.last_review").alias("last_review"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("airbnb_listing_price_model_fe").alias("model_name"),  # create model name column
    )

    # Get test data and skewed inference data
    test_set = spark.table(f"{catalog_name}.{config.general.SILVER_SCHEMA}.airbnb_listing_price_test")

    inference_data_skewed = spark.table(f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}.inference_data_skewed")

    # Join the labels of the test set and the skewed inference data to the model log
    # so that I get a labeled inference dataframe
    df_final_with_status = (
        df_final.join(test_set.select("id", config.model.TARGET), on="id", how="left")
        .withColumnRenamed(config.model.TARGET, "log_price_test")
        .join(inference_data_skewed.select("id", config.model.TARGET), on="id", how="left")
        .withColumnRenamed(config.model.TARGET, "log_price_skewed")
        .select(
            "*",
            F.coalesce(F.col("log_price_test"), F.col("log_price_skewed")).alias("log_price"),
        )
        .drop("log_price_test", "log_price_skewed")
        .withColumn("log_price", F.col("log_price").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["log_price", "prediction"])
    )

    # Add the features from the feature table to the processed inference data
    feature_table_name = config.general.FEATURE_TABLE_NAME
    feature_table_df = spark.table(f"{catalog_name}.{config.general.GOLD_SCHEMA}.{feature_table_name}")

    df_final_with_features = df_final_with_status.join(feature_table_df, on="id", how="left")

    # Persist the processed inference data, I will build Lakehouse Monitoring Dashboard on this table
    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}.model_monitoring"
    )
    logger.info("✅ Processed model monitoring table created successfully!")


def create_or_refresh_monitoring(processed_inference_table_name, config, env, spark):
    """Create or refresh the Lakehouse Monitoring object, dashboard, and profile and drift metrics tables"""
    workspace = WorkspaceClient()

    try:
        # Get the Lakehouse Monitoring object
        workspace.quality_monitors.get(processed_inference_table_name)
        # If exist, refresh the Lakehouse Monitoring object
        workspace.quality_monitors.run_refresh(processed_inference_table_name)
        logger.info("✅ Lakehouse Monitoring object exists-- it has been refreshed successfully!")
    except NotFound:
        # If not exist, create the Lakehouse Monitoring object
        create_monitoring(
            processed_inference_table_name=processed_inference_table_name,
            config=config,
            env=env,
            spark=spark,
        )


def create_monitoring(
    processed_inference_table_name: str,
    config: Config,
    env: str,
    spark: object,
) -> None:
    """Create the Lakehouse Monitoring object, dashboard, and profile and drift metrics tables

    Args:
        processed_inference_table_name (str): name of the processed inference table
        config (Config): configuration object
        env (str): target environment
        spark (object): spark session
    """
    workspace = WorkspaceClient()

    logger.info("Creating new monitoring object...")

    catalog_name = get_env_catalog(env, config)

    workspace.quality_monitors.create(
        table_name=processed_inference_table_name,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{processed_inference_table_name}",
        output_schema_name=f"{catalog_name}.{config.general.ML_ASSET_SCHEMA}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="log_price",
        ),
    )
    logger.info("✅ Monitoring object successfully created!")

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {processed_inference_table_name} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
