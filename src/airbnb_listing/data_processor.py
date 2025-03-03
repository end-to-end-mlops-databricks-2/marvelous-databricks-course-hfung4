import time

import numpy as np
import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from airbnb_listing.config import config
from airbnb_listing.logging import logger

spark = DatabricksSession.builder.getOrCreate()


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df  # Store the DataFrame as self.df
        # self.config = config  # Store the configuration

    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data and perform feature engineering

        Returns:
            pd.DataFrame: processed dataframe
        """
        # Drop all rows with missing values in the target column
        self.df.dropna(subset=["price"], inplace=True)

        # Convert certain float columns to Int64 (pandas nullable integer type)
        # for col in config.model.INTEGER_COLUMNS:
        #    self.df[col] = self.df[col].astype("Int64")  # Nullable integer

        # Convert the id column to a string
        self.df[config.model.ID_COLUMN] = self.df[config.model.ID_COLUMN].astype(str)

        # Log the price
        self.df["log_price"] = np.log1p(self.df["price"])

        # Drop duplicates
        self.df.drop_duplicates(inplace=True)

        # Create a is_manhattan column
        self.df["is_manhattan"] = self.df["neighbourhood_group"] == "Manhattan"

        # Cap minimum nights at 14
        self.df["minimum_nights"] = self.df["minimum_nights"].clip(upper=14)

        # elapse time since last review
        # self.df["last_review"] = pd.to_datetime(self.df["last_review"], format="%Y-%m-%d", errors="coerce")
        # NOTE: days_since_last_review is now created with feature function
        # self.df["days_since_last_review"] = (
        #    datetime.now() - self.df["last_review"]
        # ).dt.days

        # Estimate for how long a house has been listed.
        # This duration is calculated by dividing the total number of reviews
        # that the house has received by the number of reviews per month.
        # Handles division by zero
        self.df["estimated_listed_months"] = np.where(
            self.df["reviews_per_month"] == 0,
            np.nan,
            self.df["number_of_reviews"] / self.df["reviews_per_month"],
        )

        # Lump rare neghbourhoods into 'Other'
        neighbourhood_percentage = self.df["neighbourhood"].value_counts(normalize=True) * 100
        self.df["neighbourhood"] = self.df["neighbourhood"].where(
            self.df["neighbourhood"].map(neighbourhood_percentage) >= config.model.THRESHOLD_NEIGHBOURHOOD,
            "Other",
        )

        # Select the columns to be used for traing
        selected_columns = (
            [config.model.ID_COLUMN]
            + config.model.SELECTED_CATEGORICAL_FEATURES
            + config.model.SELECTED_NUMERIC_FEATURES
            + config.model.SELECTED_TIMESTAMP_FEATURES
            + [config.model.TARGET]
        )
        self.df = self.df.loc[:, selected_columns]

        return self.df

    def write_processed_data(self, df: pd.DataFrame, table_name: str):
        """Write the processed data to a file

        Args:
            df (pd.DataFrame): processed dataframe
            table_name (str): three-level name of the table in Unity Catalog
        """
        # Convert the processed pandas dataFrame to a Spark DataFrame
        processed_spark = spark.createDataFrame(df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Write table to Unity Catalog

        if config.general.GENERATE_AND_APPEND_SYN_DATA:
            # Append synthetic data to the existing table
            processed_spark.write.mode("append").saveAsTable(table_name)
        else:
            # Reading and processing bronze data for the first time
            processed_spark.write.mode("overwrite").saveAsTable(table_name)

        # Modify a Delta table property to enable Change Data Feed (CDF)
        # CDF allows tracking row-level changes (INSERT, UPDATE, DELETE) in Delta Tables.
        # With CDF enabled, you can query changes since a specific version or timestamp.
        # This is useful for incremental data processing, audting, and real-time analytics.
        spark.sql(f"ALTER TABLE {table_name} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        logger.info(f"Data written to {table_name} in Unity Catalog.")


def generate_synthetic_data(df, num_rows=10):
    """Generate synthetic data based on the distribution of the input DataFrame.

    Args:
        df (pd.DataFrame): input DataFrame
        num_rows (int): number of rows to generate

    Returns:
        pd.DataFrame: synthetic DataFrame
    """
    synthetic_data = pd.DataFrame()

    numeric_treat_as_objects = ["host_id", "latitude", "longitude"]
    round_to_nearest_whole_num = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
    ]

    for column in df.columns:
        # Skip the id column
        if column == config.model.ID_COLUMN:
            continue

        # Handling numeric data
        if pd.api.types.is_numeric_dtype(df[column]) and column not in numeric_treat_as_objects:
            # Generate synthetic data based on the distribution of the input data
            synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)
            # Ensure the generated data is non-negative
            synthetic_data[column] = synthetic_data[column].abs()

        # Handling categorical data
        elif pd.api.types.is_object_dtype(df[column]) or column in numeric_treat_as_objects:
            value_counts = df[column].value_counts(normalize=True)
            unique_values = value_counts.index.astype(str)
            probabilities = value_counts.values
            synthetic_data[column] = np.random.choice(unique_values, num_rows, p=probabilities)

    # Handle Latitude & Longitude Sampling from the Same Neighbourhood
    if {"neighbourhood", "latitude", "longitude"}.issubset(df.columns):
        synthetic_data["latitude"] = np.nan
        synthetic_data["longitude"] = np.nan

        for i in range(num_rows):
            neighbourhood = synthetic_data.loc[i, "neighbourhood"]
            if pd.notna(neighbourhood) and neighbourhood in df["neighbourhood"].values:
                subset = df[df["neighbourhood"] == neighbourhood]
                if not subset.empty:
                    sampled_row = subset.sample(n=1)
                    synthetic_data.at[i, "latitude"] = sampled_row["latitude"].values[0]
                    synthetic_data.at[i, "longitude"] = sampled_row["longitude"].values[0]
    # Round certain float columns to integers
    synthetic_data[round_to_nearest_whole_num] = synthetic_data[round_to_nearest_whole_num].round(0)

    # Generate id
    timestamp_base = int(time.time() * 1000)
    synthetic_data[config.model.ID_COLUMN] = [str(timestamp_base + i) for i in range(num_rows)]

    # Reorder columns to match the input DataFrame ordering
    ordered_columns = [col for col in df.columns if col in synthetic_data.columns]
    synthetic_data = synthetic_data[ordered_columns]

    # Cast host_id to float and id to int32
    synthetic_data["host_id"] = synthetic_data["host_id"].astype(float)
    synthetic_data["id"] = synthetic_data["id"].astype("int32")

    return synthetic_data
