import pandas as pd
import numpy as np
from datetime import datetime
from airbnb_listing.config import config


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame):
        self.df = pandas_df  # Store the DataFrame as self.df
        # self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the data and perform feature engineering

        Returns:
            pd.DataFrame: processed dataframe
        """
        # Drop all rows with missing values in the target column
        self.df.dropna(subset=["price"], inplace=True)

        # Convert certain float columns to Int64 (pandas nullable integer type)
        for col in config.model.INTEGER_COLUMNS:
            self.df[col] = self.df[col].astype("Int64")  # Nullable integer

        # Log the price
        self.df["log_price"] = np.log1p(self.df["price"])

        # Drop duplicates
        self.df.drop_duplicates(inplace=True)

        # Create a is_manhattan column
        self.df["is_manhattan"] = self.df["neighbourhood_group"] == "Manhattan"

        # Cap minimum nights at 14
        self.df["minimum_nights"] = self.df["minimum_nights"].clip(upper=14)

        # elapse time since last review
        self.df["last_review"] = pd.to_datetime(
            self.df["last_review"], format="%Y-%m-%d", errors="coerce"
        )
        self.df["days_since_last_review"] = (
            datetime.now() - self.df["last_review"]
        ).dt.days

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
        neighbourhood_percentage = (
            self.df["neighbourhood"].value_counts(normalize=True) * 100
        )
        self.df["neighbourhood"] = self.df["neighbourhood"].where(
            self.df["neighbourhood"].map(neighbourhood_percentage)
            >= config.model.THRESHOLD_NEIGHBOURHOOD,
            "Other",
        )

        # Select the columns to be used for traing
        selected_columns = (
            [config.model.ID_COLUMN]
            + config.model.SELECTED_CATEGORICAL_FEATURES
            + config.model.SELECTED_NUMERIC_FEATURES
            + config.model.SELECTED_TEXT_FEATURES
            + [config.model.TARGET]
        )
        self.df = self.df.loc[:, selected_columns]

        return self.df
