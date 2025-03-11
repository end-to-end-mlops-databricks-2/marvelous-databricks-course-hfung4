# Databricks notebook source

from databricks.connect import DatabricksSession
from airbnb_listing.data_processor import DataProcessor, generate_synthetic_data
from airbnb_listing.config import get_config
import os


# COMMAND ----------
# NOTE: root path is: /Workspace/Users/<user email>/.bundle/
current_dir = os.path.dirname(os.path.abspath(__file__))
new_root_path = os.path.dirname(current_dir)  # Move one level up
config_path = f"{new_root_path}/project_config.yml"
config = get_config(config_path)
# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate()

bronze_table_name = f"prod.bronze.airbnb_listing_price"
bronze = spark.table(bronze_table_name).toPandas()

# COMMAND ----------
bronze.info()
# COMMAND ----------
bronze.neighbourhood_group.value_counts(normalize=True)  # 93% 5% 2% (52;45;2)

# COMMAND ----------
synthetic_df = generate_synthetic_data(
    df=bronze, config=config, num_rows=100, drift=False
)
# COMMAND ----------
synthetic_df.neighbourhood_group.value_counts(normalize=True)
# COMMAND ----------
skewed_df = generate_synthetic_data(df=bronze, config=config, num_rows=100, drift=True)
# COMMAND ----------
skewed_df.neighbourhood_group.value_counts(normalize=True)
# COMMAND ----------
bronze.room_type.value_counts(normalize=True)
# COMMAND ----------
data_processor_skewed = DataProcessor(skewed_df, config)
# COMMAND ----------
silver_skewed = data_processor_skewed.preprocess()
# COMMAND ----------
silver_skewed.shape
# COMMAND ----------
silver_skewed.is_manhattan.value_counts(normalize=True)
# COMMAND ----------
silver_skewed.room_type.value_counts(normalize=True)
# COMMAND ----------
