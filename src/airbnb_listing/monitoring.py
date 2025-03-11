from airbnb_listing.data_manager import get_env_catalog


def create_or_refresh_monitoring(config, env, spark, workspace):
    # Get the catalog name
    catalog_name = get_env_catalog(env, config)

    # raw inference table
    inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{config.general.ASSET}.`model-serving-fe_payload_payload`")
    return inf_table
