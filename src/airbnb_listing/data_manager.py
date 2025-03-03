from databricks.connect import DatabricksSession

from airbnb_listing.config import config


def get_env_catalog(env):
    """Get the catalog based on the environment

    Args:
        env (str): target environment

    Raises:
        ValueError: Invalid environment

    Returns:
        str: catalog name
    """

    if env == "dev":
        return config.general.DEV_CATALOG
    elif env == "staging":
        return config.general.STAGING_CATALOG
    elif env == "prod":
        return config.general.PROD_CATALOG
    else:
        raise ValueError(f"Invalid environment: {env}")


def get_env_pipeline_id(env):
    """Get the pipeline ID based on the environment

    Args:
        env (str): target environment

    Raises:
        ValueError: Invalid environment

    Returns:
        str: pipeline ID
    """

    if env == "dev":
        return config.general.DEV_PIPELINE_ID
    elif env == "staging":
        return config.general.STAGING_PIPELINE_ID
    elif env == "prod":
        return config.general.PROD_PIPELINE_ID
    else:
        raise ValueError(f"Invalid environment: {env}")


def table_exists(catalog, schema, table):
    """Check if a table exists in the catalog

    Args:
        catalog (str): catalog name
        schema (str): schema name
        table (str): table name

    Returns:
        bool: True if the table exists, False otherwise
    """
    spark = DatabricksSession.builder.getOrCreate()

    table_exists = spark.sql(f"SHOW TABLES IN {catalog}.{schema} LIKE '{table}'").count() > 0

    return table_exists
