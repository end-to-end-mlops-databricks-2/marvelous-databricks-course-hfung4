from pathlib import Path
from typing import Any, Dict, List, Optional

# Yaml
import yaml

# Pydantic
from pydantic import BaseModel

# Project Directories

PACKAGE_ROOT = Path(__file__).resolve().parent  # path/to/airbnb_listing
ROOT = PACKAGE_ROOT.parent.parent  # repo root
LOGS_DIR = Path(ROOT / "logs")
TESTS_DIR = Path(ROOT / "tests")
ENV_DIR = Path(ROOT / ".env")


# Pydantic model for project configuration
class GeneralConfig(BaseModel):
    RANDOM_STATE: int
    RUN_ON_DATABRICKS_WORKSPACE: bool
    GENERATE_AND_APPEND_SYN_DATA: bool
    DEV_CATALOG: str
    STAGING_CATALOG: str
    PROD_CATALOG: str
    BRONZE_SCHEMA: str
    SILVER_SCHEMA: str
    GOLD_SCHEMA: str
    ML_ASSET_SCHEMA: str
    DEV_PIPELINE_ID: str
    STAGING_PIPELINE_ID: str
    PROD_PIPELINE_ID: str
    FEATURE_TABLE_NAME: str
    EXPERIMENT_NAME_FE: Optional[str]
    EXPERIMENT_NAME_BASIC: Optional[str]


class ModelConfig(BaseModel):
    MODEL_NAME: str
    TARGET: str
    ID_COLUMN: str
    INTEGER_COLUMNS: List[str]
    SELECTED_CATEGORICAL_FEATURES: List[str]
    SELECTED_NUMERIC_FEATURES: List[str]
    SELECTED_TIMESTAMP_FEATURES: List[str]
    THRESHOLD_NEIGHBOURHOOD: float
    TEST_SIZE: float
    MODEL_PARAMS: Dict[str, Any]  # Dictionary to hold model-related parameters


# Master config object
class Config(BaseModel):
    """Master config object."""

    general: GeneralConfig
    model: ModelConfig


def fetch_config_from_yaml(cfg_path: Path = None) -> object:
    """Parse YAML containing the package configuration
    Args:
        cfg_path (Path, optional): Path to the configuration yaml. Defaults to None.

    Raises:
        OSError: Cannot find the config file at the specified path

    Returns:
        parsed_config: parsed configuration from yaml
    """
    with open(cfg_path, "r") as conf_file:
        parsed_config = yaml.safe_load(conf_file)
        return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def get_config(config_file_path: Path) -> Config:
    """Run validation on config values, and return a validated config object

    Args:
        config_file_path (Path): Path to the configuration yaml

    Returns:
        Config: validated configuration object
    """
    parsed_config = fetch_config_from_yaml(config_file_path)

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        general=GeneralConfig(**parsed_config),
        model=ModelConfig(**parsed_config),
    )

    return _config


# Validated tags
class Tags(BaseModel):
    git_sha: str
    branch: str
