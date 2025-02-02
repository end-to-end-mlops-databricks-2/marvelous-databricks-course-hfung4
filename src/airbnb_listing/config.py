import os
from pathlib import Path
from typing import Dict, List, Literal

# Pydantic
from pydantic import BaseModel

# Yaml
import yaml

# Project Directories

# Move up one levels from `config`
# (e.g., the immediate parent directory path/to/config, and the grandparent directory path/to/ptb_ltc)
PACKAGE_ROOT = Path(__file__).resolve().parent  # path/to/airbnb_listing
ROOT = PACKAGE_ROOT.parent.parent  # two levels up from /airbnb_listing
LOGS_DIR = Path(ROOT / "logs")
TESTS_DIR = Path(ROOT / "tests")
CONFIG_FILE_PATH = ROOT / "project_config.yml"


# Pydantic model for project configuration
class GeneralConfig(BaseModel):
    RANDOM_STATE: int
    MODEL_NAME: str
    RUN_ON_DATABRICKS_WORKSPACE: bool


class ModelConfig(BaseModel):
    TARGET: str
    ID_COLUMN: str
    INTEGER_COLUMNS: List[str]
    SELECTED_CATEGORICAL_FEATURES: List[str]
    SELECTED_NUMERIC_FEATURES: List[str]
    SELECTED_TEXT_FEATURES: List[str]
    THRESHOLD_NEIGHBOURHOOD: float


# Master config object
class Config(BaseModel):
    """Master config object."""

    general: GeneralConfig
    model: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file.

    Raises:
        Exception: Configuration file not found at the specified path

    Returns:
        Path: path to the configuration file
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> object:
    """Parse YAML containing the package configuration
    Args:
        cfg_path (Path, optional): Path to the configuration yaml. Defaults to None.

    Raises:
        OSError: Cannot find the config file at the specified path

    Returns:
        parsed_config: parsed configuration from yaml
    """
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = yaml.safe_load(conf_file)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: object = None) -> Config:
    """Run validation on config values

    Args:
        parsed_config (object, optional): parsed configuration object. Defaults to None.

    Returns:
        Config: validated configuration object
    """

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml(CONFIG_FILE_PATH)

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        general=GeneralConfig(**parsed_config),
        model=ModelConfig(**parsed_config),
    )

    return _config


# Load the configuration
config = create_and_validate_config()
