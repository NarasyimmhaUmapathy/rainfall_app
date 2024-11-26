from pydantic import BaseModel
from typing import Dict, List, Optional, Sequence
import yaml
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os


class DataConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int

class ModelConfig(BaseModel):
    ### has the params of the model which is gonna be tuned###
    scorer : List[str]
    grid_params : List[str]
    max_iter: int
    features: List[str]
    num_features: List[str]
    cat_features: List[str]


 

class Config(BaseModel):
    """Master config object."""

    data_config: DataConfig
    b_config: ModelConfig


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> yaml:
    """Parse YAML containing the package configuration."""

    with open("config.yml", "r") as conf_file:
        parsed_config = yaml.safe_load(conf_file.read())
    return parsed_config


def create_and_validate_config(parsed_config: yaml = None) -> Config:
    """Run validation on config values."""

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        b_config=ModelConfig(**parsed_config),
        data_config=DataConfig(**parsed_config)

    )

    return _config





conf_data = fetch_config_from_yaml()
#config = create_and_validate_config(conf_data)
print(conf_data)
