import os
from typing import Optional, Union, Any, Dict, Tuple, List
import numpy as np
from pathlib import Path
import yaml
import logging
import random
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BertForSequenceClassification,
)


class Config:
    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def dict2config(data: Dict[str, Any]) -> Any:
    """
    Recursively convert a nested dictionary to an instance of the Config class.

    Args:
        data: Nested dictionary.

    Returns:
        Any: Instance of the Config class.

    Example:
        config_instance = convert_dict_to_config(nested_dict)
    """
    config_instance = Config(data)
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(config_instance, key, dict2config(value))
    return config_instance


def print_config(config: Config, indent: int = 0, verbose: bool = True) -> None:
    if verbose:
        for key, value in vars(config).items():
            if isinstance(value, Config):
                print(f"{' ' * indent}{key}:")
                print_config(value, indent + 2)
            else:
                print(f"{' ' * indent}{key}: {value}")


def merge_configs(args: Any, yaml_config: Config) -> Config:
    """
    Merge command-line arguments and YAML config, preserving existing values.

    Args:
        args: Command-line arguments parsed by argparse.
        yaml_config: Configuration loaded from a YAML file.

    Returns:
        Config: Merged configuration.

    Example:
        merged_config = merge_configs(parsed_args, yaml_config)
    """
    args_dict = vars(args)
    yaml_dict = vars(yaml_config)

    merged_values = args_dict.copy()

    for key, value in yaml_dict.items():
        if key not in merged_values:
            merged_values[key] = value

    merged_config = dict2config(merged_values)
    return merged_config


def read_config_file(config_path: str) -> Union[Config, Exception]:
    """
    Read the contents of a configuration file.

    Parameters:
    - config_path: The path to the configuration file (yaml).

    Returns:
    - Optional: The content of the configuration file, or None if there is an error.
    """
    try:
        if os.path.exists(config_path):
            logging.debug(f"Reading config file: {config_path}")
            with open(config_path, "r") as file:
                config_content = yaml.safe_load(file)

            return Config(config_content)
        else:
            logging.error(f"Error: The specified path '{config_path}' does not exist.")
            return None
    except Exception as e:
        logging.error(f"Error: An exception occurred - {e}")
        return e


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def getTokenizer(type: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast | None:
    if type.lower() == "bert-cased":
        return AutoTokenizer.from_pretrained("bert-base-cased")
    else:
        raise NotImplementedError


def get_model(
    type: str, n_labels: int, model_path: str = None, device: torch.device = None
):
    if type.lower() == "bert-cased":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=n_labels
        )

        if model_path:
            # load the checkpoint
            model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        raise NotImplementedError
