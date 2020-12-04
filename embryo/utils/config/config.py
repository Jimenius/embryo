'''
'''


import os
from typing import Any, Dict
import yaml


def merge_config(
    cfg: Dict[str, Any],
    default: Dict[str, Any],
) -> None:
    '''Merge default configurations into user configurations in-place.
    
    Args:
    cfg:
        User configurations
    default:
        Default configurations
    '''

    for k in default:
        if isinstance(default[k], dict):
            merge_config(cfg.setdefault(k, {}), default[k]) # Recursively set sub-dictionaries
        else:
            cfg.setdefault(k, default[k])


def parse_config(
    config_file_path: str = ''
) -> Dict[str, Any]:
    '''Load user configurations and merge them with default

    Args:
    config_file_path
        Path to the user configuration YAML file
    
    Returns:
        Combined configurations
    '''

    try:
        with open(config_file_path) as fp:
            cfg = yaml.load(config_file_path)
    except:
        raise Exception
    default_config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    with open(default_config_path) as fp:
        default_config = yaml.load(fp)

    merge_config(cfg=cfg, default=default_config)
    return cfg