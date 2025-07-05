import os
from typing import Any, Dict, Optional
import yaml

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    path : str | None
        Path to the configuration file. If ``None``, loads ``config.yaml``
        located in the same directory as this module.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path is None:
        _CONFIG_CACHE = cfg

    return cfg
