"""Central project configuration loaded from config.yaml.

Usage:
    from config import load_config, get_bbox

    cfg = load_config()
    bbox = get_bbox(cfg)           # {"north": ..., "south": ..., "east": ..., "west": ...}
    rf   = cfg["rf"]               # freq_mhz, tx_height_m, ...
    paths = cfg["paths"]           # pop_file, hifld_csv, ...
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load project configuration from a YAML file.

    Args:
        path: Path to YAML file. Defaults to config.yaml at the project root.

    Returns:
        Nested dict with configuration values.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_bbox(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return the region bounding box as a flat dict.

    Args:
        cfg: Pre-loaded config dict. Loaded from default path if None.

    Returns:
        Dict with ``north``, ``south``, ``east``, ``west`` float keys.
    """
    return dict((cfg or load_config())["region"])
