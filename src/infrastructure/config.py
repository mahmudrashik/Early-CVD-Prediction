from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from infrastructure.validation import validate_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    
    # Validate configuration parameters
    validate_config(config)
    
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    for key in ["artifacts_dir", "reports_dir", "figures_dir", "tables_dir", "manuscript_dir", "mlruns_dir"]:
        value = config.get("paths", {}).get(key)
        if value:
            resolve_path(value).mkdir(parents=True, exist_ok=True)

