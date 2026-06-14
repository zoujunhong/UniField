from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    spec = importlib.util.spec_from_file_location(f"_whole_field_config_{config_path.stem}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config file must export CONFIG: {config_path}")
    config = copy.deepcopy(module.CONFIG)
    if not isinstance(config, dict):
        raise TypeError(f"CONFIG must be a dict: {config_path}")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def load_referenced_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return load_config(resolve_path(config[key], config.get("_config_dir")))


def require_keys(config: dict[str, Any], keys: tuple[str, ...] | list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
