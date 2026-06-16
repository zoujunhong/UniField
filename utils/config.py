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


def _apply_dataset_kwargs(dataset_config: dict[str, Any], values: dict[str, Any]) -> None:
    if not values:
        return
    if not isinstance(values, dict):
        raise TypeError("dataset_kwargs must be a dict.")

    names = dataset_config.get("name")
    values = copy.deepcopy(values)
    if isinstance(names, str):
        kwargs = dict(dataset_config.get("kwargs", {}))
        common_values = values.pop("common_kwargs", {})
        if common_values:
            if not isinstance(common_values, dict):
                raise TypeError("dataset_kwargs['common_kwargs'] must be a dict.")
            kwargs.update(common_values)
        nested_values = values.pop("kwargs", {})
        if nested_values:
            if not isinstance(nested_values, dict):
                raise TypeError("dataset_kwargs['kwargs'] must be a dict.")
            if names in nested_values and isinstance(nested_values[names], dict):
                kwargs.update(nested_values[names])
            else:
                kwargs.update(nested_values)
        dataset_values = values.pop(names, {})
        if dataset_values:
            if not isinstance(dataset_values, dict):
                raise TypeError(f"dataset_kwargs[{names!r}] must be a dict.")
            kwargs.update(dataset_values)
        kwargs.update(values)
        dataset_config["kwargs"] = kwargs
        return

    if isinstance(names, (list, tuple)):
        dataset_names = set(names)
        common_patch = {}
        specific_patch = {}
        explicit_common = values.pop("common_kwargs", {})
        if explicit_common:
            if not isinstance(explicit_common, dict):
                raise TypeError("dataset_kwargs['common_kwargs'] must be a dict.")
            common_patch.update(explicit_common)
        explicit_kwargs = values.pop("kwargs", {})
        if explicit_kwargs:
            if not isinstance(explicit_kwargs, dict):
                raise TypeError("dataset_kwargs['kwargs'] must be a dict.")
            values.update(explicit_kwargs)
        for key, value in values.items():
            if key in dataset_names and isinstance(value, dict):
                specific_patch[key] = value
            else:
                common_patch[key] = value

        if common_patch:
            common_kwargs = dict(dataset_config.get("common_kwargs", {}))
            common_kwargs.update(common_patch)
            dataset_config["common_kwargs"] = common_kwargs
        if specific_patch:
            kwargs = copy.deepcopy(dataset_config.get("kwargs", {}))
            for name, patch in specific_patch.items():
                dataset_kwargs = dict(kwargs.get(name, {}))
                dataset_kwargs.update(patch)
                kwargs[name] = dataset_kwargs
            dataset_config["kwargs"] = kwargs
        return

    raise TypeError("dataset config 'name' must be a string or a list of strings.")


def dataset_config_for_phase(
    config: dict[str, Any],
    phase: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_config = copy.deepcopy(config["dataset"])
    top_level_values = config.get("dataset_kwargs", {})
    if top_level_values:
        if not isinstance(top_level_values, dict):
            raise TypeError("Top-level dataset_kwargs must be a dict.")
        if phase in top_level_values:
            _apply_dataset_kwargs(dataset_config, top_level_values[phase])
        elif {"train", "test", "visualization"}.isdisjoint(top_level_values):
            _apply_dataset_kwargs(dataset_config, top_level_values)

    phase_config = config.get(phase, {})
    if phase_config:
        if not isinstance(phase_config, dict):
            raise TypeError(f"Config section {phase!r} must be a dict.")
        _apply_dataset_kwargs(dataset_config, phase_config.get("dataset_kwargs", {}))
    if extra_kwargs:
        _apply_dataset_kwargs(dataset_config, extra_kwargs)
    return dataset_config


def require_keys(config: dict[str, Any], keys: tuple[str, ...] | list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
