from __future__ import annotations

import importlib
from typing import Any

from torch.utils.data import ConcatDataset


def _build_one_dataset(name: str, kwargs: dict[str, Any]):
    module = importlib.import_module(f"dataset.{name}")
    if not hasattr(module, "Dataset"):
        raise AttributeError(f"dataset/{name}.py must export Dataset.")
    return module.Dataset(**kwargs)


def _kwargs_for_dataset(name: str, config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config.get("common_kwargs", {}))
    specific = config.get("kwargs", {})
    if isinstance(specific, dict) and name in specific and isinstance(specific[name], dict):
        kwargs.update(specific[name])
    elif isinstance(specific, dict):
        kwargs.update(specific)
    else:
        raise TypeError("dataset config 'kwargs' must be a dict.")
    return kwargs


def _check_concat_compatibility(datasets: list) -> None:
    signatures = [getattr(dataset, "interface_signature", None) for dataset in datasets]
    if any(signature is None for signature in signatures):
        raise AttributeError("All merged datasets must expose interface_signature.")
    first = signatures[0]
    for idx, signature in enumerate(signatures[1:], start=1):
        if signature != first:
            raise ValueError(
                "Merged datasets must have identical interface signatures. "
                f"dataset[0]={first}, dataset[{idx}]={signature}"
            )


def build_dataset(config: dict[str, Any]):
    names = config.get("name")
    if names is None:
        raise KeyError("dataset config must contain 'name'.")
    if isinstance(names, str):
        return _build_one_dataset(names, _kwargs_for_dataset(names, config))
    if isinstance(names, (list, tuple)):
        datasets = [_build_one_dataset(name, _kwargs_for_dataset(name, config)) for name in names]
        _check_concat_compatibility(datasets)
        return ConcatDataset(datasets)
    raise TypeError("dataset config 'name' must be a string or a list of strings.")


__all__ = ["build_dataset", "ConcatDataset"]
