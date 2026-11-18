from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from torch.utils.data import ConcatDataset


def _build_one_dataset(name: str, kwargs: dict[str, Any]):
    module = importlib.import_module(f"dataset.{name}")
    if not hasattr(module, "Dataset"):
        raise AttributeError(f"dataset/{name}.py must export Dataset.")
    kwargs = dict(kwargs)
    if name == "DrivAerNet" and "split" in kwargs:
        split = kwargs.pop("split")
        if split not in {"all", "train", "val", "test"}:
            raise ValueError(f"Unsupported DrivAerNet split: {split!r}")
        if split != "all":
            kwargs.setdefault("ids_file", str(Path(__file__).parent / "splits/DrivAerNet" / f"{split}_design_ids.txt"))
    validation = bool(kwargs.pop("validation", False))
    if validation:
        from dataset.validation import prepare_validation_kwargs
        kwargs = prepare_validation_kwargs(name, kwargs)
    kwargs.pop("unsupervised_surface_context", None)
    result = module.Dataset(**kwargs)
    if validation:
        from dataset.validation import configure_validation
        configure_validation(result)
    return result


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


def selected_dataset_names(config: dict[str, Any]) -> list[str]:
    """Resolve a cross-domain config's selected domain groups to dataset names."""

    names = config.get("name")
    if isinstance(names, str):
        declared_names = [names]
    elif isinstance(names, (list, tuple)) and all(isinstance(name, str) for name in names):
        declared_names = list(names)
    else:
        raise TypeError("dataset config 'name' must be a string or a list of strings.")

    groups = config.get("domain_groups")
    selected = config.get("selected_domains")
    if groups is None:
        if selected is not None:
            raise KeyError("selected_domains requires a dataset config with domain_groups.")
        return declared_names
    if not isinstance(groups, dict) or not groups:
        raise TypeError("domain_groups must be a non-empty mapping from domain names to dataset-name lists.")

    if selected is None:
        selected_names = list(groups)
    elif isinstance(selected, str):
        selected_names = [selected]
    elif isinstance(selected, (list, tuple)) and all(isinstance(name, str) for name in selected):
        selected_names = list(selected)
    else:
        raise TypeError("selected_domains must be a domain name or a list of domain names.")
    if not selected_names:
        raise ValueError("selected_domains must contain at least one domain.")

    resolved: list[str] = []
    for domain_name in selected_names:
        if domain_name not in groups:
            raise KeyError(f"Unknown domain {domain_name!r}; available domains: {sorted(groups)}.")
        group_names = groups[domain_name]
        if isinstance(group_names, str):
            group_names = [group_names]
        if not isinstance(group_names, (list, tuple)) or not all(isinstance(name, str) for name in group_names):
            raise TypeError(f"domain_groups[{domain_name!r}] must be a dataset-name list.")
        for name in group_names:
            if name not in declared_names:
                raise KeyError(f"Domain {domain_name!r} refers to undeclared dataset {name!r}.")
            if name not in resolved:
                resolved.append(name)
    return resolved


def build_dataset(config: dict[str, Any]):
    names = selected_dataset_names(config)
    if len(names) == 1:
        return _build_one_dataset(names[0], _kwargs_for_dataset(names[0], config))
    if len(names) > 1:
        datasets = [_build_one_dataset(name, _kwargs_for_dataset(name, config)) for name in names]
        _check_concat_compatibility(datasets)
        if bool(config.get("drop_empty", False)):
            nonempty_datasets = [dataset for dataset in datasets if len(dataset) > 0]
            if len(nonempty_datasets) < len(datasets):
                print(
                    f"Merged dataset: skipped {len(datasets) - len(nonempty_datasets)} empty dataset(s).",
                    flush=True,
                )
            if not nonempty_datasets:
                raise RuntimeError("Merged dataset has no samples after filtering incomplete or empty datasets.")
            datasets = nonempty_datasets
        return ConcatDataset(datasets)
    raise RuntimeError("No datasets were selected.")


__all__ = ["build_dataset", "selected_dataset_names", "ConcatDataset"]
