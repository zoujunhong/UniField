from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn


def clean_checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod.") :]
        cleaned[clean_key] = value
    return cleaned


_LEGACY_ENCODER_PREFIXES = (
    "proj_in.",
    "proj_down_layers.",
    "proj_up_layers.",
    "tf_down_layers.",
    "tf_up_layers.",
    "downsample_layers.",
    "adapt_down_layers.",
    "adapt_up_layers.",
    "adapt_down_proj_down_layers.",
    "adapt_down_proj_up_layers.",
    "adapt_up_proj_down_layers.",
    "adapt_up_proj_up_layers.",
)


def _add_legacy_query_aliases(
    source: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    aliased = dict(source)
    alias_to_source: dict[str, str] = {}
    for name, value in source.items():
        if name.startswith(("encoder.", "decoder.")):
            continue
        if name.startswith("proj_out."):
            alias = f"decoder.{name}"
        elif name.startswith(_LEGACY_ENCODER_PREFIXES):
            alias = f"encoder.{name}"
        else:
            continue
        if alias not in aliased:
            aliased[alias] = value
            alias_to_source[alias] = name
    return aliased, alias_to_source


def _legacy_route_tensor_name(target_name: str) -> tuple[str, str] | None:
    match = re.match(r"(.+)branches\.(\d+)\.(.+)$", target_name)
    if match is None:
        return None
    prefix, _, suffix = match.groups()
    legacy_prefix = f"{prefix}norm."
    mapping = {
        "adaLN_modulation.0.weight": "route_fc1_weight",
        "adaLN_modulation.0.bias": "route_fc1_bias",
        "adaLN_modulation.3.weight": "route_fc2_weight",
        "adaLN_modulation.3.bias": "route_fc2_bias",
        "shift": "route_shift",
        "scale": "route_scale",
    }
    if suffix not in mapping:
        return None
    return legacy_prefix + mapping[suffix], suffix


def load_route_compatible_state_dict(
    module: nn.Module,
    checkpoint: Mapping[str, Any],
    route_map: Optional[Sequence[int]] = None,
    strict: bool = False,
) -> dict[str, list[str]]:
    source, alias_to_source = _add_legacy_query_aliases(clean_checkpoint_state_dict(checkpoint))
    target = module.state_dict()
    loaded: list[str] = []
    legacy_loaded: list[str] = []
    route_loaded: list[str] = []
    skipped: list[str] = []
    loaded_names: set[str] = set()

    route_map_t: Optional[torch.Tensor] = None
    if route_map is not None:
        route_map_t = torch.as_tensor(list(route_map), dtype=torch.long)

    for name, src in source.items():
        if name not in target:
            continue
        dst = target[name]
        src = src.to(device=dst.device, dtype=dst.dtype)
        if dst.shape == src.shape:
            target[name] = src
            if name in alias_to_source:
                legacy_loaded.append(f"{alias_to_source[name]} -> {name}")
            else:
                loaded.append(name)
            loaded_names.add(name)
        else:
            skipped.append(f"{name}: shape mismatch {tuple(src.shape)} -> {tuple(dst.shape)}")

    for name, dst in list(target.items()):
        if name in loaded_names:
            continue
        branch_match = re.match(r"(.+branches\.)(\d+)(\..+)$", name)
        if branch_match is None:
            legacy = _legacy_route_tensor_name(name)
            if legacy is None:
                continue
            legacy_name, _ = legacy
            if legacy_name not in source:
                continue
            route_match = re.search(r"branches\.(\d+)\.", name)
            if route_match is None:
                continue
            target_route = int(route_match.group(1))
            src_route = int(route_map_t[target_route].item()) if route_map_t is not None else target_route
            legacy_src = source[legacy_name]
            if legacy_src.ndim == 0 or src_route >= legacy_src.shape[0]:
                continue
            src = legacy_src[src_route].to(device=dst.device, dtype=dst.dtype)
            if src.shape == dst.shape:
                target[name] = src
                route_loaded.append(name)
                loaded_names.add(name)
            continue

        prefix, target_route_text, suffix = branch_match.groups()
        target_route = int(target_route_text)
        src_route = int(route_map_t[target_route].item()) if route_map_t is not None else target_route
        src_name = f"{prefix}{src_route}{suffix}"
        if src_name in source and source[src_name].shape == dst.shape:
            target[name] = source[src_name].to(device=dst.device, dtype=dst.dtype)
            route_loaded.append(name)
            loaded_names.add(name)
            continue

        legacy = _legacy_route_tensor_name(name)
        if legacy is not None:
            legacy_name, _ = legacy
            if legacy_name in source:
                legacy_src = source[legacy_name]
                if legacy_src.ndim > 0 and src_route < legacy_src.shape[0]:
                    src = legacy_src[src_route].to(device=dst.device, dtype=dst.dtype)
                    if src.shape == dst.shape:
                        target[name] = src
                        route_loaded.append(name)
                        loaded_names.add(name)

    if strict and skipped:
        raise RuntimeError("Route-compatible load skipped weights:\n" + "\n".join(skipped))
    module.load_state_dict(target, strict=True)
    return {"loaded": loaded, "legacy_loaded": legacy_loaded, "route_loaded": route_loaded, "skipped": skipped}


__all__ = ["clean_checkpoint_state_dict", "load_route_compatible_state_dict"]
