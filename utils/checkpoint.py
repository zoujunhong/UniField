from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.module.", "")
        key = key.replace("_orig_mod.", "")
        key = key.replace("module.", "")
        cleaned[key] = value
    return cleaned


def _partial_leading_dim_load(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor | None:
    if src.ndim != dst.ndim or src.ndim == 0:
        return None
    if src.shape[0] > dst.shape[0] or tuple(src.shape[1:]) != tuple(dst.shape[1:]):
        return None
    merged = dst.clone()
    merged[: src.shape[0]] = src.to(device=dst.device, dtype=dst.dtype)
    return merged


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    route_map: Sequence[int] | None = None,
    strict: bool = False,
    partial_load: bool = False,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    target = unwrap_model(model)
    if hasattr(target, "load_route_compatible_state_dict"):
        report = target.load_route_compatible_state_dict(
            checkpoint,
            route_map=route_map,
            strict=strict,
            partial_load=partial_load,
        )
        print(
            "Route-compatible load: "
            f"{len(report['loaded'])} exact, "
            f"{len(report.get('legacy_loaded', []))} legacy-remapped, "
            f"{len(report['route_loaded'])} route-adapted, "
            f"{len(report.get('partial_loaded', []))} partial, "
            f"{len(report['skipped'])} skipped",
            flush=True,
        )
        for item in report.get("partial_loaded", []):
            print(f"Partially loaded weight: {item}", flush=True)
        for item in report["skipped"]:
            print(f"Skipping weight: {item}", flush=True)
        return model

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)
    model_state = target.state_dict()
    filtered = {}
    for name, param in state_dict.items():
        if name not in model_state:
            if not strict:
                print(f"Skipping weight: {name} as it is not in the model", flush=True)
                continue
            raise KeyError(f"Checkpoint key not found in model: {name}")
        if model_state[name].shape != param.shape:
            if partial_load:
                merged = _partial_leading_dim_load(model_state[name], param)
                if merged is not None:
                    filtered[name] = merged
                    print(
                        f"Partially loaded weight: {name}: leading slice {tuple(param.shape)} -> "
                        f"{tuple(model_state[name].shape)}",
                        flush=True,
                    )
                    continue
            if not strict:
                print(f"Skipping weight: {name} due to shape mismatch ({param.shape} vs {model_state[name].shape})", flush=True)
                continue
            raise RuntimeError(f"Shape mismatch for {name}: {param.shape} vs {model_state[name].shape}")
        filtered[name] = param
    model_state.update(filtered)
    target.load_state_dict(model_state, strict=False)
    return model


def resolve_eval_checkpoint(config: dict, phase: str) -> str:
    phase_config = config.get(phase, {})
    if phase_config.get("checkpoint") is None:
        return ""
    checkpoint = str(phase_config.get("checkpoint", ""))
    if checkpoint:
        return checkpoint
    train_config = config.get("train", {})
    saveroot = train_config.get("saveroot")
    total_epoch = train_config.get("total_epoch")
    if not saveroot or total_epoch is None:
        return ""
    return str(Path(saveroot) / f"model_epoch_{int(total_epoch)}.pth")


def save_checkpoint(model: torch.nn.Module, save_dir: str | Path, epoch: int, filename: str | None = None) -> Path:
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or f"model_epoch_{int(epoch)}.pth"
    path = save_dir / filename
    torch.save(unwrap_model(model).state_dict(), path)
    return path
