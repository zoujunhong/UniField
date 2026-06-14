from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


FIELD_DIMS = {"p": 1, "U": 3, "k": 1, "omega": 1, "nut": 1}


def split_surface_input(surface_input: torch.Tensor, surface_input_list: list[str]):
    fields = ["xyz", *[field for field in surface_input_list if field != "xyz"]]
    offset = 0
    surface_pos = None
    surface_normals = None
    surface_area = None
    for field in fields:
        if field == "xyz":
            surface_pos = surface_input[..., offset : offset + 3]
            offset += 3
        elif field in {"normal", "normals"}:
            surface_normals = surface_input[..., offset : offset + 3]
            offset += 3
        elif field in {"area", "log_area", "p", "k", "omega", "nut"}:
            value = surface_input[..., offset : offset + 1]
            if field == "area":
                surface_area = value
            offset += 1
        elif field == "U":
            offset += 3
        else:
            raise ValueError(f"Unsupported surface input field: {field}")
    if surface_pos is None:
        raise ValueError("surface_input_list must include xyz.")
    return surface_pos, surface_normals, surface_area


def field_slices(fields: list[str]) -> dict[str, slice]:
    offset = 0
    slices = {}
    for field in fields:
        if field not in FIELD_DIMS:
            raise ValueError(f"Unsupported target field: {field}")
        dim = FIELD_DIMS[field]
        slices[field] = slice(offset, offset + dim)
        offset += dim
    return slices


def build_query(surface_query: torch.Tensor, volume_query: torch.Tensor, add_type_channel: bool = True) -> torch.Tensor:
    if not add_type_channel:
        return torch.cat([surface_query, volume_query], dim=1)
    surface_type = surface_query.new_zeros(surface_query.shape[0], surface_query.shape[1], 1)
    volume_type = volume_query.new_ones(volume_query.shape[0], volume_query.shape[1], 1)
    return torch.cat(
        [
            torch.cat([surface_query, surface_type], dim=-1),
            torch.cat([volume_query, volume_type], dim=-1),
        ],
        dim=1,
    )


def move_batch_to_device(batch, device: int | torch.device, surface_input_list: list[str]) -> dict[str, torch.Tensor | None]:
    surface_input, surface_query, surface_target, volume_query, volume_target, cond, routes = batch
    surface_input = surface_input.float().to(device, non_blocking=True)
    surface_pos, normals, area = split_surface_input(surface_input, surface_input_list)
    return {
        "surface_input": surface_input,
        "surface_pos": surface_pos,
        "normals": None if normals is None else normals.float().to(device, non_blocking=True),
        "area": None if area is None else area.float().to(device, non_blocking=True),
        "surface_query": surface_query.float().to(device, non_blocking=True),
        "surface_target": surface_target.float().to(device, non_blocking=True),
        "volume_query": volume_query.float().to(device, non_blocking=True),
        "volume_target": volume_target.float().to(device, non_blocking=True),
        "cond": cond.float().to(device, non_blocking=True),
        "routes": routes.long().to(device, non_blocking=True),
    }


def compute_query_loss(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor | None],
    task_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    surface_count = int(batch["surface_query"].shape[1])
    surface_pred = pred[:, :surface_count]
    volume_pred = pred[:, surface_count:]
    surface_target = batch["surface_target"]
    volume_target = batch["volume_target"]
    surface_slices = field_slices(task_config.get("surface_target_list", []))
    volume_slices = field_slices(task_config.get("volume_target_list", []))
    total = pred.new_zeros(())
    losses: dict[str, torch.Tensor] = {}

    for item in task_config.get("losses", []):
        domain = item["domain"]
        field = item["field"]
        pred_slice = slice(*item["pred_slice"])
        weight = float(item.get("weight", 1.0))
        if domain == "surface":
            if surface_count == 0:
                continue
            target_slice = surface_slices[field]
            value = F.l1_loss(surface_pred[..., pred_slice], surface_target[..., target_slice])
        elif domain == "volume":
            if volume_pred.shape[1] == 0:
                continue
            target_slice = volume_slices[field]
            value = F.l1_loss(volume_pred[..., pred_slice], volume_target[..., target_slice])
        else:
            raise ValueError(f"Unsupported loss domain: {domain}")
        losses[f"{domain}_{field}"] = value
        total = total + weight * value
    losses["total"] = total
    return total, losses


def prediction_field(pred: torch.Tensor, domain: str, field: str, task_config: dict[str, Any], surface_count: int) -> torch.Tensor:
    items = [item for item in task_config.get("losses", []) if item["domain"] == domain and item["field"] == field]
    if not items:
        raise KeyError(f"No prediction slice configured for {domain}:{field}")
    pred_slice = slice(*items[0]["pred_slice"])
    if domain == "surface":
        return pred[:, :surface_count, pred_slice]
    if domain == "volume":
        return pred[:, surface_count:, pred_slice]
    raise ValueError(f"Unsupported domain: {domain}")


def target_field(target: torch.Tensor, fields: list[str], field: str) -> torch.Tensor:
    return target[..., field_slices(fields)[field]]
