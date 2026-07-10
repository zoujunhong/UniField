from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .basic import masked_mean, pointwise_abs_loss


MAIN_FIELD_CHANNELS = 7
GRAD_U8_SLICE = slice(7, 15)
HESS_U18_SLICE = slice(15, 33)
GRAD_P3_SLICE = slice(33, 36)
JET_OUT_CHANNELS = 36


def _high_speed_refs(cond: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cond is None:
        raise ValueError("HighSpeedTrain physical jet loss needs batch['cond'].")
    cond = cond.to(device=values.device, dtype=values.dtype)
    ux = cond[:, 0] * 100.0
    uy = cond[:, 1] * 20.0
    speed = torch.sqrt(ux.square() + uy.square()).clamp_min(1e-12).view(-1, 1, 1)
    dynamic_pressure = 0.5 * speed.square()
    return speed, dynamic_pressure


def _denormalize_field(
    values: torch.Tensor,
    field: str,
    task_config: dict[str, Any] | None,
    batch: dict[str, torch.Tensor | None] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = dict(((task_config or {}).get("denormalize") or {}).get(field, {"type": "none"}))
    spec_type = str(spec.get("type", "none")).lower()
    if spec_type in {"none", "raw"}:
        return values, torch.ones_like(values)
    if spec_type == "affine":
        mean = values.new_tensor(spec["mean"]).view(*([1] * (values.ndim - 1)), -1)
        std = values.new_tensor(spec["std"]).view(*([1] * (values.ndim - 1)), -1).abs().clamp_min(1e-12)
        return values * std + mean, std.expand_as(values)
    if spec_type == "physical":
        dataset = str(spec.get("dataset", "HighSpeedTrain"))
        if dataset not in {"HighSpeedTrain", "CRH450", "Maglev"}:
            raise ValueError(f"Jet physics loss only supports HighSpeedTrain physical denormalization, got {dataset}.")
        cond = None if batch is None else batch.get("cond")
        speed, dynamic_pressure = _high_speed_refs(cond, values)
        if field == "U":
            scale = speed
        elif field == "p":
            scale = dynamic_pressure
        elif field == "k":
            scale = 0.01 * speed.square()
        elif field == "omega":
            scale = 100.0 * speed
        elif field == "nut":
            scale = 0.01 * speed
        else:
            raise ValueError(f"Unsupported HighSpeedTrain physical field for jet loss: {field}")
        return values * scale, scale.expand_as(values)
    raise ValueError(f"Unsupported denormalize type for jet loss: {spec}")


def _require_volume_fields(volume_slices: dict[str, slice], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in volume_slices]
    if missing:
        raise KeyError(f"Jet physics loss needs volume targets {fields}, missing {missing}.")


def _raw_main_fields(
    volume_pred: torch.Tensor,
    volume_slices: dict[str, slice],
    task_config: dict[str, Any] | None,
    batch: dict[str, torch.Tensor | None] | None,
) -> dict[str, torch.Tensor]:
    _require_volume_fields(volume_slices, ("U", "p", "k", "omega", "nut"))
    fields = {}
    for field in ("U", "p", "k", "omega", "nut"):
        raw, scale = _denormalize_field(volume_pred[..., volume_slices[field]], field, task_config, batch)
        fields[field] = raw
        fields[f"{field}_scale"] = scale
    return fields


def _check_jet_channels(volume_pred: torch.Tensor) -> None:
    if volume_pred.shape[-1] < JET_OUT_CHANNELS:
        raise ValueError(
            f"Jet physics losses require at least {JET_OUT_CHANNELS} prediction channels, "
            f"got {volume_pred.shape[-1]}."
        )


def _jet_scale_refs(
    volume_pred: torch.Tensor,
    batch: dict[str, torch.Tensor | None] | None,
    item: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    length_scale = float(item.get("length_scale", 1.0))
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive.")
    cond = None if batch is None else batch.get("cond")
    speed, dynamic_pressure = _high_speed_refs(cond, volume_pred)
    return speed, dynamic_pressure, length_scale


def _parse_jets(
    volume_pred: torch.Tensor,
    batch: dict[str, torch.Tensor | None] | None,
    item: dict[str, Any],
) -> dict[str, torch.Tensor]:
    _check_jet_channels(volume_pred)
    speed, dynamic_pressure, length_scale = _jet_scale_refs(volume_pred, batch, item)

    grad_u8 = volume_pred[..., GRAD_U8_SLICE] * (speed / length_scale)
    hess_u18 = volume_pred[..., HESS_U18_SLICE] * (speed / (length_scale * length_scale))
    grad_p = volume_pred[..., GRAD_P3_SLICE] * (dynamic_pressure / length_scale)

    grad_u = volume_pred.new_zeros(*volume_pred.shape[:-1], 3, 3)
    grad_u[..., 0, 0] = grad_u8[..., 0]
    grad_u[..., 0, 1] = grad_u8[..., 1]
    grad_u[..., 0, 2] = grad_u8[..., 2]
    grad_u[..., 1, 0] = grad_u8[..., 3]
    grad_u[..., 1, 1] = grad_u8[..., 4]
    grad_u[..., 1, 2] = grad_u8[..., 5]
    grad_u[..., 2, 0] = grad_u8[..., 6]
    grad_u[..., 2, 1] = grad_u8[..., 7]
    grad_u[..., 2, 2] = -(grad_u[..., 0, 0] + grad_u[..., 1, 1])

    hess_u = hess_u18.view(*volume_pred.shape[:-1], 3, 6)
    lap_u = hess_u[..., 0] + hess_u[..., 1] + hess_u[..., 2]
    return {
        "grad_u": grad_u,
        "hess_u": hess_u,
        "lap_u": lap_u,
        "grad_p": grad_p,
        "speed": speed,
        "dynamic_pressure": dynamic_pressure,
    }


def _gather_points(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (values.ndim - 2)
    expand_shape = [indices.shape[0], indices.shape[1], *values.shape[2:]]
    return values.gather(1, indices.view(*view_shape).expand(*expand_shape))


def _sample_nearest_pairs(query: torch.Tensor, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if query.ndim != 3 or query.shape[-1] < 3:
        raise ValueError("Taylor jet loss needs batch['volume_query'] with at least xyz channels.")
    query = query[..., :3]
    batch, count, _ = query.shape
    if count < 2:
        empty_idx = torch.empty(batch, 0, dtype=torch.long, device=query.device)
        empty_dx = query.new_empty(batch, 0, 3)
        empty_mask = torch.empty(batch, 0, dtype=torch.bool, device=query.device)
        return empty_idx, empty_idx, empty_dx, empty_mask

    sample_size = int(item.get("pair_sample_size", min(count, 1024)))
    sample_size = max(1, min(sample_size, count))
    candidate_size = int(item.get("pair_candidate_size", count))
    candidate_size = max(2, min(candidate_size, count))

    if sample_size == count:
        anchor_idx = torch.arange(count, device=query.device).view(1, -1).expand(batch, -1)
    else:
        anchor_idx = torch.randint(0, count, (batch, sample_size), device=query.device)

    if candidate_size == count:
        candidate_idx = torch.arange(count, device=query.device).view(1, -1).expand(batch, -1)
    else:
        candidate_idx = torch.randint(0, count, (batch, candidate_size), device=query.device)

    anchors = _gather_points(query, anchor_idx).float()
    candidates = _gather_points(query, candidate_idx).float()
    distances = torch.cdist(anchors, candidates, p=2)
    distances = distances.masked_fill(candidate_idx.unsqueeze(1) == anchor_idx.unsqueeze(2), float("inf"))
    nearest_pos = distances.argmin(dim=-1)
    nearest_dist = distances.gather(-1, nearest_pos.unsqueeze(-1)).squeeze(-1)
    neighbor_idx = candidate_idx.gather(1, nearest_pos)
    dx = _gather_points(query, neighbor_idx) - _gather_points(query, anchor_idx)

    min_distance = float(item.get("min_pair_distance", 1e-8))
    mask = torch.isfinite(nearest_dist) & (nearest_dist > min_distance)
    max_distance = item.get("max_pair_distance")
    if max_distance is not None:
        mask = mask & (nearest_dist <= float(max_distance))
    return anchor_idx, neighbor_idx, dx, mask


def _hessian_quadratic(hess_u: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    dx2 = dx.square()
    cross_xy = dx[..., 0] * dx[..., 1]
    cross_xz = dx[..., 0] * dx[..., 2]
    cross_yz = dx[..., 1] * dx[..., 2]
    return (
        hess_u[..., 0] * dx2[..., 0:1]
        + hess_u[..., 1] * dx2[..., 1:2]
        + hess_u[..., 2] * dx2[..., 2:3]
        + 2.0 * hess_u[..., 3] * cross_xy.unsqueeze(-1)
        + 2.0 * hess_u[..., 4] * cross_xz.unsqueeze(-1)
        + 2.0 * hess_u[..., 5] * cross_yz.unsqueeze(-1)
    )


def _taylor_direction_loss(
    fields: dict[str, torch.Tensor],
    jets: dict[str, torch.Tensor],
    anchor_idx: torch.Tensor,
    neighbor_idx: torch.Tensor,
    dx: torch.Tensor,
    mask: torch.Tensor,
    item: dict[str, Any],
) -> torch.Tensor:
    u_i = _gather_points(fields["U"], anchor_idx)
    u_j = _gather_points(fields["U"], neighbor_idx)
    p_i = _gather_points(fields["p"], anchor_idx)
    p_j = _gather_points(fields["p"], neighbor_idx)
    grad_u_i = _gather_points(jets["grad_u"], anchor_idx)
    hess_u_i = _gather_points(jets["hess_u"], anchor_idx)
    grad_p_i = _gather_points(jets["grad_p"], anchor_idx)

    if bool(item.get("detach_field_values", False)):
        u_i = u_i.detach()
        u_j = u_j.detach()
        p_i = p_i.detach()
        p_j = p_j.detach()

    u_linear = torch.einsum("bsij,bsj->bsi", grad_u_i, dx)
    u_quadratic = 0.5 * _hessian_quadratic(hess_u_i, dx)
    p_linear = torch.einsum("bsj,bsj->bs", grad_p_i, dx).unsqueeze(-1)

    u_ref = _gather_points(jets["speed"], torch.zeros_like(anchor_idx)).clamp_min(1e-12)
    p_ref = _gather_points(jets["dynamic_pressure"], torch.zeros_like(anchor_idx)).clamp_min(1e-12)
    u_residual = (u_i + u_linear + u_quadratic - u_j) / u_ref
    p_residual = (p_i + p_linear - p_j) / p_ref
    max_residual = item.get("max_residual", item.get("max_taylor_residual"))
    if max_residual is not None:
        max_residual = float(max_residual)
        u_residual = u_residual.clamp(min=-max_residual, max=max_residual)
        p_residual = p_residual.clamp(min=-max_residual, max=max_residual)

    loss_type = str(item.get("loss_type", "smooth_l1"))
    beta = float(item.get("beta", 0.02))
    u_loss = masked_mean(pointwise_abs_loss(u_residual, loss_type, beta), mask.unsqueeze(-1))
    p_loss = masked_mean(pointwise_abs_loss(p_residual, loss_type, beta), mask.unsqueeze(-1))
    return float(item.get("u_weight", 1.0)) * u_loss + float(item.get("p_weight", 1.0)) * p_loss


def jet_taylor_up_loss(
    volume_pred: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
    *,
    batch: dict[str, torch.Tensor | None] | None,
    task_config: dict[str, Any] | None,
) -> torch.Tensor:
    if batch is None or batch.get("volume_query") is None:
        raise ValueError("Taylor jet loss needs batch['volume_query'].")
    volume_pred = volume_pred.float()
    fields = _raw_main_fields(volume_pred, volume_slices, task_config, batch)
    jets = _parse_jets(volume_pred, batch, item)
    anchor_idx, neighbor_idx, dx, mask = _sample_nearest_pairs(batch["volume_query"], item)
    if anchor_idx.numel() == 0 or not bool(mask.any()):
        return volume_pred.new_zeros(())

    value = _taylor_direction_loss(fields, jets, anchor_idx, neighbor_idx, dx, mask, item)
    if bool(item.get("symmetric", True)):
        reverse = _taylor_direction_loss(fields, jets, neighbor_idx, anchor_idx, -dx, mask, item)
        value = 0.5 * (value + reverse)
    return value


def local_constant_nu_momentum_ratio_loss(
    volume_pred: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
    *,
    batch: dict[str, torch.Tensor | None] | None,
    task_config: dict[str, Any] | None,
) -> torch.Tensor:
    volume_pred = volume_pred.float()
    fields = _raw_main_fields(volume_pred, volume_slices, task_config, batch)
    jets = _parse_jets(volume_pred, batch, item)
    u = fields["U"]
    nut = fields["nut"]
    if bool(item.get("clamp_nut", True)):
        nut = nut.clamp_min(0.0)
    nu_eff = nut + float(item.get("nu_lam", 1.5e-5))

    convective = torch.einsum("bnj,bnij->bni", u, jets["grad_u"])
    velocity_term = convective - nu_eff * jets["lap_u"]
    pressure_term = jets["grad_p"]
    residual = velocity_term + pressure_term

    length_scale = float(item.get("length_scale", 1.0))
    eps_factor = float(item.get("pressure_grad_eps_factor", 1e-3))
    eps_abs = float(item.get("pressure_grad_eps", 0.0))
    pressure_floor = eps_factor * jets["dynamic_pressure"] / max(length_scale, 1e-12)
    denominator = pressure_term.detach().abs().clamp_min(pressure_floor + eps_abs)
    ratio = residual / denominator
    max_ratio = item.get("max_ratio")
    if max_ratio is not None:
        ratio = ratio.clamp(min=-float(max_ratio), max=float(max_ratio))

    loss_type = str(item.get("loss_type", "smooth_l1"))
    beta = float(item.get("beta", 0.1))
    return pointwise_abs_loss(ratio, loss_type, beta).mean()


def rans_nonnegative_loss(
    volume_pred: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
    *,
    batch: dict[str, torch.Tensor | None] | None,
    task_config: dict[str, Any] | None,
) -> torch.Tensor:
    volume_pred = volume_pred.float()
    fields = _raw_main_fields(volume_pred, volume_slices, task_config, batch)
    value = volume_pred.new_zeros(())
    for field in item.get("fields", ("k", "omega", "nut")):
        raw = fields[field]
        scale = fields[f"{field}_scale"].abs().clamp_min(1e-12)
        value = value + F.relu(-raw / scale).mean()
    return value
