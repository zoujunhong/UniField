from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .basic import masked_mean, pointwise_abs_loss
from .jet_physics import (
    jet_taylor_up_loss,
    local_constant_nu_momentum_ratio_loss,
    rans_nonnegative_loss,
)


def _require_k_omega_nut(volume_slices: dict[str, slice]) -> None:
    required = {"k", "omega", "nut"}
    missing = [field for field in required if field not in volume_slices]
    if missing:
        raise KeyError(f"k-omega-nut consistency needs volume targets {sorted(required)}, missing {missing}.")


def _k_omega_nut_fields(
    volume: torch.Tensor,
    volume_slices: dict[str, slice],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        volume[..., volume_slices["k"]],
        volume[..., volume_slices["omega"]],
        volume[..., volume_slices["nut"]],
    )


def _valid_target_mask(
    target_k: torch.Tensor,
    target_omega: torch.Tensor,
    target_nut: torch.Tensor,
    item: dict[str, Any],
) -> torch.Tensor:
    omega_eps = float(item.get("omega_eps", 1e-4))
    return (
        (target_k > float(item.get("min_target_k", 0.0)))
        & (target_omega > float(item.get("min_target_omega", omega_eps)))
        & (target_nut >= float(item.get("min_target_nut", 0.0)))
    )


def _relation_gap_mask(
    target_k: torch.Tensor,
    target_omega: torch.Tensor,
    target_nut: torch.Tensor,
    item: dict[str, Any],
    relation_scale: float = 1.0,
) -> torch.Tensor:
    omega_eps = float(item.get("omega_eps", 1e-4))
    mask_eps = float(item.get("mask_eps", 1e-6))
    tol = float(item.get("equality_tol", item.get("strict_tol", 1e-3)))

    target_nut_from_komega = relation_scale * target_k.clamp_min(0.0) / target_omega.clamp_min(omega_eps)
    denom_mode = str(item.get("mask_denom", "target_nut")).lower()
    if denom_mode in {"target_nut", "nut"}:
        denom = target_nut.abs() + mask_eps
    elif denom_mode in {"symmetric", "both"}:
        denom = target_nut.abs() + target_nut_from_komega.abs() + mask_eps
    elif denom_mode in {"product", "k"}:
        denom = (relation_scale * target_k).abs() + mask_eps
        return (target_omega * target_nut - relation_scale * target_k).abs() / denom <= tol
    else:
        raise ValueError(f"Unsupported SST relation mask_denom: {denom_mode}")
    return (target_nut - target_nut_from_komega).abs() / denom <= tol


def _high_speed_speed(cond: torch.Tensor) -> torch.Tensor:
    ux = cond[:, 0] * 100.0
    uy = cond[:, 1] * 20.0
    return torch.sqrt(ux.square() + uy.square()).clamp_min(1e-12).view(-1, 1, 1)


def _denormalize_field(
    values: torch.Tensor,
    field: str,
    denormalize: dict[str, Any],
    cond: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = denormalize.get(field, {"type": "none"})
    spec_type = str(spec.get("type", "none")).lower()
    if spec_type in {"none", "raw"}:
        return values, torch.ones_like(values[..., :1])
    if spec_type == "affine":
        mean = values.new_tensor(spec["mean"]).view(*([1] * (values.ndim - 1)), -1)
        std = values.new_tensor(spec["std"]).view(*([1] * (values.ndim - 1)), -1).abs().clamp_min(1e-12)
        return values * std + mean, std.expand_as(values)
    if spec_type == "physical":
        dataset = str(spec.get("dataset", "HighSpeedTrain"))
        if dataset not in {"HighSpeedTrain", "CRH450", "Maglev"}:
            raise ValueError(f"SST relation loss only supports HighSpeedTrain physical denormalization, got {dataset}.")
        if cond is None:
            raise ValueError("HighSpeedTrain physical SST relation loss needs batch['cond'].")
        speed = _high_speed_speed(cond).to(device=values.device, dtype=values.dtype)
        if field == "k":
            scale = 0.01 * speed.square()
        elif field == "omega":
            scale = 100.0 * speed
        elif field == "nut":
            scale = 0.01 * speed
        else:
            raise ValueError(f"Unsupported physical SST field: {field}")
        return values * scale, scale.expand_as(values)
    raise ValueError(f"Unsupported denormalize type for SST relation loss: {spec}")


def _raw_k_omega_nut_fields(
    volume: torch.Tensor,
    volume_slices: dict[str, slice],
    denormalize: dict[str, Any],
    cond: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k, omega, nut = _k_omega_nut_fields(volume, volume_slices)
    raw_k, k_scale = _denormalize_field(k, "k", denormalize, cond)
    raw_omega, omega_scale = _denormalize_field(omega, "omega", denormalize, cond)
    raw_nut, nut_scale = _denormalize_field(nut, "nut", denormalize, cond)
    return raw_k, raw_omega, raw_nut, k_scale, omega_scale, nut_scale


def _relation_residual_scale(
    item: dict[str, Any],
    k_scale: torch.Tensor,
    target_k_raw: torch.Tensor,
) -> torch.Tensor:
    mode = str(item.get("residual_normalization", "field_scale")).lower()
    eps = float(item.get("residual_scale_eps", 1e-12))
    if mode in {"field_scale", "k_scale", "normalization_scale"}:
        return k_scale.abs().clamp_min(eps)
    if mode in {"target_k", "relative_target_k"}:
        return target_k_raw.abs().clamp_min(eps)
    if mode in {"none", "raw"}:
        return torch.ones_like(k_scale)
    raise ValueError(f"Unsupported SST residual_normalization: {mode}")


def legacy_k_omega_nut_sst_loss(
    volume_pred: torch.Tensor,
    volume_target: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
) -> torch.Tensor:
    _require_k_omega_nut(volume_slices)
    pred_k, pred_omega, pred_nut = _k_omega_nut_fields(volume_pred, volume_slices)
    target_k, target_omega, target_nut = _k_omega_nut_fields(volume_target, volume_slices)

    omega_eps = float(item.get("omega_eps", 1e-4))
    relation_scale = float(item.get("relation_scale", 0.01))
    max_relation_value = item.get("max_relation_value", 10.0)

    pred_nut_from_komega = relation_scale * pred_k.clamp_min(0.0) / pred_omega.clamp_min(omega_eps)
    target_nut_from_komega = relation_scale * target_k.clamp_min(0.0) / target_omega.clamp_min(omega_eps)
    if max_relation_value is not None:
        pred_nut_from_komega = pred_nut_from_komega.clamp(max=float(max_relation_value))
        target_nut_from_komega = target_nut_from_komega.clamp(max=float(max_relation_value))

    mode = str(item.get("mode", "target_equality_mask")).lower()
    mask = None
    if mode in {"target_equality_mask", "equality_mask", "masked"}:
        tol = float(item.get("equality_tol", 0.2))
        denom = target_nut.abs() + target_nut_from_komega.abs() + float(item.get("mask_eps", 1e-6))
        relation_gap = (target_nut - target_nut_from_komega).abs() / denom
        mask = relation_gap <= tol
        mask = mask & _valid_target_mask(target_k, target_omega, target_nut, item)
    elif mode in {"all", "global"}:
        mask = None
    else:
        raise ValueError(f"Unsupported k-omega-nut consistency mode: {mode}")

    residual = pred_nut - pred_nut_from_komega
    value = masked_mean(
        pointwise_abs_loss(residual, str(item.get("loss_type", "smooth_l1")), float(item.get("beta", 0.02))),
        mask,
    )

    positive_weight = float(item.get("positive_weight", 0.0))
    if positive_weight > 0.0:
        positive = F.relu(-pred_k).mean() + F.relu(-pred_omega).mean() + F.relu(-pred_nut).mean()
        value = value + positive_weight * positive

    upper_bound_weight = float(item.get("upper_bound_weight", 0.0))
    if upper_bound_weight > 0.0:
        upper = masked_mean(F.relu(pred_nut - pred_nut_from_komega), mask)
        value = value + upper_bound_weight * upper

    return value


def k_omega_nut_sst_branch_loss(
    volume_pred: torch.Tensor,
    volume_target: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
    *,
    batch: dict[str, torch.Tensor | None] | None = None,
    task_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    _require_k_omega_nut(volume_slices)
    denormalize = dict((task_config or {}).get("denormalize", {}))
    denormalize.update(item.get("denormalize", {}))
    cond = None if batch is None else batch.get("cond")
    pred_k, pred_omega, pred_nut, k_scale, omega_scale, nut_scale = _raw_k_omega_nut_fields(
        volume_pred,
        volume_slices,
        denormalize,
        cond,
    )
    target_k, target_omega, target_nut, _, _, _ = _raw_k_omega_nut_fields(
        volume_target,
        volume_slices,
        denormalize,
        cond,
    )

    valid_mask = _valid_target_mask(target_k, target_omega, target_nut, item)
    strict_mask = _relation_gap_mask(target_k, target_omega, target_nut, item, relation_scale=1.0) & valid_mask
    outside_mask = (~strict_mask) & valid_mask

    residual_raw = pred_omega * pred_nut - pred_k
    residual_scale = _relation_residual_scale(item, k_scale, target_k)
    residual = residual_raw / residual_scale
    loss_type = str(item.get("loss_type", "l1"))
    beta = float(item.get("beta", 0.02))
    equality_loss = masked_mean(pointwise_abs_loss(residual, loss_type, beta), strict_mask)

    hinge = F.relu(residual)
    hinge_loss_type = str(item.get("hinge_loss_type", "l1")).lower()
    if hinge_loss_type in {"l1", "mae", "abs"}:
        hinge_values = hinge
    elif hinge_loss_type in {"l2", "mse", "square"}:
        hinge_values = hinge.square()
    elif hinge_loss_type in {"smooth_l1", "huber"}:
        hinge_values = F.smooth_l1_loss(hinge, torch.zeros_like(hinge), reduction="none", beta=beta)
    else:
        raise ValueError(f"Unsupported SST hinge_loss_type: {hinge_loss_type}")
    upper_bound_loss = masked_mean(hinge_values, outside_mask)

    value = (
        float(item.get("equality_weight", item.get("strict_weight", 1.0))) * equality_loss
        + float(item.get("upper_bound_weight", item.get("outside_weight", 1.0))) * upper_bound_loss
    )

    positive_weight = float(item.get("positive_weight", 0.0))
    if positive_weight > 0.0:
        positive = (
            F.relu(-pred_k / k_scale.abs().clamp_min(1e-12)).mean()
            + F.relu(-pred_omega / omega_scale.abs().clamp_min(1e-12)).mean()
            + F.relu(-pred_nut / nut_scale.abs().clamp_min(1e-12)).mean()
        )
        value = value + positive_weight * positive

    return value


def volume_relation_loss(
    field: str,
    volume_pred: torch.Tensor,
    volume_target: torch.Tensor,
    volume_slices: dict[str, slice],
    item: dict[str, Any],
    *,
    batch: dict[str, torch.Tensor | None] | None = None,
    task_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    if field in {"k_omega_nut_sst", "sst_k_omega_nut", "k_omega_nut"}:
        return legacy_k_omega_nut_sst_loss(volume_pred, volume_target, volume_slices, item)
    if field in {
        "k_omega_nut_sst_branch",
        "sst_k_omega_nut_branch",
        "k_omega_nut_upper_bound",
        "sst_branch",
    }:
        return k_omega_nut_sst_branch_loss(
            volume_pred,
            volume_target,
            volume_slices,
            item,
            batch=batch,
            task_config=task_config,
        )
    if field in {"rans_nonnegative", "k_omega_nut_nonnegative"}:
        return rans_nonnegative_loss(
            volume_pred,
            volume_slices,
            item,
            batch=batch,
            task_config=task_config,
        )
    if field in {"jet_taylor_up", "up_taylor", "taylor_up"}:
        return jet_taylor_up_loss(
            volume_pred,
            volume_slices,
            item,
            batch=batch,
            task_config=task_config,
        )
    if field in {
        "local_constant_nu_momentum_ratio",
        "ns_momentum_ratio",
        "rans_momentum_ratio",
    }:
        return local_constant_nu_momentum_ratio_loss(
            volume_pred,
            volume_slices,
            item,
            batch=batch,
            task_config=task_config,
        )
    raise ValueError(f"Unsupported volume_relation field: {field}")
