from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    if not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].mean()


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean()


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(mse_loss(pred, target) + eps)


def pointwise_abs_loss(diff: torch.Tensor, loss_type: str = "l1", beta: float = 0.02) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type in {"l1", "mae", "abs"}:
        return diff.abs()
    if loss_type in {"l2", "mse", "square"}:
        return diff.square()
    if loss_type in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=beta)
    raise ValueError(f"Unsupported pointwise loss_type: {loss_type}")


def supervised_field_loss(pred: torch.Tensor, target: torch.Tensor, item: dict[str, Any]) -> torch.Tensor:
    diff = pred - target
    has_weight = any(
        key in item
        for key in (
            "mae_weight",
            "l1_weight",
            "mse_weight",
            "l2_weight",
            "rmse_weight",
            "smooth_l1_weight",
        )
    )

    if has_weight:
        mae_weight = float(item.get("mae_weight", item.get("l1_weight", 0.0)))
        mse_weight = float(item.get("mse_weight", item.get("l2_weight", 0.0)))
        rmse_weight = float(item.get("rmse_weight", 0.0))
        smooth_l1_weight = float(item.get("smooth_l1_weight", 0.0))
    else:
        loss_type = str(item.get("loss_type", "l1")).lower()
        mae_weight = 1.0 if loss_type in {"l1", "mae", "abs"} else 0.0
        mse_weight = 1.0 if loss_type in {"l2", "mse", "square"} else 0.0
        rmse_weight = 1.0 if loss_type == "rmse" else 0.0
        smooth_l1_weight = 1.0 if loss_type in {"smooth_l1", "huber"} else 0.0
        if mae_weight + mse_weight + rmse_weight + smooth_l1_weight == 0.0:
            raise ValueError(f"Unsupported supervised loss_type: {loss_type}")

    value = diff.new_zeros(())
    if mae_weight > 0.0:
        value = value + mae_weight * diff.abs().mean()
    if mse_weight > 0.0:
        value = value + mse_weight * diff.square().mean()
    if rmse_weight > 0.0:
        value = value + rmse_weight * torch.sqrt(diff.square().mean() + float(item.get("rmse_eps", 1e-12)))
    if smooth_l1_weight > 0.0:
        beta = float(item.get("beta", 0.02))
        value = value + smooth_l1_weight * F.smooth_l1_loss(pred, target, reduction="mean", beta=beta)
    return value
