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


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample/per-channel relative L2, averaged over batch and field channels."""

    if pred.shape != target.shape:
        raise ValueError("relative_l2_loss requires prediction and target tensors with the same shape.")
    if pred.ndim < 2:
        return torch.sqrt((pred - target).square().sum() / target.square().sum().clamp_min(eps))
    if pred.ndim == 2:
        numerator = (pred - target).square().sum(dim=1)
        denominator = target.square().sum(dim=1).clamp_min(eps)
        return torch.sqrt(numerator / denominator).mean()
    reduce_dims = tuple(range(1, pred.ndim - 1))
    numerator = (pred - target).square().sum(dim=reduce_dims)
    denominator = target.square().sum(dim=reduce_dims).clamp_min(eps)
    return torch.sqrt(numerator / denominator).mean()


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean cosine distance for vector fields, evaluated in FP32.

    Surface-normal supervision is often evaluated while the model runs under
    mixed precision.  Keeping the norms and division explicitly in FP32
    avoids a zero epsilon/zero-norm division in FP16 and gives a bounded,
    inspectable loss for a newly initialized normal head.
    """

    if pred.shape != target.shape:
        raise ValueError("cosine_loss requires prediction and target tensors with the same shape.")
    if pred.ndim < 1 or pred.shape[-1] < 1:
        raise ValueError("cosine_loss requires a non-empty vector channel dimension.")
    eps_value = max(float(eps), float(torch.finfo(torch.float32).eps))
    pred_float = pred.float()
    target_float = target.float()
    target_norm = torch.linalg.vector_norm(target_float, dim=-1)
    valid = target_norm > eps_value
    if not bool(valid.any()):
        return pred_float.new_zeros(())
    pred_unit = pred_float / torch.linalg.vector_norm(pred_float, dim=-1, keepdim=True).clamp_min(eps_value)
    target_unit = target_float / target_norm.unsqueeze(-1).clamp_min(eps_value)
    cosine = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine)[valid].mean()


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
            "relative_l2_weight",
            "cosine_weight",
        )
    )

    if has_weight:
        mae_weight = float(item.get("mae_weight", item.get("l1_weight", 0.0)))
        mse_weight = float(item.get("mse_weight", item.get("l2_weight", 0.0)))
        rmse_weight = float(item.get("rmse_weight", 0.0))
        smooth_l1_weight = float(item.get("smooth_l1_weight", 0.0))
        relative_l2_weight = float(item.get("relative_l2_weight", 0.0))
        cosine_weight = float(item.get("cosine_weight", 0.0))
    else:
        loss_type = str(item.get("loss_type", "l1")).lower()
        mae_weight = 1.0 if loss_type in {"l1", "mae", "abs"} else 0.0
        mse_weight = 1.0 if loss_type in {"l2", "mse", "square"} else 0.0
        rmse_weight = 1.0 if loss_type == "rmse" else 0.0
        smooth_l1_weight = 1.0 if loss_type in {"smooth_l1", "huber"} else 0.0
        relative_l2_weight = 1.0 if loss_type in {"relative_l2", "rel_l2"} else 0.0
        cosine_weight = 1.0 if loss_type in {"cosine", "cosine_distance", "normal_cosine"} else 0.0
        if mae_weight + mse_weight + rmse_weight + smooth_l1_weight + relative_l2_weight + cosine_weight == 0.0:
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
    if relative_l2_weight > 0.0:
        value = value + relative_l2_weight * relative_l2_loss(pred, target, eps=float(item.get("relative_l2_eps", 1e-12)))
    if cosine_weight > 0.0:
        value = value + cosine_weight * cosine_loss(pred, target, eps=float(item.get("cosine_eps", 1e-6)))
    return value
