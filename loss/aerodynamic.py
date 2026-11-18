from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def surface_force_control_variate_coefficient(
    pred_cp: torch.Tensor,
    true_cp: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    reference_area: torch.Tensor,
    reference_coefficient: torch.Tensor,
    *,
    direction: Sequence[float],
) -> torch.Tensor:
    """Return a low-variance coefficient estimate derived from sampled Cp.

    The sampled coefficient estimate is

    ``C_cv = C_CFD + sum((Cp_pred - Cp_CFD) * (n dot d) * dA) / Aref``.

    Comparing ``C_cv`` with ``C_CFD`` cancels the Monte Carlo quadrature noise
    shared by prediction and target.  This is therefore a force constraint on
    the pressure error, not a second noisy estimate of the absolute force.
    """

    if pred_cp.shape != true_cp.shape or pred_cp.ndim != 3 or pred_cp.shape[-1] != 1:
        raise ValueError("pred_cp and true_cp must have the same (B, N, 1) shape.")
    if normals.shape != (*pred_cp.shape[:2], 3):
        raise ValueError("normals must have shape (B, N, 3) aligned with Cp samples.")
    if area.shape != pred_cp.shape:
        raise ValueError("area must have shape (B, N, 1) aligned with Cp samples.")
    dtype = torch.float32
    vector = torch.as_tensor(direction, device=pred_cp.device, dtype=dtype)
    if vector.shape != (3,) or not bool(torch.isfinite(vector).all()):
        raise ValueError("direction must contain three finite values.")
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(torch.finfo(dtype).eps)
    aref = reference_area.float().reshape(pred_cp.shape[0], -1)
    if aref.shape[1] != 1 or not bool(torch.isfinite(aref).all()) or bool((aref <= 0.0).any()):
        raise ValueError("reference_area must contain one finite positive value per batch sample.")
    coefficient_cfd = reference_coefficient.float().reshape(pred_cp.shape[0], -1)
    if coefficient_cfd.shape[1] != 1 or not bool(torch.isfinite(coefficient_cfd).all()):
        raise ValueError("reference_coefficient must contain one finite value per batch sample.")

    projected_normal = torch.einsum("bni,i->bn", normals.float(), vector)
    coefficient_residual = (
        (pred_cp.float() - true_cp.float()).squeeze(-1)
        * projected_normal
        * area.float().squeeze(-1)
    ).sum(dim=1) / aref[:, 0]
    # Written explicitly as a corrected estimate against its CFD label.  The
    # shared sampled truth term cancels algebraically, which is the control
    # variate that removes absolute Monte Carlo force noise.
    corrected_coefficient = coefficient_cfd[:, 0] + coefficient_residual
    return corrected_coefficient[:, None]


def surface_force_control_variate_loss(
    pred_cp: torch.Tensor,
    true_cp: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    reference_area: torch.Tensor,
    reference_coefficient: torch.Tensor,
    *,
    direction: Sequence[float],
    scale: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Huber loss between the control-variate estimate and its CFD label."""

    if float(scale) <= 0.0 or float(beta) <= 0.0:
        raise ValueError("scale and beta must be positive.")
    corrected_coefficient = surface_force_control_variate_coefficient(
        pred_cp,
        true_cp,
        normals,
        area,
        reference_area,
        reference_coefficient,
        direction=direction,
    )
    coefficient_cfd = reference_coefficient.float().reshape(pred_cp.shape[0], -1)
    normalized = (corrected_coefficient[:, 0] - coefficient_cfd[:, 0]) / float(scale)
    return F.smooth_l1_loss(
        normalized,
        torch.zeros_like(normalized),
        reduction="mean",
        beta=float(beta),
    )


__all__ = ["surface_force_control_variate_coefficient", "surface_force_control_variate_loss"]
