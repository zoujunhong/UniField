from __future__ import annotations

import torch
import torch.nn.functional as F


TRIPLANE_AXES = ((0, 1), (1, 2), (0, 2))


def normalize_query_to_bounds(query: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    if bounds.dim() == 2 and bounds.shape[-1] == 6:
        bounds = bounds.view(bounds.shape[0], 2, 3)
    if bounds.dim() != 3 or bounds.shape[1:] != (2, 3):
        raise ValueError("bounds must have shape (B, 2, 3) or (B, 6).")
    lower = bounds[:, 0, :].to(device=query.device, dtype=query.dtype)
    upper = bounds[:, 1, :].to(device=query.device, dtype=query.dtype)
    span = (upper - lower).clamp_min(1e-12)
    return 2.0 * (query - lower[:, None, :]) / span[:, None, :] - 1.0


def sample_triplanes(
    planes: torch.Tensor,
    query: torch.Tensor,
    bounds: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample xy/yz/xz feature planes at 3D query locations."""
    if planes.dim() != 5 or planes.shape[1] != 3:
        raise ValueError("planes must have shape (B, 3, C, H, W).")
    if query.dim() != 3 or query.shape[-1] != 3:
        raise ValueError("query must have shape (B, N, 3).")
    if planes.shape[0] != query.shape[0]:
        raise ValueError("planes and query batch size must match.")

    query_norm = normalize_query_to_bounds(query, bounds)
    samples = []
    for plane_idx, (axis_u, axis_v) in enumerate(TRIPLANE_AXES):
        grid = torch.stack(
            [query_norm[..., axis_u], query_norm[..., axis_v]],
            dim=-1,
        ).unsqueeze(2)
        sampled = F.grid_sample(
            planes[:, plane_idx],
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        samples.append(sampled.squeeze(-1).transpose(1, 2))
    return torch.cat(samples, dim=-1), query_norm


__all__ = ["TRIPLANE_AXES", "normalize_query_to_bounds", "sample_triplanes"]
