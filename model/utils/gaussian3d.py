from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def normalize_quaternion(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def quaternion_rotate_inverse(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate world-space vectors into the local frame of normalized quaternions.

    Quaternions use (w, x, y, z) order.
    """
    quat = normalize_quaternion(quat)
    w = quat[..., :1]
    qvec = -quat[..., 1:]
    uv = torch.cross(qvec, vectors, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vectors + 2.0 * (w * uv + uuv)


def gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError("values must have shape (B, M, C).")
    if indices.dim() != 3:
        raise ValueError("indices must have shape (B, N, K).")
    b, n, k = indices.shape
    if values.shape[0] != b:
        raise ValueError("Batch size mismatch between values and indices.")
    _, m, c = values.shape
    offsets = torch.arange(b, device=indices.device, dtype=indices.dtype).view(b, 1, 1) * m
    flat_indices = (indices + offsets).reshape(-1)
    gathered = values.reshape(b * m, c).index_select(0, flat_indices)
    return gathered.view(b, n, k, c)


def gaussian_rbf_aggregate(
    query: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity_logits: torch.Tensor,
    features: torch.Tensor,
    *,
    k: int = 32,
    eps: float = 1e-8,
    max_query_chunk: int = 0,
    return_density: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Aggregate Gaussian features at query points with anisotropic RBF weights."""
    if query.dim() != 3 or query.shape[-1] < 3:
        raise ValueError("query must have shape (B, N, C) with C >= 3.")
    if means.dim() != 3 or means.shape[-1] != 3:
        raise ValueError("means must have shape (B, M, 3).")
    if features.dim() != 3 or features.shape[:2] != means.shape[:2]:
        raise ValueError("features must have shape (B, M, C).")
    if max_query_chunk and query.shape[1] > int(max_query_chunk):
        outputs = []
        densities = []
        for start in range(0, query.shape[1], int(max_query_chunk)):
            chunk = query[:, start : start + int(max_query_chunk)]
            value = gaussian_rbf_aggregate(
                chunk,
                means,
                log_scales,
                rotations,
                opacity_logits,
                features,
                k=k,
                eps=eps,
                max_query_chunk=0,
                return_density=return_density,
            )
            if return_density:
                chunk_features, chunk_density = value
                outputs.append(chunk_features)
                densities.append(chunk_density)
            else:
                outputs.append(value)
        features_out = torch.cat(outputs, dim=1)
        if return_density:
            return features_out, torch.cat(densities, dim=1)
        return features_out

    query_xyz = query[..., :3]
    k_eff = min(int(k), int(means.shape[1]))
    with torch.no_grad():
        distances = torch.cdist(query_xyz.float(), means.float())
        _, indices = torch.topk(distances, k=k_eff, dim=-1, largest=False, sorted=False)

    nbr_means = gather_neighbors(means, indices)
    nbr_log_scales = gather_neighbors(log_scales, indices)
    nbr_rotations = gather_neighbors(rotations, indices)
    nbr_opacity = gather_neighbors(opacity_logits, indices)
    nbr_features = gather_neighbors(features, indices)

    diff = query_xyz[:, :, None, :] - nbr_means
    local = quaternion_rotate_inverse(nbr_rotations, diff)
    scales = nbr_log_scales.clamp(-8.0, 4.0).exp().clamp_min(1e-5)
    energy = (local / scales).square().sum(dim=-1, keepdim=True).clamp_max(80.0)
    weights = torch.exp(-0.5 * energy) * torch.sigmoid(nbr_opacity)
    density = weights.sum(dim=-2)
    weights = weights / density[:, :, None, :].clamp_min(float(eps))
    aggregated = (weights.to(dtype=nbr_features.dtype) * nbr_features).sum(dim=-2)
    if return_density:
        return aggregated, density
    return aggregated


def _part1by2(x: torch.Tensor) -> torch.Tensor:
    x = x & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def morton3d_codes(points: torch.Tensor, bounds: torch.Tensor, bits: int = 10) -> torch.Tensor:
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B, N, 3).")
    if bounds.shape != (points.shape[0], 2, 3):
        raise ValueError("bounds must have shape (B, 2, 3).")
    max_coord = (1 << int(bits)) - 1
    span = (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-6)
    coords = ((points - bounds[:, None, 0]) / span[:, None]).clamp(0.0, 1.0)
    coords_i = torch.floor(coords * max_coord).to(torch.long)
    x = _part1by2(coords_i[..., 0])
    y = _part1by2(coords_i[..., 1])
    z = _part1by2(coords_i[..., 2])
    return x | (y << 1) | (z << 2)


def morton_sort_indices(points: torch.Tensor, bounds: torch.Tensor, bits: int = 10) -> torch.Tensor:
    return torch.argsort(morton3d_codes(points, bounds, bits=bits), dim=-1, stable=True)


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tokens.dim() != 3 or indices.dim() != 2:
        raise ValueError("tokens must be (B, N, C) and indices must be (B, N).")
    return torch.gather(tokens, 1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


def inverse_permutation(indices: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(indices)
    arange = torch.arange(indices.shape[1], device=indices.device).expand_as(indices)
    inv.scatter_(1, indices, arange)
    return inv


def atomic_save_npz(path: str | Path, **arrays) -> None:
    import os
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


__all__ = [
    "atomic_save_npz",
    "gather_neighbors",
    "gather_tokens",
    "gaussian_rbf_aggregate",
    "inverse_permutation",
    "morton3d_codes",
    "morton_sort_indices",
    "normalize_quaternion",
    "quaternion_rotate_inverse",
]
