from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn


def stop_grad(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def load_and_freeze(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    stop_grad(model)


def voxel_hash(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    return torch.floor(points / voxel_size).int()


def voxel_knn(
    x: torch.Tensor,
    query: torch.Tensor,
    k: int = 16,
    voxel_size: Optional[float] = None,
    radius_factor: int = 2,
) -> torch.Tensor:
    if voxel_size is None:
        span = x.max(dim=0)[0] - x.min(dim=0)[0]
        voxel_size = float(span.max().item()) / 16.0

    x_vox = voxel_hash(x, voxel_size)
    q_vox = voxel_hash(query, voxel_size)
    voxel_dict: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in range(x.shape[0]):
        voxel_dict[tuple(x_vox[idx].tolist())].append(idx)

    offsets = torch.stack(
        torch.meshgrid(
            [
                torch.arange(-radius_factor, radius_factor + 1, device=x.device),
                torch.arange(-radius_factor, radius_factor + 1, device=x.device),
                torch.arange(-radius_factor, radius_factor + 1, device=x.device),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)

    knn_idx = torch.empty(query.shape[0], k, dtype=torch.long, device=x.device)
    for i in range(query.shape[0]):
        candidates: list[int] = []
        for offset in offsets:
            candidates += voxel_dict.get(tuple((q_vox[i] + offset).tolist()), [])
        if not candidates:
            dists = torch.norm(x - query[i], dim=1)
            idx = torch.topk(dists, k, largest=False)[1]
        else:
            candidate_t = torch.as_tensor(candidates, dtype=torch.long, device=x.device)
            dists = torch.norm(x[candidate_t] - query[i], dim=1)
            topk = torch.topk(dists, k=min(k, len(candidates)), largest=False)[1]
            idx = candidate_t[topk]
            if idx.numel() < k:
                pad = idx[torch.randint(0, idx.numel(), (k - idx.numel(),), device=x.device)]
                idx = torch.cat([idx, pad], dim=0)
        knn_idx[i] = idx
    return knn_idx


def knn(x: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(query, x)
    return dists.topk(k, largest=False)[1]


def knn_interpolation(
    dst_xyz: torch.Tensor,
    src_xyz: torch.Tensor,
    src_feat: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    src_feat = src_feat.permute(0, 2, 1).contiguous()
    batch, points, _ = dst_xyz.shape
    channels = src_feat.shape[1]
    dist = torch.cdist(dst_xyz, src_xyz, p=2)
    dists, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    dists = torch.clamp(dists, min=1e-10)
    weights = (1.0 / dists)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    idx_expand = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    src_feat_expand = src_feat.unsqueeze(2).expand(-1, -1, points, -1)
    neighbor_feats = torch.gather(src_feat_expand, dim=3, index=idx_expand)
    interpolated = torch.sum(neighbor_feats * weights.unsqueeze(1), dim=-1)
    return interpolated.permute(0, 2, 1).contiguous()


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    batch, points, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, points), float("inf"), device=xyz.device)
    farthest = torch.randint(0, points, (batch,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


__all__ = [
    "stop_grad",
    "load_and_freeze",
    "voxel_hash",
    "voxel_knn",
    "knn",
    "knn_interpolation",
    "farthest_point_sample",
]
