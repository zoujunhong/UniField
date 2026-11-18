from __future__ import annotations

import torch


def knn_interpolation(
    dst_xyz: torch.Tensor,
    src_xyz: torch.Tensor,
    src_feat: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    if src_xyz.shape[1] == 0:
        raise ValueError("src_xyz must contain at least one point.")
    k = min(int(k), src_xyz.shape[1])
    dist = torch.cdist(dst_xyz, src_xyz, p=2)
    dists, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    weights = 1.0 / dists.clamp_min(1e-10)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, src_feat.shape[-1])
    feat_expand = src_feat.unsqueeze(1).expand(-1, dst_xyz.shape[1], -1, -1)
    neighbor_feat = torch.gather(feat_expand, dim=2, index=idx_expand)
    return (neighbor_feat * weights.unsqueeze(-1)).sum(dim=2)
