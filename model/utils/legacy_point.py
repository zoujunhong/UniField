from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(int(k), x.shape[1])
    dist = torch.cdist(query, x)
    return dist.topk(k, largest=False, dim=-1)[1]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(points.shape[0], device=points.device).view(points.shape[0], 1, 1)
    return points[batch_indices, idx, :]


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


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    batch, num_points, _ = xyz.shape
    if num_points == 0:
        raise ValueError("xyz must contain at least one point.")
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, num_points), float("inf"), device=xyz.device)
    farthest = torch.randint(0, num_points, (batch,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=1)[1]
    return centroids


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LinearNorm(nn.Module):
    """Legacy projection block with old checkpoint-compatible attribute names."""

    def __init__(self, inplanes: int, planes: int, act: bool = True, affine: bool = True):
        super().__init__()
        self.norm = nn.RMSNorm(planes, elementwise_affine=affine)
        self.conv = nn.Linear(inplanes, planes)
        self.act = bool(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        return F.gelu(x) if self.act else x


class MultiHeadPointAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 16, num_heads: int = 4):
        super().__init__()
        if out_channels % num_heads != 0:
            raise ValueError("out_channels must be divisible by num_heads.")
        self.k = int(k)
        self.num_heads = int(num_heads)
        self.head_dim = out_channels // num_heads
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_kv = nn.Linear(in_channels, out_channels * 2)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.fc_out = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        batch, num_points, _ = x.shape
        q = self.fc_q(x)
        k, v = self.fc_kv(x).chunk(2, dim=-1)
        q = q.view(batch, num_points, self.num_heads, self.head_dim)
        k = k.view(batch, num_points, self.num_heads, self.head_dim)
        v = v.view(batch, num_points, self.num_heads, self.head_dim)

        with torch.no_grad():
            idx = knn(pos, pos, self.k)
        k_neighbors = index_points(k, idx)
        v_neighbors = index_points(v, idx)
        pos_neighbors = index_points(pos, idx)
        pos_diff = pos.unsqueeze(2) - pos_neighbors
        pos_enc = self.pos_mlp(pos_diff).view(batch, num_points, idx.shape[-1], self.num_heads, self.head_dim)

        rel = k_neighbors - q.unsqueeze(2) + pos_enc
        attn = self.attn_mlp(rel.reshape(batch, num_points, idx.shape[-1], -1))
        attn = attn.view(batch, num_points, idx.shape[-1], self.num_heads, self.head_dim)
        attn = F.softmax(attn, dim=2)
        out = (attn * (v_neighbors + pos_enc)).sum(dim=2)
        return self.fc_out(out.reshape(batch, num_points, -1))


class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels: int, k: int = 16, num_heads: int = 4):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads.")
        self.k = int(k)
        self.num_heads = int(num_heads)
        self.head_dim = in_channels // num_heads
        self.attention = MultiHeadPointAttention(in_channels, in_channels, k, num_heads)
        self.FFN = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.LayerNorm(4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels),
            nn.LayerNorm(in_channels),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, pos)
        return x + self.FFN(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, rel_pos: torch.Tensor) -> torch.Tensor:
        return self.mlp(rel_pos)


class PointTransformerAttention(nn.Module):
    def __init__(self, dim: int, k: int | None = 16):
        super().__init__()
        self.k = k
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.pos_enc = PositionalEncoding(dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        q_feat: torch.Tensor,
        k_feat: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        return_attn: bool = False,
    ):
        q = self.q_proj(q_feat)
        k = self.k_proj(k_feat)
        v = self.v_proj(k_feat)
        batch, slots, dim = q.shape

        if self.k is None:
            rel_pos = q_pos.unsqueeze(2) - k_pos.unsqueeze(1)
            pos_enc = self.pos_enc(rel_pos)
            attn = F.softmax((q.unsqueeze(2) - k.unsqueeze(1) + pos_enc).sum(-1), dim=1)
            attn_norm = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            out = (attn_norm.unsqueeze(-1) * v.unsqueeze(1)).sum(2)
        else:
            idx = knn(k_pos, q_pos, self.k)
            idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, dim)
            k_neighbors = torch.gather(k.unsqueeze(1).expand(-1, slots, -1, -1), 2, idx_feat)
            v_neighbors = torch.gather(v.unsqueeze(1).expand(-1, slots, -1, -1), 2, idx_feat)
            idx_pos = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            k_pos_neighbors = torch.gather(k_pos.unsqueeze(1).expand(-1, slots, -1, -1), 2, idx_pos)
            pos_enc = self.pos_enc(q_pos.unsqueeze(2) - k_pos_neighbors)
            attn_norm = self.softmax((q.unsqueeze(2) - k_neighbors + pos_enc).sum(-1))
            attn = attn_norm / attn_norm.sum(dim=1, keepdim=True).clamp_min(1e-6)
            out = (attn.unsqueeze(-1) * v_neighbors).sum(2)

        if not return_attn:
            return out
        if self.k is None:
            return out, attn
        one_hot_idx = torch.zeros(
            (batch, idx.shape[1], idx.shape[2], k_feat.shape[1]),
            device=k_feat.device,
            dtype=attn.dtype,
        ).scatter_(-1, idx.unsqueeze(-1), 1.0)
        return out, torch.matmul(attn.unsqueeze(2), one_hot_idx).squeeze(2)


class PointSlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3, hidden_dim: int = 128, k: int | None = None):
        super().__init__()
        self.num_slots = int(num_slots)
        self.iters = int(iters)
        self.dim = int(dim)
        self.attn = PointTransformerAttention(dim, k=k)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, pos: torch.Tensor, return_attn: bool = False):
        batch, _, dim = inputs.shape
        inputs = self.norm_inputs(inputs)
        idx = farthest_point_sample(pos, self.num_slots)
        batch_indices = torch.arange(batch, device=inputs.device).unsqueeze(-1)
        slots = inputs[batch_indices, idx]
        slot_pos = pos[batch_indices, idx]

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            if return_attn:
                updates, attn = self.attn(slots_norm, inputs, slot_pos, pos, return_attn=True)
            else:
                updates = self.attn(slots_norm, inputs, slot_pos, pos)
            slots = self.gru(updates.reshape(-1, dim), slots_prev.reshape(-1, dim)).reshape(batch, -1, dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        if return_attn:
            return slots, slot_pos, attn
        return slots, slot_pos


class AdaptLayer(nn.Module):
    def __init__(self, in_channels: int, adapt_channels: int, cond_channels: int):
        super().__init__()
        self.in_proj = LinearNorm(in_channels, adapt_channels)
        self.out_proj = LinearNorm(adapt_channels, in_channels)
        self.cond_proj = nn.Sequential(
            LinearNorm(cond_channels, 2 * adapt_channels),
            nn.Linear(2 * adapt_channels, 2 * adapt_channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.cond_proj(cond).chunk(2, dim=1)
        return x + self.out_proj(modulate(self.in_proj(x), shift, scale))


class MultiFlowCondAdaptRouter(nn.Module):
    def __init__(self, branches: list[nn.Module]):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        if route is None:
            route = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        route = route.to(device=x.device, dtype=torch.long).view(-1)
        if route.shape[0] != x.shape[0]:
            raise ValueError("route must have shape (B,).")
        if route.min().item() < 0 or route.max().item() >= len(self.branches):
            raise ValueError(f"route must be in [0, {len(self.branches)}).")

        out = torch.empty_like(x)
        for route_id in route.unique(sorted=True):
            route_int = int(route_id.item())
            mask = route == route_int
            out[mask] = self.branches[route_int](x[mask], cond[mask])
        return out


__all__ = [
    "AdaptLayer",
    "LinearNorm",
    "MultiFlowCondAdaptRouter",
    "MultiHeadPointAttention",
    "PointSlotAttention",
    "PointTransformerLayer",
    "farthest_point_sample",
    "index_points",
    "knn",
    "knn_interpolation",
    "modulate",
]
