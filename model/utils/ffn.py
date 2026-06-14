from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act: bool = True, affine: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=affine)
        self.act = bool(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.proj(x))
        return F.gelu(x) if self.act else x


class AdaLN(nn.Module):
    """Single conditional LayerNorm branch."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        if self.cond_dim > 0:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 2),
            )
        else:
            self.shift = nn.Parameter(torch.zeros(dim))
            self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm(x)
        if self.cond_dim > 0:
            if cond is None:
                cond = x.new_zeros(x.shape[0], self.cond_dim)
            else:
                cond = cond.to(device=x.device, dtype=x.dtype)
            shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        else:
            shift = self.shift.to(dtype=x.dtype).expand(x.shape[0], -1)
            scale = self.scale.to(dtype=x.dtype).expand(x.shape[0], -1)
        return modulate(x, shift, scale)


class RoutedAdaLN(nn.Module):
    """Route-aware AdaLN wrapper. Each route owns one AdaLN branch."""

    def __init__(self, dim: int, cond_dim: int, num_routes: int = 1):
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        if self.num_routes < 1:
            raise ValueError("num_routes must be >= 1.")
        self.branches = nn.ModuleList([AdaLN(dim, cond_dim) for _ in range(self.num_routes)])

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.num_routes == 1:
            return self.branches[0](x, cond).to(dtype=x.dtype)

        batch = x.shape[0]
        if routes is None:
            routes_t = torch.zeros(batch, dtype=torch.long, device=x.device)
        else:
            routes_t = routes.to(device=x.device, dtype=torch.long).view(-1)
        if routes_t.shape[0] != batch:
            raise ValueError("routes must have shape (B,), be scalar-expanded before windowing, or be None.")
        if routes_t.min().item() < 0 or routes_t.max().item() >= self.num_routes:
            raise ValueError(f"routes must be in [0, {self.num_routes}).")

        cond_t = None
        if cond is not None:
            cond_t = cond.to(device=x.device, dtype=x.dtype)

        out = torch.empty_like(x)
        for route_id in routes_t.unique(sorted=True):
            route = int(route_id.item())
            mask = routes_t == route
            branch_out = self.branches[route](x[mask], None if cond_t is None else cond_t[mask])
            out[mask] = branch_out.to(dtype=x.dtype)
        return out


def apply_norm(
    norm: nn.Module,
    x: torch.Tensor,
    cond: Optional[torch.Tensor],
    routes: Optional[torch.Tensor],
) -> torch.Tensor:
    if isinstance(norm, RoutedAdaLN):
        return norm(x, cond, routes)
    return norm(x)


class FFN(nn.Module):
    """Plain transformer feed-forward network."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


__all__ = ["AdaLN", "RoutedAdaLN", "FFN", "LinearNorm", "apply_norm", "modulate"]
