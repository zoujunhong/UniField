from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath

from model.utils.ffn import AdaLN, FFN
from model.utils.gaussian3d import gather_tokens, inverse_permutation, morton_sort_indices, normalize_quaternion


class WindowSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attn_drop = float(attn_drop)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_windows, window_size, _ = x.shape
        qkv = self.qkv(x).view(batch_windows, window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_mask = None
        if valid is not None and not bool(valid.all()):
            attn_mask = x.new_zeros(batch_windows, 1, 1, window_size)
            attn_mask = attn_mask.masked_fill(~valid[:, None, None, :], -torch.finfo(x.dtype).max)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_windows, window_size, self.dim)
        return self.proj_drop(self.proj(y))


class GaussianShiftWindowBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        cond_dim: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.shift_size < 0 or self.shift_size >= self.window_size:
            raise ValueError("shift_size must satisfy 0 <= shift_size < window_size.")
        self.norm_attn = AdaLN(self.dim, int(cond_dim))
        self.attn = WindowSelfAttention(self.dim, int(num_heads), attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()
        self.norm_ffn = AdaLN(self.dim, int(cond_dim))
        self.ffn = FFN(self.dim, mlp_ratio=float(mlp_ratio), drop=float(proj_drop))
        self.drop_path2 = DropPath(float(drop_path)) if float(drop_path) > 0.0 else nn.Identity()

    def _partition(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]:
        batch, token_count, channels = x.shape
        if self.shift_size > 0:
            left = x.new_zeros(batch, self.shift_size, channels)
            right = x.new_zeros(batch, self.window_size - self.shift_size, channels)
            x = torch.cat([left, x, right], dim=1)
            valid = torch.cat(
                [
                    torch.zeros(batch, self.shift_size, dtype=torch.bool, device=x.device),
                    torch.ones(batch, token_count, dtype=torch.bool, device=x.device),
                    torch.zeros(batch, self.window_size - self.shift_size, dtype=torch.bool, device=x.device),
                ],
                dim=1,
            )
        else:
            remainder = token_count % self.window_size
            pad = 0 if remainder == 0 else self.window_size - remainder
            if pad > 0:
                x = torch.cat([x, x.new_zeros(batch, pad, channels)], dim=1)
            valid = torch.cat(
                [
                    torch.ones(batch, token_count, dtype=torch.bool, device=x.device),
                    torch.zeros(batch, pad, dtype=torch.bool, device=x.device),
                ],
                dim=1,
            )
        total = x.shape[1]
        groups = total // self.window_size
        x_windows = x.view(batch, groups, self.window_size, channels).reshape(batch * groups, self.window_size, channels)
        valid_windows = valid.view(batch, groups, self.window_size).reshape(batch * groups, self.window_size)
        cond_windows = None
        if cond is not None:
            cond_windows = cond[:, None, :].expand(-1, groups, -1).reshape(batch * groups, -1)
        return x_windows, cond_windows, valid_windows, total

    def _reverse(self, x_windows: torch.Tensor, batch: int, token_count: int, total: int) -> torch.Tensor:
        x = x_windows.view(batch, total // self.window_size, self.window_size, self.dim).reshape(batch, total, self.dim)
        if self.shift_size > 0:
            return x[:, self.shift_size : self.shift_size + token_count]
        return x[:, :token_count]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        batch, token_count, _ = x.shape
        x_windows, cond_windows, valid_windows, total = self._partition(x, cond)
        y = self.norm_attn(x_windows, cond_windows)
        x_windows = x_windows + self.drop_path1(self.attn(y, valid_windows))
        y = self.norm_ffn(x_windows, cond_windows)
        x_windows = x_windows + self.drop_path2(self.ffn(y))
        return self._reverse(x_windows, batch, token_count, total)


class Gaussian3DMortonShiftEncoder(nn.Module):
    """Morton-sorted shifted-window transformer encoder for 3D Gaussian caches."""

    def __init__(
        self,
        num_gaussians: int = 4096,
        gaussian_channels: int = 32,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        window_size: int = 512,
        shift_size: int | None = None,
        cond_dim: int = 2,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        morton_bits: int = 10,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.gaussian_channels = int(gaussian_channels)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.window_size = int(window_size)
        self.shift_size = self.window_size // 2 if shift_size is None else int(shift_size)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.morton_bits = int(morton_bits)
        self.memory_dim = self.embed_dim if memory_dim is None else int(memory_dim)
        if self.memory_dim != self.embed_dim:
            raise ValueError("Gaussian3DMortonShiftEncoder memory_dim must equal embed_dim.")
        if self.num_routes < 1:
            raise ValueError("num_routes must be >= 1.")
        input_dim = 3 + 3 + 4 + 1 + self.gaussian_channels
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
        )
        dpr = torch.linspace(0.0, float(drop_path_rate), self.depth).tolist()
        self.blocks = nn.ModuleList(
            [
                GaussianShiftWindowBlock(
                    dim=self.embed_dim,
                    num_heads=int(num_heads),
                    window_size=self.window_size,
                    shift_size=0 if i % 2 == 0 else self.shift_size,
                    cond_dim=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                )
                for i in range(self.depth)
            ]
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def _prepare_cond(self, pos: torch.Tensor, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cond_dim <= 0:
            return None
        if cond is None:
            return pos.new_zeros(pos.shape[0], self.cond_dim)
        if cond.shape != (pos.shape[0], self.cond_dim):
            raise ValueError(f"cond must have shape (B, {self.cond_dim}).")
        return cond.to(device=pos.device, dtype=pos.dtype)

    def _prepare_routes(self, pos: torch.Tensor, routes: Optional[torch.Tensor | int]) -> torch.Tensor:
        if routes is None:
            return torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        if isinstance(routes, int):
            return torch.full((pos.shape[0],), routes, dtype=torch.long, device=pos.device)
        routes_t = routes.to(device=pos.device, dtype=torch.long)
        if routes_t.dim() == 0:
            routes_t = routes_t.expand(pos.shape[0])
        return routes_t.view(-1)

    def _prepare_flow(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor],
        flow_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del cond
        if flow_vec is not None:
            flow = flow_vec.to(device=pos.device, dtype=pos.dtype)
            if flow.shape != (pos.shape[0], 3):
                raise ValueError("flow_vec must have shape (B, 3).")
            return F.normalize(flow, dim=-1)
        default = pos.new_tensor([1.0, 0.0, 0.0]).view(1, 3)
        return default.expand(pos.shape[0], -1)

    @staticmethod
    def _metadata_tensor(
        geometry_meta: Mapping[str, torch.Tensor] | None,
        key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if geometry_meta is None or key not in geometry_meta:
            raise ValueError(f"Gaussian3DMortonShiftEncoder requires geometry_meta[{key!r}].")
        value = geometry_meta[key]
        if not torch.is_tensor(value):
            raise TypeError(f"geometry_meta[{key!r}] must be a tensor.")
        return value.to(device=reference.device, dtype=reference.dtype)

    def _parse_input(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = 3 + 3 + 4 + 1 + self.gaussian_channels
        if pos.dim() != 3 or pos.shape[1] != self.num_gaussians or pos.shape[-1] != expected:
            raise ValueError(f"Gaussian3D input must have shape (B, {self.num_gaussians}, {expected}).")
        means = pos[..., 0:3]
        log_scales = pos[..., 3:6]
        rotations = normalize_quaternion(pos[..., 6:10])
        opacity_logits = pos[..., 10:11]
        features = pos[..., 11:]
        return means, log_scales, rotations, opacity_logits, features

    def forward(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        return_pyramid: bool = False,
        geometry_meta: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del normals, area, flow_vec, routes, return_pyramid
        means, log_scales, rotations, opacity_logits, features = self._parse_input(pos)
        cond_t = self._prepare_cond(pos, cond)
        bounds = self._metadata_tensor(geometry_meta, "bounds", pos)
        if bounds.dim() == 2 and bounds.shape[-1] == 6:
            bounds = bounds.view(pos.shape[0], 2, 3)
        if bounds.shape != (pos.shape[0], 2, 3):
            raise ValueError("geometry_meta['bounds'] must have shape (B, 2, 3) or (B, 6).")

        span = (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-6)
        means_norm = 2.0 * (means - bounds[:, None, 0]) / span[:, None] - 1.0
        token_input = torch.cat([means_norm, log_scales, rotations, opacity_logits, features], dim=-1)
        sort_idx = morton_sort_indices(means, bounds, bits=self.morton_bits)
        inv_idx = inverse_permutation(sort_idx)
        x = gather_tokens(self.input_proj(token_input), sort_idx)
        for block in self.blocks:
            x = block(x, cond_t)
        x = self.out_norm(x)
        refined = gather_tokens(x, inv_idx)
        opacity = torch.sigmoid(opacity_logits)
        return {
            "means": means,
            "log_scales": log_scales,
            "rotations": rotations,
            "opacity_logits": opacity_logits,
            "features": refined,
            "bounds": bounds,
            "points": means,
            "area": opacity,
            "log_area": torch.log(opacity.clamp_min(1e-12)),
        }


__all__ = ["Gaussian3DMortonShiftEncoder", "GaussianShiftWindowBlock", "WindowSelfAttention"]
