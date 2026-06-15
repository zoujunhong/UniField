from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Attention

from model.utils.downsample import PointCrossAttention, build_downsampled_coords
from model.utils.ffn import modulate


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return modulate(self.norm(x), shift, scale)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int = 2,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        del num_routes
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=nn.LayerNorm,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaLN(dim, cond_dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        del routes
        x = x + self.drop_path1(self.attn(self.norm1(x, cond)))
        return x + self.drop_path2(self.mlp(self.norm2(x, cond)))


class SwinBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        patch_size: int = 0,
        shift_size: int = 0,
        cond_dim: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.shift_size = int(shift_size)
        self.patch_size = int(patch_size)
        self.attn = Block(
            dim,
            num_heads,
            cond_dim,
            num_routes,
            mlp_ratio,
            qkv_bias=True,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        batch, points, channels = x.shape
        if points % self.patch_size != 0:
            raise ValueError(f"Point count {points} must be divisible by patch_size {self.patch_size}.")
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
        groups = points // self.patch_size
        x = x.view(batch * groups, self.patch_size, channels)
        cond_w = cond.view(batch, 1, -1).repeat(1, groups, 1).view(batch * groups, -1)
        routes_w = routes.view(batch, 1).repeat(1, groups).view(batch * groups)
        x = self.attn(x, cond_w, routes_w).view(batch, points, channels)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        return x


class PTv3EncoderStage1D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_routes: int,
        patch_size: int,
        down_ratio: Optional[int],
        next_dim: Optional[int],
        cond_dims: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock1D(
                    dim=dim,
                    num_heads=num_heads,
                    num_routes=num_routes,
                    mlp_ratio=mlp_ratio,
                    patch_size=patch_size,
                    shift_size=0 if i % 2 == 0 else patch_size // 2,
                    cond_dim=cond_dims,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )
        self.sample_patch_size = int(patch_size)
        self.down_ratio = down_ratio
        self.downsample = down_ratio is not None and next_dim is not None
        if self.downsample:
            self.sample_patch_size_out = patch_size // int(down_ratio)
            self.next_dim = int(next_dim)
            self.proj_down = nn.Linear(dim, next_dim)
            self.downsample_layer = PointCrossAttention(
                dim,
                num_heads,
                int(down_ratio),
                int(patch_size),
                hidden_dim=dim * 4,
            )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        cond: torch.Tensor,
        routes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, cond, routes)
        x_skip, pos_skip = x, pos
        if self.downsample:
            pos_ds, x_ds = build_downsampled_coords(pos, x, int(self.down_ratio))
            x_ds = self.downsample_layer(x, x_ds)
            x_ds = self.proj_down(x_ds)
            return x_ds, pos_ds, x_skip, pos_skip
        return x, pos, x_skip, pos_skip


class UniFieldV2Encoder(nn.Module):
    """UniField_1210_2 encoder half, adapted to the QueryUniField memory interface."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (4, 4, 8, 8),
        patch_sizes: tuple[int, ...] = (16, 16, 16, 16),
        down_ratios: tuple[int, ...] = (2, 2, 2),
        cond_dim: int = 2,
        cond_dims: Optional[int] = None,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        default_flow_vec: tuple[float, float, float] = (1.0, 0.0, 0.0),
        flow_from_cond: bool = False,
        **_: object,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("UniFieldV2Encoder expects xyz input with 3 channels.")
        if cond_dims is not None:
            cond_dim = int(cond_dims)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.flow_from_cond = bool(flow_from_cond)
        self.num_stages = len(embed_dims)
        self.embed_dims = tuple(int(v) for v in embed_dims)
        self.depths = tuple(int(v) for v in depths)
        self.num_heads = tuple(int(v) for v in num_heads)
        self.patch_sizes = tuple(int(v) for v in patch_sizes)
        self.down_ratios = tuple(int(v) for v in down_ratios)
        if not (len(depths) == len(num_heads) == len(patch_sizes) == self.num_stages):
            raise ValueError("embed_dims, depths, num_heads, and patch_sizes must have the same length.")
        if len(down_ratios) != self.num_stages - 1:
            raise ValueError("down_ratios length must be num_stages - 1.")
        self.register_buffer(
            "default_flow_vec",
            torch.tensor(default_flow_vec, dtype=torch.float32).view(1, 3),
            persistent=False,
        )
        self.input_proj = nn.Linear(in_channels, self.embed_dims[0])

        total_blocks = sum(self.depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dpr_idx = 0
        stages = []
        for i in range(self.num_stages):
            next_dim = self.embed_dims[i + 1] if i < self.num_stages - 1 else None
            down_ratio = self.down_ratios[i] if i < self.num_stages - 1 else None
            stages.append(
                PTv3EncoderStage1D(
                    dim=self.embed_dims[i],
                    depth=self.depths[i],
                    num_heads=self.num_heads[i],
                    num_routes=self.num_routes,
                    patch_size=self.patch_sizes[i],
                    down_ratio=down_ratio,
                    next_dim=next_dim,
                    cond_dims=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx + self.depths[i] - 1] if self.depths[i] > 0 else 0.0,
                )
            )
            dpr_idx += self.depths[i]
        self.enc_stages = nn.ModuleList(stages)

    def _prepare_cond(self, pos: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_dim <= 0:
            return pos.new_zeros(pos.shape[0], 0)
        if cond is None:
            return pos.new_zeros(pos.shape[0], self.cond_dim)
        if cond.shape[0] != pos.shape[0] or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}).")
        return cond.to(device=pos.device, dtype=pos.dtype)

    def _prepare_routes(self, pos: torch.Tensor, routes: Optional[torch.Tensor | int]) -> torch.Tensor:
        if routes is None:
            routes_t = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        elif isinstance(routes, int):
            routes_t = torch.full((pos.shape[0],), routes, dtype=torch.long, device=pos.device)
        else:
            routes_t = routes.to(device=pos.device, dtype=torch.long)
            if routes_t.dim() == 0:
                routes_t = routes_t.expand(pos.shape[0])
            routes_t = routes_t.view(-1)
        if routes_t.shape[0] != pos.shape[0]:
            raise ValueError("routes must have shape (B,), be scalar, or be None.")
        return routes_t

    def _prepare_flow(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor],
        flow_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if flow_vec is not None:
            flow = flow_vec.to(device=pos.device, dtype=pos.dtype)
        elif self.flow_from_cond and cond is not None and cond.shape[-1] >= 2:
            flow = torch.cat([cond[:, :2], cond.new_zeros(cond.shape[0], 1)], dim=-1)
        else:
            flow = self.default_flow_vec.to(device=pos.device, dtype=pos.dtype).expand(pos.shape[0], -1)
        if flow.shape != (pos.shape[0], 3):
            raise ValueError("flow_vec must have shape (B, 3).")
        return F.normalize(flow, dim=-1)

    def forward(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        return_pyramid: bool = True,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        del normals, area
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError("pos must have shape (B, N, 3).")
        cond_t = self._prepare_cond(pos, cond)
        routes_t = self._prepare_routes(pos, routes)
        flow = self._prepare_flow(pos, cond_t, flow_vec)
        x = self.input_proj(pos)
        feats_skip: list[torch.Tensor] = []
        poss_skip: list[torch.Tensor] = []
        pyramid: list[dict[str, torch.Tensor]] = []
        for stage in self.enc_stages:
            x, pos, x_skip, pos_skip = stage(x, pos, cond_t, routes_t)
            feats_skip.append(x_skip)
            poss_skip.append(pos_skip)
            if return_pyramid:
                pyramid.append(
                    {
                        "features": x_skip,
                        "points": pos_skip,
                        "normals": pos_skip.new_zeros(pos_skip.shape),
                        "area": pos_skip.new_ones(pos_skip.shape[0], pos_skip.shape[1], 1),
                    }
                )
        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "features": x,
            "points": pos,
            "feats_skip": feats_skip,
            "poss_skip": poss_skip,
            "normals": pos.new_zeros(pos.shape),
            "area": pos.new_ones(pos.shape[0], pos.shape[1], 1),
            "log_area": pos.new_zeros(pos.shape[0], pos.shape[1], 1),
            "flow": flow,
        }
        if return_pyramid:
            output["pyramid"] = pyramid
        return output


__all__ = [
    "AdaLN",
    "Block",
    "PTv3EncoderStage1D",
    "SwinBlock1D",
    "UniFieldV2Encoder",
]
