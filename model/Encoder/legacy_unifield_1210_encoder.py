from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Attention

from model.utils.attention import MultiheadAttention
from model.utils.ffn import AdaLN


class LegacyTransformerBlock(nn.Module):
    """Block used by the old UniField_1210_2 point U-Net."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=nn.LayerNorm,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = AdaLN(dim, cond_dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x, cond)))
        x = x + self.drop_path2(self.mlp(self.norm2(x, cond)))
        return x


class LegacySwinBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: int,
        shift_size: int = 0,
        cond_dim: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.shift_size = int(shift_size)
        self.attn = LegacyTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor], routes: torch.Tensor) -> torch.Tensor:
        del routes
        batch, point_count, channels = x.shape
        if point_count % self.patch_size != 0:
            raise ValueError(f"Point count {point_count} must be divisible by patch_size={self.patch_size}.")

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        groups = point_count // self.patch_size
        x = x.view(batch * groups, self.patch_size, channels)
        if cond is not None:
            cond = cond[:, None, :].expand(-1, groups, -1).reshape(batch * groups, -1)
        x = self.attn(x, cond).view(batch, point_count, channels)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        return x


class LegacyPointCrossAttention(nn.Module):
    """Patch-local slot update used for old UniField_1210_2 downsampling."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        down_ratio: int,
        patch_size: int,
        iters: int = 3,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.iters = int(iters)
        self.dim = int(dim)
        self.down_ratio = int(down_ratio)
        self.patch_size = int(patch_size)
        hidden_dim = int(dim * 4 if hidden_dim is None else hidden_dim)

        self.attn = MultiheadAttention(dim, num_heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, slots_init: torch.Tensor) -> torch.Tensor:
        batch, input_count, dim = inputs.shape
        slot_count = slots_init.shape[1]
        if input_count // slot_count != self.down_ratio:
            raise ValueError("LegacyPointCrossAttention expects slot_count = input_count / down_ratio.")
        if input_count % self.patch_size != 0:
            raise ValueError(f"Input count {input_count} must be divisible by patch_size={self.patch_size}.")

        groups = input_count // self.patch_size
        slot_patch_size = self.patch_size // self.down_ratio
        inputs = inputs.view(batch, groups, self.patch_size, dim).flatten(0, 1)
        slots = slots_init.view(batch, groups, slot_patch_size, dim).flatten(0, 1)
        inputs = self.norm_inputs(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            updates = self.attn(self.norm_slots(slots), inputs, inputs)
            slots = self.gru(updates.reshape(-1, dim), slots_prev.reshape(-1, dim)).reshape(
                batch * groups,
                slot_patch_size,
                dim,
            )
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots.view(batch, groups, slot_patch_size, dim).flatten(1, 2)


def legacy_downsample_coords(
    pos: torch.Tensor,
    features: torch.Tensor,
    area: torch.Tensor,
    down_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return pos[:, ::down_ratio], features[:, ::down_ratio], area[:, ::down_ratio]


class LegacyEncoderStage1D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        down_ratio: int | None,
        next_dim: int | None,
        cond_dim: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.down_ratio = None if down_ratio is None else int(down_ratio)
        self.downsample = self.down_ratio is not None and next_dim is not None
        self.blocks = nn.ModuleList(
            [
                LegacySwinBlock1D(
                    dim=dim,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    shift_size=0 if i % 2 == 0 else int(patch_size) // 2,
                    cond_dim=cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for i in range(int(depth))
            ]
        )
        if self.downsample:
            if int(patch_size) % self.down_ratio != 0:
                raise ValueError("patch_size must be divisible by down_ratio.")
            self.downsample_layer = LegacyPointCrossAttention(
                dim=dim,
                num_heads=num_heads,
                down_ratio=self.down_ratio,
                patch_size=int(patch_size),
                hidden_dim=dim * 4,
            )
            self.proj_down = nn.Linear(dim, int(next_dim))
        else:
            self.downsample_layer = None
            self.proj_down = None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        area: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, cond, routes)
        if not self.downsample:
            return x, pos, area

        pos_ds, x_ds, area_ds = legacy_downsample_coords(pos, x, area, self.down_ratio)
        x_ds = self.downsample_layer(x, x_ds)
        x_ds = self.proj_down(x_ds)
        return x_ds, pos_ds, area_ds


class LegacyUniField1210Encoder(nn.Module):
    """
    Encoder-only port of ~/zoujunhong/code/field/model/UniField_1210_2.py.

    It intentionally uses only xyz coordinates as input, matching the old
    DrivAerNet demo. The current QueryUniField wrapper then attaches a query
    decoder for volume-field prediction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, ...] = (256, 512, 1024, 2048),
        depths: tuple[int, ...] = (4, 4, 4, 4),
        num_heads: tuple[int, ...] = (2, 4, 8, 8),
        patch_sizes: tuple[int, ...] = (1024, 512, 256, 128),
        down_ratios: tuple[int, ...] = (4, 4, 4),
        cond_dim: int = 2,
        cond_dims: int | None = None,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        if cond_dims is not None:
            cond_dim = int(cond_dims)
        self.in_channels = int(in_channels)
        self.embed_dims = tuple(int(v) for v in embed_dims)
        self.depths = tuple(int(v) for v in depths)
        self.num_heads = tuple(int(v) for v in num_heads)
        self.patch_sizes = tuple(int(v) for v in patch_sizes)
        self.down_ratios = tuple(int(v) for v in down_ratios)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.memory_dim = int(self.embed_dims[-1] if memory_dim is None else memory_dim)
        if self.memory_dim != self.embed_dims[-1]:
            raise ValueError("LegacyUniField1210Encoder memory_dim must equal embed_dims[-1].")
        if not (len(self.embed_dims) == len(self.depths) == len(self.num_heads) == len(self.patch_sizes)):
            raise ValueError("embed_dims, depths, num_heads, and patch_sizes must have the same length.")
        if len(self.down_ratios) != len(self.embed_dims) - 1:
            raise ValueError("down_ratios must have len(embed_dims) - 1 entries.")

        self.input_proj = nn.Linear(self.in_channels, self.embed_dims[0])
        dpr = torch.linspace(0.0, float(drop_path_rate), sum(self.depths)).tolist()
        dpr_idx = 0
        stages = []
        for i, dim in enumerate(self.embed_dims):
            depth = self.depths[i]
            stages.append(
                LegacyEncoderStage1D(
                    dim=dim,
                    depth=depth,
                    num_heads=self.num_heads[i],
                    patch_size=self.patch_sizes[i],
                    down_ratio=self.down_ratios[i] if i < len(self.down_ratios) else None,
                    next_dim=self.embed_dims[i + 1] if i + 1 < len(self.embed_dims) else None,
                    cond_dim=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx + depth - 1] if depth > 0 else 0.0,
                )
            )
            dpr_idx += depth
        self.enc_stages = nn.ModuleList(stages)

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

    def forward(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        return_pyramid: bool = False,
        geometry_meta=None,
    ) -> dict[str, torch.Tensor]:
        del normals, flow_vec, return_pyramid, geometry_meta
        if pos.dim() != 3 or pos.shape[-1] < self.in_channels:
            raise ValueError(f"pos must have shape (B, N, C) with C >= {self.in_channels}.")
        cond_t = self._prepare_cond(pos, cond)
        routes_t = self._prepare_routes(pos, routes)
        x = self.input_proj(pos[..., : self.in_channels])
        points = pos[..., :3]
        if area is None:
            area_t = points.new_ones(points.shape[0], points.shape[1], 1)
        else:
            area_t = area.to(device=points.device, dtype=points.dtype)
            if area_t.dim() == 2:
                area_t = area_t.unsqueeze(-1)
            if area_t.shape[:2] != points.shape[:2]:
                raise ValueError("area must have shape (B, N) or (B, N, 1).")

        for stage in self.enc_stages:
            x, points, area_t = stage(x, points, area_t, cond_t, routes_t)

        log_area = torch.log(area_t.clamp_min(1e-12))
        return {
            "features": x,
            "points": points,
            "area": area_t,
            "log_area": log_area,
        }


__all__ = [
    "LegacyUniField1210Encoder",
    "LegacyEncoderStage1D",
    "LegacyPointCrossAttention",
    "LegacySwinBlock1D",
]
