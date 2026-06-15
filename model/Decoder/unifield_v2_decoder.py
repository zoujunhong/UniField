from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.Encoder.unifield_v2_encoder import SwinBlock1D
from model.utils.downsample import PointCrossAttention
from model.utils.legacy_point import knn_interpolation


class PTv3DecoderStage1D(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        num_heads: int,
        num_routes: int,
        patch_size: int,
        down_ratios: int,
        cond_dims: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.sample_patch_size_in = int(patch_size // down_ratios)
        self.sample_patch_size_out = int(patch_size)
        self.fuse = nn.Linear(dim_in, dim_out)
        self.blocks = nn.ModuleList(
            [
                SwinBlock1D(
                    dim=dim_out,
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
        self.upsample_layer = PointCrossAttention(
            dim_out,
            num_heads,
            down_ratios,
            patch_size,
            hidden_dim=dim_out * 4,
        )

    def forward(
        self,
        x_coarse: torch.Tensor,
        pos_coarse: torch.Tensor,
        x_skip: torch.Tensor,
        pos_skip: torch.Tensor,
        cond: torch.Tensor,
        routes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pos_coarse
        x = x_skip + self.upsample_layer(self.fuse(x_coarse), x_skip)
        pos = pos_skip
        for block in self.blocks:
            x = block(x, cond, routes)
        return x, pos


class UniFieldV2Decoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        decoder_depths: Optional[tuple[int, ...]] = None,
        num_heads: tuple[int, ...] = (4, 4, 8, 8),
        patch_sizes: tuple[int, ...] = (16, 16, 16, 16),
        down_ratios: tuple[int, ...] = (2, 2, 2),
        cond_dim: int = 2,
        cond_dims: Optional[int] = None,
        num_routes: int = 1,
        out_channels: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        interp_k: int = 3,
        **_: object,
    ):
        super().__init__()
        del memory_dim
        if cond_dims is not None:
            cond_dim = int(cond_dims)
        self.embed_dims = tuple(int(v) for v in embed_dims)
        self.depths = tuple(int(v) for v in depths)
        self.num_heads = tuple(int(v) for v in num_heads)
        self.patch_sizes = tuple(int(v) for v in patch_sizes)
        self.down_ratios = tuple(int(v) for v in down_ratios)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.interp_k = int(interp_k)
        self.num_stages = len(self.embed_dims)
        if decoder_depths is None:
            decoder_depths = tuple(reversed(self.depths[:-1]))
        self.decoder_depths = tuple(int(v) for v in decoder_depths)
        if len(self.decoder_depths) != self.num_stages - 1:
            raise ValueError("decoder_depths length must be num_stages - 1.")

        total_blocks = sum(self.depths) + sum(self.decoder_depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dpr_idx = sum(self.depths)
        dec_stages = []
        for dec_i, enc_i in enumerate(range(self.num_stages - 2, -1, -1)):
            depth = self.decoder_depths[dec_i]
            dec_stages.append(
                PTv3DecoderStage1D(
                    dim_in=self.embed_dims[enc_i + 1],
                    dim_out=self.embed_dims[enc_i],
                    depth=depth,
                    num_heads=self.num_heads[enc_i],
                    num_routes=self.num_routes,
                    patch_size=self.patch_sizes[enc_i],
                    down_ratios=self.down_ratios[enc_i],
                    cond_dims=self.cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx + depth - 1] if depth > 0 else 0.0,
                )
            )
            dpr_idx += depth
        self.dec_stages = nn.ModuleList(dec_stages)
        self.out_proj = nn.Linear(self.embed_dims[0], out_channels) if out_channels is not None else None

    @staticmethod
    def _query_xyz(query_pos: torch.Tensor) -> torch.Tensor:
        if query_pos.dim() != 3 or query_pos.shape[-1] < 3:
            raise ValueError("query_pos must have shape (B, N, C) with C >= 3.")
        return query_pos[..., :3]

    def _decode_surface(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if cond is None:
            features = memory["features"]
            cond = features.new_zeros(features.shape[0], self.cond_dim)
        x = memory["features"]
        pos = memory["points"]
        feats_skip = memory["feats_skip"]
        poss_skip = memory["poss_skip"]
        for dec_stage, enc_i in zip(self.dec_stages, range(self.num_stages - 2, -1, -1)):
            x, pos = dec_stage(x, pos, feats_skip[enc_i], poss_skip[enc_i], cond, routes)
        return {"features": x, "points": pos}

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del flow
        return self._decode_surface(memory, cond, routes)

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        del cond, routes, flow
        query_xyz = self._query_xyz(query_pos)
        features = prepared["features"]
        points = prepared["points"]
        if query_xyz.shape == points.shape and torch.equal(query_xyz, points):
            query_features = features
        else:
            query_features = knn_interpolation(query_xyz, points, features, k=self.interp_k)
        return query_features if self.out_proj is None else self.out_proj(query_features)

    def forward(
        self,
        query_pos: torch.Tensor,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.prepare_memory(memory, cond, routes, flow)
        return self.decode_with_memory(query_pos, prepared, cond, routes, flow)


__all__ = ["PTv3DecoderStage1D", "UniFieldV2Decoder"]
