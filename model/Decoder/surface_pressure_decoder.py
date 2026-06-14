from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.legacy_point import knn_interpolation


class SurfacePressureDecoder(nn.Module):
    """Surface-only pressure query decoder for AdaField/UniFieldV1 backbones."""

    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        out_channels: int = 1,
        interp_k: int = 3,
        query_in_dim: int = 3,
        **_: object,
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.out_channels = int(out_channels)
        self.interp_k = int(interp_k)
        self.query_in_dim = int(query_in_dim)
        self.proj_out = nn.Linear(self.memory_dim, self.out_channels)

    def _query_xyz(self, query_pos: torch.Tensor) -> torch.Tensor:
        if query_pos.dim() != 3 or query_pos.shape[-1] < 3:
            raise ValueError("query_pos must have shape (B, N, C) with C >= 3.")
        return query_pos[..., :3]

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cond, routes, flow
        return {
            "features": memory["features"],
            "points": memory["points"],
        }

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
        query_feat = knn_interpolation(
            query_xyz,
            prepared["points"],
            prepared["features"],
            k=self.interp_k,
        )
        return self.proj_out(query_feat)

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


__all__ = ["SurfacePressureDecoder"]
