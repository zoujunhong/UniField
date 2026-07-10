from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.gaussian3d import gaussian_rbf_aggregate


class Gaussian3DQueryDecoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        out_channels: int = 7,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        knn: int = 32,
        max_query_chunk: int = 0,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.knn = int(knn)
        self.max_query_chunk = int(max_query_chunk)

        layers: list[nn.Module] = []
        dim = self.memory_dim
        for _ in range(max(self.depth - 1, 0)):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, self.out_channels))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _query_xyz(query_pos: torch.Tensor) -> torch.Tensor:
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
            "means": memory["means"],
            "log_scales": memory["log_scales"],
            "rotations": memory["rotations"],
            "opacity_logits": memory["opacity_logits"],
            "features": memory["features"],
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
        features = gaussian_rbf_aggregate(
            query_xyz,
            prepared["means"],
            prepared["log_scales"],
            prepared["rotations"],
            prepared["opacity_logits"],
            prepared["features"],
            k=self.knn,
            max_query_chunk=self.max_query_chunk,
        )
        return self.mlp(features)

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


__all__ = ["Gaussian3DQueryDecoder"]
