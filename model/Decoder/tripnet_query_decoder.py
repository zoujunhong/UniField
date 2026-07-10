from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.triplane import sample_triplanes


class TripNetQueryDecoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        out_channels: int = 7,
        cond_dim: int = 2,
        hidden_dim: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        include_query_coords: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.out_channels = int(out_channels)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.include_query_coords = bool(include_query_coords)
        self.padding_mode = padding_mode

        in_dim = self.memory_dim + (3 if self.include_query_coords else 0) + max(self.cond_dim, 0)
        layers: list[nn.Module] = []
        dim = in_dim
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

    def _cond_features(
        self,
        cond: Optional[torch.Tensor],
        query_xyz: torch.Tensor,
    ) -> torch.Tensor:
        if self.cond_dim <= 0:
            return query_xyz.new_zeros(query_xyz.shape[0], query_xyz.shape[1], 0)
        if cond is None:
            cond_t = query_xyz.new_zeros(query_xyz.shape[0], self.cond_dim)
        else:
            cond_t = cond.to(device=query_xyz.device, dtype=query_xyz.dtype)
            if cond_t.shape != (query_xyz.shape[0], self.cond_dim):
                raise ValueError(f"cond must have shape (B, {self.cond_dim}).")
        return cond_t[:, None, :].expand(-1, query_xyz.shape[1], -1)

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cond, routes, flow
        return {
            "planes": memory["planes"],
            "bounds": memory["bounds"],
        }

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        del routes, flow
        query_xyz = self._query_xyz(query_pos)
        features, query_norm = sample_triplanes(
            prepared["planes"],
            query_xyz,
            prepared["bounds"],
            padding_mode=self.padding_mode,
        )
        inputs = [features]
        if self.include_query_coords:
            inputs.append(query_norm)
        if self.cond_dim > 0:
            inputs.append(self._cond_features(cond, query_xyz))
        return self.mlp(torch.cat(inputs, dim=-1))

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


__all__ = ["TripNetQueryDecoder"]
