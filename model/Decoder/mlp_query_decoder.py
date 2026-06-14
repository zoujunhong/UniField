from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.ffn import LinearNorm


class MLPQueryDecoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        query_dim: Optional[int] = None,
        out_channels: int = 3,
        cond_dim: int = 2,
        query_in_dim: int = 3,
        depth: int = 2,
        proj_drop: float = 0.0,
        **_: object,
    ):
        super().__init__()
        query_dim = int(memory_dim if query_dim is None else query_dim)
        self.cond_dim = int(cond_dim)
        self.query_in_dim = int(query_in_dim)
        self.memory_proj = nn.Identity() if memory_dim == query_dim else nn.Linear(memory_dim, query_dim)
        self.query_pos_emb = LinearNorm(self.query_in_dim, query_dim)
        self.query_cond_emb = LinearNorm(cond_dim, query_dim) if cond_dim > 0 else None
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(LinearNorm(query_dim, query_dim))
            if proj_drop > 0.0:
                layers.append(nn.Dropout(proj_drop))
        layers.append(nn.Linear(query_dim, out_channels))
        self.query_mlp = nn.Sequential(*layers)

    def _query_embedding_input(self, query_pos: torch.Tensor) -> torch.Tensor:
        query_xyz = query_pos[..., :3]
        if self.query_in_dim == 3:
            return query_xyz
        if query_pos.shape[-1] != self.query_in_dim:
            raise ValueError(f"query_pos must have shape (B, N, {self.query_in_dim}).")
        return torch.cat([query_xyz, query_pos[..., 3:]], dim=-1)

    def _cond_emb(self, cond: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if self.query_cond_emb is None:
            return like.new_zeros(like.shape[0], 1, like.shape[-1])
        if cond is None:
            cond = like.new_zeros(like.shape[0], self.cond_dim)
        return self.query_cond_emb(cond.to(device=like.device, dtype=like.dtype)).unsqueeze(1)

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cond, routes, flow
        pooled = self.memory_proj(memory["features"]).mean(dim=1, keepdim=True)
        return {"pooled": pooled}

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        del routes, flow
        query_x = prepared["pooled"] + self.query_pos_emb(self._query_embedding_input(query_pos))
        query_x = query_x + self._cond_emb(cond, query_x)
        return self.query_mlp(query_x)

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


__all__ = ["MLPQueryDecoder"]
