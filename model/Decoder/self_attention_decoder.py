from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.attention import SelfAttentionBlock
from model.utils.ffn import LinearNorm


class SelfAttentionDecoder(nn.Module):
    supports_chunked_forward = False

    def __init__(
        self,
        memory_dim: int,
        query_dim: Optional[int] = None,
        out_channels: int = 3,
        depth: int = 4,
        num_heads: int = 8,
        cond_dim: int = 2,
        num_routes: int = 1,
        query_in_dim: int = 3,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_flow_ffn: bool = True,
        **_: object,
    ):
        super().__init__()
        query_dim = int(memory_dim if query_dim is None else query_dim)
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads.")
        self.query_in_dim = int(query_in_dim)
        self.cond_dim = int(cond_dim)
        self.memory_proj = nn.Identity() if memory_dim == query_dim else nn.Linear(memory_dim, query_dim)
        self.surface_pos_emb = LinearNorm(3, query_dim)
        self.query_pos_emb = LinearNorm(self.query_in_dim, query_dim)
        self.cond_emb = LinearNorm(cond_dim, query_dim) if cond_dim > 0 else None
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=query_dim,
                    num_heads=num_heads,
                    cond_dim=cond_dim,
                    num_routes=num_routes,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    use_adaln=use_flow_ffn,
                )
                for i in range(depth)
            ]
        )
        self.out = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, out_channels))

    def _query_embedding_input(self, query_pos: torch.Tensor) -> torch.Tensor:
        query_xyz = query_pos[..., :3]
        if self.query_in_dim == 3:
            return query_xyz
        if query_pos.shape[-1] != self.query_in_dim:
            raise ValueError(f"query_pos must have shape (B, N, {self.query_in_dim}).")
        return torch.cat([query_xyz, query_pos[..., 3:]], dim=-1)

    def _cond_emb(self, cond: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if self.cond_emb is None:
            return like.new_zeros(like.shape[0], 1, like.shape[-1])
        if cond is None:
            cond = like.new_zeros(like.shape[0], self.cond_dim)
        return self.cond_emb(cond.to(device=like.device, dtype=like.dtype)).unsqueeze(1)

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del routes, flow
        surface = self.memory_proj(memory["features"]) + self.surface_pos_emb(memory["points"])
        surface = surface + self._cond_emb(cond, surface)
        return {"surface": surface}

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        del flow
        query = self.query_pos_emb(self._query_embedding_input(query_pos))
        query = query + self._cond_emb(cond, query)
        query_count = query.shape[1]
        tokens = torch.cat([prepared["surface"], query], dim=1)
        for block in self.blocks:
            tokens = block(tokens, cond, routes)
        return self.out(tokens[:, -query_count:])

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


__all__ = ["SelfAttentionDecoder"]
