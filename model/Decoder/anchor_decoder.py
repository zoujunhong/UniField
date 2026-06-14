from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.attention import CrossAttentionBlock, SelfAttentionBlock
from model.utils.ffn import LinearNorm


class AnchorDecoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        query_dim: Optional[int] = None,
        out_channels: int = 3,
        cond_dim: int = 2,
        num_routes: int = 1,
        query_in_dim: int = 3,
        num_anchors: int = 512,
        anchor_depth: int = 2,
        anchor_heads: int = 8,
        query_depth: int = 2,
        query_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_flow_ffn: bool = True,
    ):
        super().__init__()
        query_dim = int(memory_dim if query_dim is None else query_dim)
        query_heads = anchor_heads if query_heads is None else query_heads
        if query_dim % anchor_heads != 0 or query_dim % query_heads != 0:
            raise ValueError("query_dim must be divisible by anchor_heads and query_heads.")
        self.query_in_dim = int(query_in_dim)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.num_anchors = int(num_anchors)
        self.memory_proj = nn.Identity() if memory_dim == query_dim else nn.Linear(memory_dim, query_dim)
        self.surface_pos_emb = LinearNorm(3, query_dim)
        self.anchor_pos_emb = LinearNorm(3, query_dim)
        self.anchor_cond_emb = LinearNorm(cond_dim, query_dim) if cond_dim > 0 else None
        self.query_pos_emb = LinearNorm(self.query_in_dim, query_dim)
        self.query_cond_emb = LinearNorm(cond_dim, query_dim) if cond_dim > 0 else None
        self.anchor_pos = nn.Parameter(torch.empty(num_anchors, 3))
        self.anchor_embed = nn.Parameter(torch.empty(num_anchors, query_dim))

        total_depth = anchor_depth * 2 + query_depth
        dpr = torch.linspace(0.0, drop_path_rate, max(total_depth, 1)).tolist()
        dpr_idx = 0
        self.anchor_surface_blocks = nn.ModuleList()
        self.anchor_self_blocks = nn.ModuleList()
        for _ in range(anchor_depth):
            self.anchor_surface_blocks.append(
                CrossAttentionBlock(
                    dim=query_dim,
                    num_heads=anchor_heads,
                    cond_dim=cond_dim,
                    num_routes=num_routes,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    use_adaln=use_flow_ffn,
                )
            )
            dpr_idx += 1
            self.anchor_self_blocks.append(
                SelfAttentionBlock(
                    dim=query_dim,
                    num_heads=anchor_heads,
                    cond_dim=cond_dim,
                    num_routes=num_routes,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    use_adaln=use_flow_ffn,
                )
            )
            dpr_idx += 1
        self.query_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=query_dim,
                    num_heads=query_heads,
                    cond_dim=cond_dim,
                    num_routes=num_routes,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[dpr_idx + i],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    use_adaln=use_flow_ffn,
                )
                for i in range(query_depth)
            ]
        )
        self.out = nn.Sequential(LinearNorm(query_dim, query_dim), nn.Linear(query_dim, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.anchor_pos, -1.0, 1.0)
        nn.init.normal_(self.anchor_embed, std=0.02)

    def _cond_emb(self, emb: Optional[nn.Module], cond: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if emb is None:
            return like.new_zeros(like.shape[0], 1, like.shape[-1])
        if cond is None:
            cond = like.new_zeros(like.shape[0], self.cond_dim)
        return emb(cond.to(device=like.device, dtype=like.dtype)).unsqueeze(1)

    def _query_embedding_input(self, query_pos: torch.Tensor) -> torch.Tensor:
        query_xyz = query_pos[..., :3]
        if self.query_in_dim == 3:
            return query_xyz
        if query_pos.shape[-1] != self.query_in_dim:
            raise ValueError(f"query_pos must have shape (B, N, {self.query_in_dim}).")
        return torch.cat([query_xyz, query_pos[..., 3:]], dim=-1)

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del flow
        surface = self.memory_proj(memory["features"]) + self.surface_pos_emb(memory["points"])
        batch = surface.shape[0]
        anchor_pos = self.anchor_pos.to(device=surface.device, dtype=surface.dtype).unsqueeze(0).expand(batch, -1, -1)
        anchors = self.anchor_embed.to(device=surface.device, dtype=surface.dtype).unsqueeze(0).expand(batch, -1, -1)
        anchors = anchors + self.anchor_pos_emb(anchor_pos) + self._cond_emb(self.anchor_cond_emb, cond, anchors)
        for surface_block, self_block in zip(self.anchor_surface_blocks, self.anchor_self_blocks):
            anchors = surface_block(anchors, surface, cond, routes)
            anchors = self_block(anchors, cond, routes)
        return {"tokens": anchors}

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        del flow
        query_x = self.query_pos_emb(self._query_embedding_input(query_pos))
        query_x = query_x + self._cond_emb(self.query_cond_emb, cond, query_x)
        for block in self.query_blocks:
            query_x = block(query_x, prepared["tokens"], cond, routes)
        return self.out(query_x)

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


__all__ = ["AnchorDecoder"]
