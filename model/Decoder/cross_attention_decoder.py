from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.utils.attention import QuerySurfaceAttention
from model.utils.ffn import FFN, LinearNorm, RoutedAdaLN, apply_norm
from timm.layers.drop import DropPath


class QueryCrossBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        num_routes: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_flow_attention: bool = True,
        use_flow_ffn: bool = True,
    ):
        super().__init__()
        self.norm_query = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.norm_surface = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.attn = QuerySurfaceAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_flow_attention=use_flow_attention,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_ffn = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.ffn = FFN(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        query_x: torch.Tensor,
        surface_x: torch.Tensor,
        query_pos: torch.Tensor,
        surface_pos: torch.Tensor,
        surface_log_area: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        query_x = query_x + self.drop_path1(
            self.attn(
                apply_norm(self.norm_query, query_x, cond, routes),
                apply_norm(self.norm_surface, surface_x, cond, routes),
                query_pos,
                surface_pos,
                surface_log_area,
                flow,
            )
        )
        query_x = query_x + self.drop_path2(
            self.ffn(apply_norm(self.norm_ffn, query_x, cond, routes))
        )
        return query_x


class CrossAttentionDecoder(nn.Module):
    supports_chunked_forward = True

    def __init__(
        self,
        memory_dim: int,
        query_dim: Optional[int] = None,
        out_channels: int = 7,
        depth: int = 2,
        num_heads: int = 8,
        cond_dim: int = 2,
        num_routes: int = 1,
        query_in_dim: int = 3,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_flow_attention: bool = True,
        use_flow_ffn: bool = True,
    ):
        super().__init__()
        query_dim = int(memory_dim if query_dim is None else query_dim)
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads.")
        self.query_dim = query_dim
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.query_in_dim = int(query_in_dim)
        if self.query_in_dim < 3:
            raise ValueError("query_in_dim must be >= 3.")

        self.query_pos_emb = LinearNorm(self.query_in_dim, query_dim)
        self.memory_proj = nn.Identity() if memory_dim == query_dim else nn.Linear(memory_dim, query_dim)
        dpr = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                QueryCrossBlock(
                    dim=query_dim,
                    num_heads=num_heads,
                    cond_dim=self.cond_dim,
                    num_routes=self.num_routes,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    use_flow_attention=use_flow_attention,
                    use_flow_ffn=use_flow_ffn,
                )
                for i in range(depth)
            ]
        )
        self.out = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, out_channels))

    @staticmethod
    def _query_xyz(query_pos: torch.Tensor) -> torch.Tensor:
        if query_pos.shape[-1] < 3:
            raise ValueError("query_pos must have at least 3 coordinate channels.")
        return query_pos[..., :3]

    def _query_embedding_input(self, query_pos: torch.Tensor) -> torch.Tensor:
        query_xyz = self._query_xyz(query_pos)
        if self.query_in_dim == 3:
            return query_xyz
        if query_pos.shape[-1] != self.query_in_dim:
            raise ValueError(f"query_pos must have shape (B, N, {self.query_in_dim}).")
        return torch.cat([query_xyz, query_pos[..., 3:]], dim=-1)

    @staticmethod
    def _make_log_area(memory: dict[str, torch.Tensor]) -> torch.Tensor:
        if "log_area" in memory:
            log_area = memory["log_area"]
            return log_area.unsqueeze(-1) if log_area.dim() == 2 else log_area
        if "area" not in memory:
            points = memory["points"]
            return points.new_zeros(points.shape[0], points.shape[1], 1)
        area = memory["area"]
        area = area.unsqueeze(-1) if area.dim() == 2 else area
        return torch.log(area.clamp_min(1e-12))

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del cond, routes, flow
        return {
            "tokens": self.memory_proj(memory["features"]),
            "points": memory["points"],
            "log_area": self._make_log_area(memory),
        }

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        query_xyz = self._query_xyz(query_pos)
        query_x = self.query_pos_emb(self._query_embedding_input(query_pos))
        for block in self.blocks:
            query_x = block(
                query_x,
                prepared["tokens"],
                query_xyz,
                prepared["points"],
                prepared["log_area"],
                cond,
                routes,
                flow,
            )
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


__all__ = ["CrossAttentionDecoder", "QueryCrossBlock"]
