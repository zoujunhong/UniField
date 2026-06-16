from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from model.Encoder.flow_surface_encoder import UniFieldEncoder
from model.utils.checkpoint import load_route_compatible_state_dict


class QueryUniField(nn.Module):
    """Composable surface-encoder + query-decoder UniField."""

    def __init__(
        self,
        encoder: UniFieldEncoder,
        decoder: nn.Module,
        default_query_chunk_size: int = 50_000,
        coefficient_head_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.default_query_chunk_size = int(default_query_chunk_size)
        self.coefficient_head = None
        if coefficient_head_kwargs:
            self.coefficient_head = CoefficientHead(**dict(coefficient_head_kwargs))

    def _pack_output(self, query_pred: torch.Tensor, memory: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.coefficient_head is None:
            return query_pred
        return {
            "query": query_pred,
            "coefficients": self.coefficient_head(memory),
        }

    def encode_surface(
        self,
        surface_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        cond_t = self.encoder._prepare_cond(surface_pos, cond)
        routes_t = self.encoder._prepare_routes(surface_pos, routes)
        flow = self.encoder._prepare_flow(surface_pos, cond_t, flow_vec)
        memory = self.encoder(
            surface_pos,
            cond=cond_t,
            normals=normals,
            area=area,
            flow_vec=flow,
            routes=routes_t,
            return_pyramid=False,
        )
        return memory, cond_t, routes_t, flow

    def forward(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        memory, cond_t, routes_t, flow = self.encode_surface(
            surface_pos,
            cond=cond,
            routes=routes,
            normals=normals,
            area=area,
            flow_vec=flow_vec,
        )
        query_pred = self.decoder(query_pos, memory, cond_t, routes_t, flow)
        return self._pack_output(query_pred, memory)

    @torch.no_grad()
    def forward_test(
        self,
        surface_pos: torch.Tensor,
        query_pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not getattr(self.decoder, "supports_chunked_forward", False):
            raise NotImplementedError(
                f"{self.decoder.__class__.__name__} uses query-query interaction and does not support "
                "chunk-equivalent forward_test."
            )
        query_chunk_size = self.default_query_chunk_size if query_chunk_size is None else int(query_chunk_size)
        memory, cond_t, routes_t, flow = self.encode_surface(
            surface_pos,
            cond=cond,
            routes=routes,
            normals=normals,
            area=area,
            flow_vec=flow_vec,
        )
        prepared = self.decoder.prepare_memory(memory, cond_t, routes_t, flow)
        outputs = []
        for start in range(0, query_pos.shape[1], query_chunk_size):
            query_chunk = query_pos[:, start : start + query_chunk_size]
            outputs.append(
                self.decoder.decode_with_memory(query_chunk, prepared, cond_t, routes_t, flow)
            )
        query_pred = torch.cat(outputs, dim=1)
        return self._pack_output(query_pred, memory)

    def load_route_compatible_state_dict(
        self,
        checkpoint: Mapping[str, Any],
        route_map: Optional[Sequence[int]] = None,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        return load_route_compatible_state_dict(self, checkpoint, route_map=route_map, strict=strict)


class CoefficientHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int = 4,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(in_dim if hidden_dim is None else hidden_dim)
        layers: list[nn.Module] = [nn.LayerNorm(int(in_dim))]
        dim = int(in_dim)
        for _ in range(max(int(depth) - 1, 0)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = hidden_dim
        layers.append(nn.Linear(dim, int(out_channels)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, memory: dict[str, torch.Tensor]) -> torch.Tensor:
        features = memory["features"]
        pooled = features.mean(dim=1)
        return self.mlp(pooled)


__all__ = ["QueryUniField", "CoefficientHead"]
