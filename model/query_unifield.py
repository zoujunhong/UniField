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
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.default_query_chunk_size = int(default_query_chunk_size)

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
    ) -> torch.Tensor:
        memory, cond_t, routes_t, flow = self.encode_surface(
            surface_pos,
            cond=cond,
            routes=routes,
            normals=normals,
            area=area,
            flow_vec=flow_vec,
        )
        return self.decoder(query_pos, memory, cond_t, routes_t, flow)

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
    ) -> torch.Tensor:
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
        return torch.cat(outputs, dim=1)

    def load_route_compatible_state_dict(
        self,
        checkpoint: Mapping[str, Any],
        route_map: Optional[Sequence[int]] = None,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        return load_route_compatible_state_dict(self, checkpoint, route_map=route_map, strict=strict)


__all__ = ["QueryUniField"]
