from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.Decoder.cross_attention_decoder import CrossAttentionDecoder
from model.Decoder.unifield_v2_decoder import UniFieldV2Decoder


class HybridSurfaceVolumeDecoder(nn.Module):
    """UniFieldV2 surface U-Net decoder plus volume cross-attention decoder."""

    supports_chunked_forward = True
    requires_full_query_forward = True

    def __init__(
        self,
        memory_dim: int,
        surface_decoder_kwargs: dict | None = None,
        volume_decoder_kwargs: dict | None = None,
        output_channels: int = 4,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if out_channels is not None:
            output_channels = int(out_channels)
        surface_kwargs = dict(surface_decoder_kwargs or {})
        volume_kwargs = dict(volume_decoder_kwargs or {})
        surface_kwargs.setdefault("memory_dim", memory_dim)
        surface_kwargs.setdefault("out_channels", 1)
        volume_kwargs.setdefault("memory_dim", memory_dim)
        volume_kwargs.setdefault("out_channels", output_channels)
        volume_kwargs.setdefault("query_in_dim", 3)

        self.surface_decoder = UniFieldV2Decoder(**surface_kwargs)
        self.volume_decoder = CrossAttentionDecoder(**volume_kwargs)
        self.output_channels = int(output_channels)
        self.surface_channels = int(surface_kwargs["out_channels"])
        self.volume_channels = int(volume_kwargs["out_channels"])
        if self.surface_channels > self.output_channels:
            raise ValueError("surface out_channels cannot exceed hybrid output_channels.")
        if self.volume_channels != self.output_channels:
            raise ValueError("volume out_channels must equal hybrid output_channels.")

    @staticmethod
    def _surface_count(memory: dict[str, torch.Tensor]) -> int:
        poss_skip = memory.get("poss_skip")
        if isinstance(poss_skip, list) and poss_skip:
            return int(poss_skip[0].shape[1])
        return int(memory["points"].shape[1])

    def prepare_memory(
        self,
        memory: dict[str, torch.Tensor],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> dict[str, object]:
        return {
            "surface": self.surface_decoder.prepare_memory(memory, cond, routes, flow),
            "volume": self.volume_decoder.prepare_memory(memory, cond, routes, flow),
            "surface_count": self._surface_count(memory),
        }

    def decode_with_memory(
        self,
        query_pos: torch.Tensor,
        prepared: dict[str, object],
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        surface_count = int(prepared["surface_count"])
        if query_pos.shape[1] < surface_count:
            raise ValueError(
                "HybridSurfaceVolumeDecoder requires the full concatenated "
                "surface+volume query because the split is inferred from the "
                "original surface point count."
            )

        surface_query = query_pos[:, :surface_count, :3]
        volume_query = query_pos[:, surface_count:, :3]
        surface_pred = self.surface_decoder.decode_with_memory(
            surface_query,
            prepared["surface"],
            cond,
            routes,
            flow,
        )

        surface_out = surface_pred.new_zeros(surface_pred.shape[0], surface_pred.shape[1], self.output_channels)
        surface_out[..., : self.surface_channels] = surface_pred
        if volume_query.shape[1] == 0:
            return surface_out

        volume_pred = self.volume_decoder.decode_with_memory(
            volume_query,
            prepared["volume"],
            cond,
            routes,
            flow,
        )
        return torch.cat([surface_out, volume_pred], dim=1)

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


__all__ = ["HybridSurfaceVolumeDecoder"]
