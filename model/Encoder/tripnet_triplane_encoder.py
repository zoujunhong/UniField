from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TripNetUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 96,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        channels = [int(base_channels) * int(mult) for mult in channel_multipliers]
        if not channels:
            raise ValueError("channel_multipliers must not be empty.")

        self.encoders = nn.ModuleList()
        current = int(in_channels)
        for channel in channels:
            self.encoders.append(ConvBlock(current, channel))
            current = channel

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        current = channels[-1] * 2
        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for skip_channel in reversed(channels):
            self.up_projs.append(nn.Conv2d(current, skip_channel, kernel_size=1))
            self.decoders.append(ConvBlock(skip_channel * 2, skip_channel))
            current = skip_channel
        self.out = nn.Conv2d(current, int(out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)
        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up_proj(x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.out(x)


class TripNetTriplaneEncoder(nn.Module):
    """TripNet-style 2D U-Net backbone over raw xy/yz/xz triplanes."""

    def __init__(
        self,
        triplane_channels: int = 32,
        resolution: int = 128,
        feature_channels: int | None = None,
        base_channels: int = 96,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 2,
        num_routes: int = 1,
        memory_dim: int | None = None,
    ):
        super().__init__()
        self.triplane_channels = int(triplane_channels)
        self.resolution = int(resolution)
        self.feature_channels = int(self.triplane_channels if feature_channels is None else feature_channels)
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.memory_dim = int(3 * self.feature_channels if memory_dim is None else memory_dim)
        if self.num_routes < 1:
            raise ValueError("num_routes must be >= 1.")
        if self.memory_dim != 3 * self.feature_channels:
            raise ValueError("TripNetTriplaneEncoder memory_dim must equal 3 * feature_channels.")

        in_channels = 3 * self.triplane_channels
        out_channels = 3 * self.feature_channels
        self.backbone = TripNetUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
        )

    def _prepare_cond(self, pos: torch.Tensor, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.cond_dim <= 0:
            return None
        if cond is None:
            return pos.new_zeros(pos.shape[0], self.cond_dim)
        if cond.shape[0] != pos.shape[0] or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}).")
        return cond.to(device=pos.device, dtype=pos.dtype)

    def _prepare_routes(
        self,
        pos: torch.Tensor,
        routes: Optional[torch.Tensor | int],
    ) -> torch.Tensor:
        if routes is None:
            return torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        if isinstance(routes, int):
            return torch.full((pos.shape[0],), routes, dtype=torch.long, device=pos.device)
        routes_t = routes.to(device=pos.device, dtype=torch.long)
        if routes_t.dim() == 0:
            routes_t = routes_t.expand(pos.shape[0])
        return routes_t.view(-1)

    def _prepare_flow(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor],
        flow_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del cond
        if flow_vec is not None:
            flow = flow_vec.to(device=pos.device, dtype=pos.dtype)
            if flow.shape != (pos.shape[0], 3):
                raise ValueError("flow_vec must have shape (B, 3).")
            return F.normalize(flow, dim=-1)
        default = pos.new_tensor([1.0, 0.0, 0.0]).view(1, 3)
        return default.expand(pos.shape[0], -1)

    @staticmethod
    def _metadata_tensor(
        geometry_meta: Mapping[str, torch.Tensor] | None,
        key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if geometry_meta is None or key not in geometry_meta:
            raise ValueError(f"TripNetTriplaneEncoder requires geometry_meta[{key!r}].")
        value = geometry_meta[key]
        if not torch.is_tensor(value):
            raise TypeError(f"geometry_meta[{key!r}] must be a tensor.")
        return value.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        return_pyramid: bool = False,
        geometry_meta: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del cond, normals, area, flow_vec, routes, return_pyramid
        if pos.dim() != 5 or pos.shape[1] != 3:
            raise ValueError("TripNet triplane input must have shape (B, 3, C, H, W).")
        if pos.shape[2] != self.triplane_channels:
            raise ValueError(f"Expected {self.triplane_channels} triplane channels.")
        if pos.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(f"Expected triplane resolution {self.resolution}x{self.resolution}.")

        batch = pos.shape[0]
        x = pos.reshape(batch, 3 * self.triplane_channels, self.resolution, self.resolution)
        refined = self.backbone(x).view(batch, 3, self.feature_channels, self.resolution, self.resolution)
        bounds = self._metadata_tensor(geometry_meta, "bounds", pos)
        if bounds.dim() == 2 and bounds.shape[-1] == 6:
            bounds = bounds.view(batch, 2, 3)
        if bounds.shape != (batch, 2, 3):
            raise ValueError("geometry_meta['bounds'] must have shape (B, 2, 3) or (B, 6).")
        pooled = refined.flatten(3).mean(dim=-1).reshape(batch, 1, 3 * self.feature_channels)
        return {
            "planes": refined,
            "bounds": bounds,
            "features": pooled,
            "points": pos.new_zeros(batch, 1, 3),
            "area": pos.new_ones(batch, 1, 1),
            "log_area": pos.new_zeros(batch, 1, 1),
        }


__all__ = ["TripNetTriplaneEncoder", "TripNetUNet"]
