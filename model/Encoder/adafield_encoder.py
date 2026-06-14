from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.legacy_point import LinearNorm, PointSlotAttention, PointTransformerLayer, knn_interpolation, modulate


class AdaFieldEncoder(nn.Module):
    """AdaField SAPT backbone converted to the current encoder-memory interface."""

    def __init__(
        self,
        in_channels: int = 3,
        depth: tuple[int, ...] | list[int] = (4, 4, 6, 12, 8),
        channels: tuple[int, ...] | list[int] = (64, 128, 256, 512, 1024),
        num_points: tuple[int, ...] | list[int] = (1024, 256, 64, 16),
        adapter_dim: int = 64,
        cond_dim: int = 2,
        cond_dims: Optional[int] = None,
        k: int = 16,
        head_dim: int = 64,
        default_flow_vec: tuple[float, float, float] = (1.0, 0.0, 0.0),
        flow_from_cond: bool = False,
        **_: object,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("AdaFieldEncoder expects xyz input with 3 channels.")
        if cond_dims is not None:
            cond_dim = int(cond_dims)
        self.depth = tuple(int(v) for v in depth)
        self.num_points = tuple(int(v) for v in num_points)
        self.channels = tuple(int(v) for v in channels)
        self.cond_dim = int(cond_dim)
        self.num_routes = 1
        self.k = int(k)
        self.head_dim = int(head_dim)
        self.flow_from_cond = bool(flow_from_cond)
        self.memory_dim = int(self.channels[0])
        if len(self.channels) != len(self.depth):
            raise ValueError("channels and depth must have the same length.")
        if len(self.num_points) != len(self.depth) - 1:
            raise ValueError("num_points length must be len(depth) - 1.")

        self.register_buffer(
            "default_flow_vec",
            torch.tensor(default_flow_vec, dtype=torch.float32).view(1, 3),
            persistent=False,
        )

        self.proj_in = nn.Linear(3, self.channels[0])
        self.proj_down_layers = nn.ModuleList()
        self.proj_up_layers = nn.ModuleList()
        self.tf_down_layers = nn.ModuleList()
        self.tf_up_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.adapt_down_layers = nn.ModuleList()
        self.adapt_up_layers = nn.ModuleList()
        self.adapt_down_proj_down_layers = nn.ModuleList()
        self.adapt_down_proj_up_layers = nn.ModuleList()
        self.adapt_up_proj_down_layers = nn.ModuleList()
        self.adapt_up_proj_up_layers = nn.ModuleList()

        for i, stage_depth in enumerate(self.depth):
            layers = nn.ModuleList()
            for _ in range(stage_depth):
                layers.append(
                    PointTransformerLayer(
                        self.channels[i],
                        num_heads=max(self.channels[i] // self.head_dim, 1),
                        k=self.k,
                    )
                )
                self.adapt_down_proj_down_layers.append(LinearNorm(self.channels[i], adapter_dim))
                self.adapt_down_layers.append(
                    nn.Sequential(
                        LinearNorm(self.cond_dim, 2 * adapter_dim),
                        nn.Linear(2 * adapter_dim, 2 * adapter_dim),
                    )
                )
                self.adapt_down_proj_up_layers.append(LinearNorm(adapter_dim, self.channels[i]))
            self.tf_down_layers.append(layers)

            if i < len(self.depth) - 1:
                self.proj_down_layers.append(nn.Linear(self.channels[i], self.channels[i + 1]))
                self.proj_up_layers.append(nn.Linear(self.channels[-(i + 1)], self.channels[-(i + 2)]))
                layers = nn.ModuleList()
                for _ in range(stage_depth):
                    layers.append(
                        PointTransformerLayer(
                            self.channels[i],
                            num_heads=max(self.channels[i] // self.head_dim, 1),
                            k=self.k,
                        )
                    )
                    self.adapt_up_proj_down_layers.append(LinearNorm(self.channels[i], adapter_dim))
                    self.adapt_up_layers.append(
                        nn.Sequential(
                            LinearNorm(self.cond_dim, 2 * adapter_dim),
                            nn.Linear(2 * adapter_dim, 2 * adapter_dim),
                        )
                    )
                    self.adapt_up_proj_up_layers.append(LinearNorm(adapter_dim, self.channels[i]))
                self.tf_up_layers.append(layers)
                self.downsample_layers.append(
                    PointSlotAttention(
                        self.num_points[i],
                        self.channels[i],
                        hidden_dim=self.channels[i] * 4,
                        k=64 if i == 0 else None,
                    )
                )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _prepare_cond(self, pos: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_dim <= 0:
            return pos.new_zeros(pos.shape[0], 0)
        if cond is None:
            return pos.new_zeros(pos.shape[0], self.cond_dim)
        if cond.shape[0] != pos.shape[0] or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must have shape (B, {self.cond_dim}).")
        return cond.to(device=pos.device, dtype=pos.dtype)

    def _prepare_routes(self, pos: torch.Tensor, routes: Optional[torch.Tensor | int]) -> torch.Tensor:
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
        if flow_vec is not None:
            flow = flow_vec.to(device=pos.device, dtype=pos.dtype)
        elif self.flow_from_cond and cond is not None and cond.shape[-1] >= 2:
            flow = torch.cat([cond[:, :2], cond.new_zeros(cond.shape[0], 1)], dim=-1)
        else:
            flow = self.default_flow_vec.to(device=pos.device, dtype=pos.dtype).expand(pos.shape[0], -1)
        if flow.shape != (pos.shape[0], 3):
            raise ValueError("flow_vec must have shape (B, 3).")
        return F.normalize(flow, dim=-1)

    def forward(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        area: Optional[torch.Tensor] = None,
        flow_vec: Optional[torch.Tensor] = None,
        routes: Optional[torch.Tensor | int] = None,
        return_pyramid: bool = True,
    ) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        del normals, area, flow_vec, routes
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError("pos must have shape (B, N, 3).")
        cond_t = self._prepare_cond(pos, cond)
        x = self.proj_in(pos)
        x_list: list[torch.Tensor] = []
        pos_list: list[torch.Tensor] = []
        pyramid: list[dict[str, torch.Tensor]] = []

        count = 0
        for i, stage_depth in enumerate(self.depth):
            for j in range(stage_depth):
                shift, scale = self.adapt_down_layers[count](cond_t).chunk(2, dim=1)
                x = x + self.adapt_down_proj_up_layers[count](
                    modulate(self.adapt_down_proj_down_layers[count](x), shift, scale)
                )
                count += 1
                x = self.tf_down_layers[i][j](x, pos)

            x_list.append(x)
            pos_list.append(pos)
            if return_pyramid:
                pyramid.append(
                    {
                        "features": x,
                        "points": pos,
                        "normals": pos.new_zeros(pos.shape),
                        "area": pos.new_ones(pos.shape[0], pos.shape[1], 1),
                    }
                )

            if i < len(self.depth) - 1:
                x, pos = self.downsample_layers[i](x, pos)
                x = self.proj_down_layers[i](x)

        count = 0
        for i in range(len(self.depth) - 1):
            dst_pos = pos_list[-(i + 2)]
            src_pos = pos_list[-(i + 1)]
            x = knn_interpolation(dst_pos, src_pos, self.proj_up_layers[i](x), k=self.k) + x_list[-(i + 2)]
            for j in range(self.depth[-(i + 2)]):
                idx = -(count + 1)
                shift, scale = self.adapt_up_layers[idx](cond_t).chunk(2, dim=1)
                x = x + self.adapt_up_proj_up_layers[idx](
                    modulate(self.adapt_up_proj_down_layers[idx](x), shift, scale)
                )
                count += 1
                x = self.tf_up_layers[-(i + 1)][j](x, dst_pos)

        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "features": x,
            "points": pos_list[0],
            "normals": pos_list[0].new_zeros(pos_list[0].shape),
            "area": pos_list[0].new_ones(pos_list[0].shape[0], pos_list[0].shape[1], 1),
            "log_area": pos_list[0].new_zeros(pos_list[0].shape[0], pos_list[0].shape[1], 1),
            "flow": self._prepare_flow(pos_list[0], cond_t, None),
        }
        if return_pyramid:
            output["pyramid"] = pyramid
        return output


__all__ = ["AdaFieldEncoder"]
