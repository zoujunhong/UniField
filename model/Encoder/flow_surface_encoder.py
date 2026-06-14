from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath

from model.utils.attention import FlowManifoldAttention
from model.utils.checkpoint import load_route_compatible_state_dict
from model.utils.ffn import FFN, RoutedAdaLN, apply_norm
from model.utils.serialization import flow_aligned_morton3d_sort, flow_aware_sfc, morton3d_sort


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _expand_batch(x: torch.Tensor, groups: int) -> torch.Tensor:
    return x[:, None, :].expand(-1, groups, -1).reshape(x.shape[0] * groups, x.shape[-1])


def _expand_routes(routes: torch.Tensor, groups: int) -> torch.Tensor:
    return routes[:, None].expand(-1, groups).reshape(routes.shape[0] * groups)


@dataclass
class SurfaceState:
    x: torch.Tensor
    pos: torch.Tensor
    normals: torch.Tensor
    area: torch.Tensor
    log_area: torch.Tensor


class FlowManifoldBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        num_routes: int = 1,
        use_flow_attention: bool = True,
        use_flow_ffn: bool = True,
    ):
        super().__init__()
        self.norm1_q = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.norm1_kv = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.attn = FlowManifoldAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_flow_attention=use_flow_attention,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = RoutedAdaLN(dim, cond_dim, num_routes) if use_flow_ffn else nn.LayerNorm(dim)
        self.ffn = FFN(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        q_pos: torch.Tensor,
        kv_pos: torch.Tensor,
        q_normals: torch.Tensor,
        kv_normals: torch.Tensor,
        q_log_area: torch.Tensor,
        kv_log_area: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: Optional[torch.Tensor],
        flow: torch.Tensor,
    ) -> torch.Tensor:
        q_norm = apply_norm(self.norm1_q, q_x, cond, routes)
        if q_x is kv_x:
            kv_norm = q_norm
        else:
            kv_norm = apply_norm(self.norm1_kv, kv_x, cond, routes)
        q_x = q_x + self.drop_path1(
            self.attn(
                q_norm,
                kv_norm,
                q_pos,
                kv_pos,
                q_normals,
                kv_normals,
                q_log_area,
                kv_log_area,
                flow,
            )
        )
        q_x = q_x + self.drop_path2(
            self.ffn(apply_norm(self.norm2, q_x, cond, routes))
        )
        return q_x


class FlowEncoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        cond_dim: int,
        down_ratio: Optional[int],
        next_dim: Optional[int],
        drop_paths: list[float],
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        sort_bins: int = 32,
        sort_bits: int = 12,
        attention_order: str = "none",
        downsample_order: str = "none",
        num_routes: int = 1,
        use_flow_attention: bool = True,
        use_flow_ffn: bool = True,
    ):
        super().__init__()
        if len(drop_paths) != depth + (1 if down_ratio is not None and next_dim is not None else 0):
            raise ValueError("drop_paths length does not match stage depth.")
        self.window_size = int(window_size)
        self.down_ratio = down_ratio
        self.sort_bins = int(sort_bins)
        self.sort_bits = int(sort_bits)
        # The encoder normally sorts once at input. Stage-level sorting remains
        # available only for ablations such as flow_sfc/morton comparisons.
        self.attention_order = attention_order
        self.downsample_order = downsample_order
        self.blocks = nn.ModuleList(
            [
                FlowManifoldBlock(
                    dim=dim,
                    num_heads=num_heads,
                    cond_dim=cond_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_paths[i],
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    num_routes=num_routes,
                    use_flow_attention=use_flow_attention,
                    use_flow_ffn=use_flow_ffn,
                )
                for i in range(depth)
            ]
        )
        self.downsample = down_ratio is not None and next_dim is not None
        if self.downsample:
            if self.window_size % int(down_ratio) != 0:
                raise ValueError("window_size must be divisible by down_ratio.")
            self.down_block = FlowManifoldBlock(
                dim=dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                drop_path=drop_paths[-1],
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                num_routes=num_routes,
                use_flow_attention=use_flow_attention,
                use_flow_ffn=use_flow_ffn,
            )
            self.proj_down = nn.Linear(dim, next_dim)
        else:
            self.down_block = None
            self.proj_down = None

    def _sort_state(self, state: SurfaceState, flow: torch.Tensor, method: str) -> SurfaceState:
        if method in {"none", None}:
            return state
        if method == "flow_sfc":
            _, idx, _ = flow_aware_sfc(
                state.pos.float(),
                flow.float(),
                num_bins=self.sort_bins,
                bits=self.sort_bits,
                snake=True,
            )
        elif method == "flow_morton":
            _, idx, _ = flow_aligned_morton3d_sort(
                state.pos.float(),
                flow.float(),
                bits=20,
                ground_aligned=True,
            )
        elif method == "morton":
            _, idx, _ = morton3d_sort(state.pos.float(), bits=20)
        else:
            raise ValueError(f"Unsupported point order: {method}.")
        return SurfaceState(
            x=_gather_tokens(state.x, idx),
            pos=_gather_tokens(state.pos, idx),
            normals=_gather_tokens(state.normals, idx),
            area=_gather_tokens(state.area, idx),
            log_area=_gather_tokens(state.log_area, idx),
        )

    def _window_state(self, state: SurfaceState) -> tuple[SurfaceState, int, int, int]:
        b, n, _ = state.x.shape
        if n % self.window_size != 0:
            raise ValueError(
                f"Point count {n} must be divisible by window_size {self.window_size}."
            )
        groups = n // self.window_size
        return (
            SurfaceState(
                x=state.x.view(b * groups, self.window_size, -1),
                pos=state.pos.view(b * groups, self.window_size, -1),
                normals=state.normals.view(b * groups, self.window_size, -1),
                area=state.area.view(b * groups, self.window_size, -1),
                log_area=state.log_area.view(b * groups, self.window_size, -1),
            ),
            b,
            n,
            groups,
        )

    @staticmethod
    def _unwindow_state(state: SurfaceState, batch_size: int, groups: int) -> SurfaceState:
        return SurfaceState(
            x=state.x.view(batch_size, groups * state.x.shape[1], -1),
            pos=state.pos.view(batch_size, groups * state.pos.shape[1], -1),
            normals=state.normals.view(batch_size, groups * state.normals.shape[1], -1),
            area=state.area.view(batch_size, groups * state.area.shape[1], -1),
            log_area=state.log_area.view(batch_size, groups * state.log_area.shape[1], -1),
        )

    def _area_weighted_downsample(self, state: SurfaceState, has_area: bool) -> SurfaceState:
        ratio = int(self.down_ratio)
        b_windows, window, _ = state.x.shape
        out_window = window // ratio
        area = state.area.view(b_windows, out_window, ratio, 1)
        weights = area / area.sum(dim=2, keepdim=True).clamp_min(1e-12)

        def weighted_mean(value: torch.Tensor) -> torch.Tensor:
            value = value.view(b_windows, out_window, ratio, value.shape[-1])
            return (value * weights).sum(dim=2)

        x = weighted_mean(state.x)
        pos = weighted_mean(state.pos)
        normals = F.normalize(weighted_mean(state.normals), dim=-1)

        if has_area:
            area_out = area.sum(dim=2)
            log_area = torch.log(area_out.clamp_min(1e-12))
        else:
            area_out = state.area.new_ones(b_windows, out_window, 1)
            log_area = state.log_area.new_zeros(b_windows, out_window, 1)

        return SurfaceState(x=x, pos=pos, normals=normals, area=area_out, log_area=log_area)

    def forward(
        self,
        state: SurfaceState,
        cond: Optional[torch.Tensor],
        routes: torch.Tensor,
        flow: torch.Tensor,
        has_area: bool,
    ) -> tuple[SurfaceState, dict[str, torch.Tensor]]:
        state = self._sort_state(state, flow, self.attention_order)
        state_w, batch_size, _, groups = self._window_state(state)
        cond_w = _expand_batch(cond, groups) if cond is not None else None
        routes_w = _expand_routes(routes, groups)
        flow_w = _expand_batch(flow, groups)

        for block in self.blocks:
            state_w.x = block(
                state_w.x,
                state_w.x,
                state_w.pos,
                state_w.pos,
                state_w.normals,
                state_w.normals,
                state_w.log_area,
                state_w.log_area,
                cond_w,
                routes_w,
                flow_w,
            )

        stage_state = self._unwindow_state(state_w, batch_size, groups)
        stage_output = {
            "features": stage_state.x,
            "points": stage_state.pos,
            "normals": stage_state.normals,
            "area": stage_state.area,
        }

        if not self.downsample:
            return stage_state, stage_output

        down_state = self._sort_state(stage_state, flow, self.downsample_order)
        down_state_w, _, _, down_groups = self._window_state(down_state)
        cond_down = _expand_batch(cond, down_groups) if cond is not None else None
        routes_down = _expand_routes(routes, down_groups)
        flow_down = _expand_batch(flow, down_groups)

        anchor = self._area_weighted_downsample(down_state_w, has_area)
        anchor.x = self.down_block(
            anchor.x,
            down_state_w.x,
            anchor.pos,
            down_state_w.pos,
            anchor.normals,
            down_state_w.normals,
            anchor.log_area,
            down_state_w.log_area,
            cond_down,
            routes_down,
            flow_down,
        )
        anchor.x = self.proj_down(anchor.x)
        next_state = self._unwindow_state(anchor, batch_size, down_groups)
        return next_state, stage_output


class UniFieldEncoder(nn.Module):
    """
    Encoder-only UniField variant with unified flow-manifold attention.

    It produces a multiscale surface feature pyramid without reading VTK files
    directly. Optional normals and point areas can be supplied by a dataset/cache
    preprocessor; if they are absent, the manifold branches fall back to zeros
    and uniform downsampling weights. By default, the input is sorted once with
    Flow-Aligned Morton; later attention windows and downsampling anchors reuse
    that same order.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: tuple[int, ...] = (128, 256, 512, 1024),
        depths: tuple[int, ...] = (3, 4, 6, 3),
        num_heads: tuple[int, ...] = (2, 4, 8, 8),
        window_sizes: tuple[int, ...] = (1024, 512, 256, 128),
        down_ratios: tuple[int, ...] = (4, 4, 4),
        cond_dim: int = 2,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        sort_bins: int = 32,
        sort_bits: int = 12,
        input_order: str = "flow_morton",
        attention_order: str = "none",
        downsample_order: str = "none",
        num_routes: int = 1,
        use_flow_attention: bool = True,
        use_flow_ffn: bool = True,
        default_flow_vec: tuple[float, float, float] = (1.0, 0.0, 0.0),
        flow_from_cond: bool = False,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.num_routes = int(num_routes)
        self.use_flow_attention = bool(use_flow_attention)
        self.use_flow_ffn = bool(use_flow_ffn)
        if self.num_routes < 1:
            raise ValueError("num_routes must be >= 1.")
        self.flow_from_cond = bool(flow_from_cond)
        self.input_order = input_order
        self.attention_order = attention_order
        self.downsample_order = downsample_order
        self.sort_bins = int(sort_bins)
        self.sort_bits = int(sort_bits)
        self.num_stages = len(embed_dims)
        if in_channels != 3:
            raise ValueError("UniFieldEncoder v1 expects pos with exactly 3 coordinates.")
        if not (
            len(depths)
            == len(num_heads)
            == len(window_sizes)
            == self.num_stages
        ):
            raise ValueError("embed_dims, depths, num_heads, and window_sizes must have the same length.")
        if len(down_ratios) != self.num_stages - 1:
            raise ValueError("down_ratios length must be num_stages - 1.")

        self.register_buffer(
            "default_flow_vec",
            torch.tensor(default_flow_vec, dtype=torch.float32).view(1, 3),
            persistent=False,
        )

        stem_dim = 3 + 3 + 1 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(stem_dim, embed_dims[0]),
            nn.LayerNorm(embed_dims[0]),
            nn.GELU(),
        )

        total_blocks = sum(depths) + len(down_ratios)
        dpr = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        dpr_idx = 0
        stages = []
        for i in range(self.num_stages):
            down_ratio = down_ratios[i] if i < self.num_stages - 1 else None
            next_dim = embed_dims[i + 1] if i < self.num_stages - 1 else None
            stage_block_count = depths[i] + (1 if down_ratio is not None else 0)
            drop_paths = dpr[dpr_idx : dpr_idx + stage_block_count]
            dpr_idx += stage_block_count
            stages.append(
                FlowEncoderStage(
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    cond_dim=self.cond_dim,
                    down_ratio=down_ratio,
                    next_dim=next_dim,
                    drop_paths=drop_paths,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    sort_bins=sort_bins,
                    sort_bits=sort_bits,
                    attention_order=attention_order,
                    downsample_order=downsample_order,
                    num_routes=self.num_routes,
                    use_flow_attention=self.use_flow_attention,
                    use_flow_ffn=self.use_flow_ffn,
                )
            )
        self.stages = nn.ModuleList(stages)

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
            routes_t = torch.full((pos.shape[0],), routes, dtype=torch.long, device=pos.device)
        else:
            routes_t = routes.to(device=pos.device, dtype=torch.long)
            if routes_t.dim() == 0:
                routes_t = routes_t.expand(pos.shape[0])
            else:
                routes_t = routes_t.view(-1)
        # if routes_t.shape[0] != pos.shape[0]:
        #     raise ValueError("routes must have shape (B,), be scalar, or be None.")
        # if routes_t.min().item() < 0 or routes_t.max().item() >= self.num_routes:
        #     raise ValueError(f"routes must be in [0, {self.num_routes}).")
        return routes_t

    def _prepare_flow(
        self,
        pos: torch.Tensor,
        cond: Optional[torch.Tensor],
        flow_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if flow_vec is not None:
            flow = flow_vec.to(device=pos.device, dtype=pos.dtype)
            if flow.shape != (pos.shape[0], 3):
                raise ValueError("flow_vec must have shape (B, 3).")
        elif self.flow_from_cond and cond is not None and cond.shape[-1] >= 2:
            zeros = cond.new_zeros(cond.shape[0], 1)
            flow = torch.cat([cond[:, :2], zeros], dim=-1).to(device=pos.device, dtype=pos.dtype)
        else:
            flow = self.default_flow_vec.to(device=pos.device, dtype=pos.dtype).expand(pos.shape[0], -1)
        return F.normalize(flow, dim=-1)

    def _prepare_geometry(
        self,
        pos: torch.Tensor,
        normals: Optional[torch.Tensor],
        area: Optional[torch.Tensor],
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        if normals is None:
            normals_t = pos.new_zeros(pos.shape)
        else:
            if normals.shape != pos.shape:
                raise ValueError("normals must have shape (B, N, 3).")
            normals_t = F.normalize(normals.to(device=pos.device, dtype=pos.dtype), dim=-1)

        has_area = area is not None
        if area is None:
            area_t = pos.new_ones(pos.shape[0], pos.shape[1], 1)
            log_area = pos.new_zeros(pos.shape[0], pos.shape[1], 1)
        else:
            if area.dim() == 2:
                area = area.unsqueeze(-1)
            if area.shape[:2] != pos.shape[:2] or area.shape[-1] != 1:
                raise ValueError("area must have shape (B, N) or (B, N, 1).")
            area_t = area.to(device=pos.device, dtype=pos.dtype).clamp_min(1e-12)
            log_area = torch.log(area_t)

        incidence = (normals_t * flow[:, None, :]).sum(dim=-1, keepdim=True)
        stem_geom = torch.cat([pos, normals_t, log_area, incidence], dim=-1)
        return normals_t, area_t, log_area, has_area, stem_geom

    def _input_sort_idx(self, pos: torch.Tensor, flow: torch.Tensor) -> Optional[torch.Tensor]:
        if self.input_order in {"none", None}:
            return None
        if self.input_order == "flow_morton":
            _, idx, _ = flow_aligned_morton3d_sort(
                pos.float(),
                flow.float(),
                bits=20,
                ground_aligned=True,
            )
        elif self.input_order == "morton":
            _, idx, _ = morton3d_sort(pos.float(), bits=20)
        elif self.input_order == "flow_sfc":
            _, idx, _ = flow_aware_sfc(
                pos.float(),
                flow.float(),
                num_bins=self.sort_bins,
                bits=self.sort_bits,
                snake=True,
            )
        else:
            raise ValueError(f"Unsupported input_order: {self.input_order}.")
        return idx

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
        if pos.dim() != 3 or pos.shape[-1] != 3:
            raise ValueError("pos must have shape (B, N, 3).")
        cond_t = self._prepare_cond(pos, cond)
        routes_t = self._prepare_routes(pos, routes)
        flow = self._prepare_flow(pos, cond_t, flow_vec)
        normals_t, area_t, log_area, has_area, stem_input = self._prepare_geometry(
            pos,
            normals,
            area,
            flow,
        )
        sort_idx = self._input_sort_idx(pos, flow)
        if sort_idx is not None:
            pos = _gather_tokens(pos, sort_idx)
            normals_t = _gather_tokens(normals_t, sort_idx)
            area_t = _gather_tokens(area_t, sort_idx)
            log_area = _gather_tokens(log_area, sort_idx)
            stem_input = _gather_tokens(stem_input, sort_idx)

        x = self.input_proj(stem_input)

        state = SurfaceState(
            x=x,
            pos=pos,
            normals=normals_t,
            area=area_t,
            log_area=log_area,
        )
        pyramid = []
        for stage in self.stages:
            state, stage_output = stage(state, cond_t, routes_t, flow, has_area)
            if return_pyramid:
                pyramid.append(stage_output)

        output: dict[str, torch.Tensor | list[dict[str, torch.Tensor]]] = {
            "features": state.x,
            "points": state.pos,
            "normals": state.normals,
            "area": state.area,
            "log_area": state.log_area,
            "flow": flow,
        }
        if return_pyramid:
            output["pyramid"] = pyramid
        return output

    def load_route_compatible_state_dict(
        self,
        checkpoint: Mapping[str, Any],
        route_map: Optional[Sequence[int]] = None,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        return load_route_compatible_state_dict(self, checkpoint, route_map=route_map, strict=strict)


__all__ = [
    "UniFieldEncoder",
    "FlowManifoldBlock",
    "FlowEncoderStage",
    "SurfaceState",
]


if __name__ == "__main__":
    model = UniFieldEncoder(
        embed_dims=(32, 64, 128, 256),
        depths=(1, 1, 1, 1),
        num_heads=(2, 4, 4, 8),
        window_sizes=(128, 128, 64, 64),
    )
    points = torch.randn(2, 4096, 3)
    cond = torch.randn(2, 2)
    out = model(points, cond)
    print(out["features"].shape, out["points"].shape)
