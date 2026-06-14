from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath

from .ffn import FFN, RoutedAdaLN, apply_norm


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(batch_first)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(q.shape[0], q.shape[2], self.embed_dim)
        x = self.proj_drop(self.out_proj(x))
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        context_dim = dim if context_dim is None else context_dim
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attn_drop = float(attn_drop)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(context))
        v = self._shape(self.v_proj(context))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], self.dim)
        return self.proj_drop(self.out_proj(x))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        num_routes: int = 1,
        context_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_adaln: bool = True,
    ):
        super().__init__()
        self.norm_q = RoutedAdaLN(dim, cond_dim, num_routes) if use_adaln else nn.LayerNorm(dim)
        context_dim = dim if context_dim is None else context_dim
        self.norm_context = nn.LayerNorm(context_dim)
        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            context_dim=context_dim,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_ffn = RoutedAdaLN(dim, cond_dim, num_routes) if use_adaln else nn.LayerNorm(dim)
        self.ffn = FFN(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.attn(apply_norm(self.norm_q, x, cond, routes), self.norm_context(context))
        )
        x = x + self.drop_path2(self.ffn(apply_norm(self.norm_ffn, x, cond, routes)))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        num_routes: int = 1,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_adaln: bool = True,
    ):
        super().__init__()
        self.norm_attn = RoutedAdaLN(dim, cond_dim, num_routes) if use_adaln else nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_ffn = RoutedAdaLN(dim, cond_dim, num_routes) if use_adaln else nn.LayerNorm(dim)
        self.ffn = FFN(dim=dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        routes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_norm = apply_norm(self.norm_attn, x, cond, routes)
        x = x + self.drop_path1(self.attn(x_norm, x_norm, x_norm))
        x = x + self.drop_path2(self.ffn(apply_norm(self.norm_ffn, x, cond, routes)))
        return x


class FlowManifoldAttention(nn.Module):
    pair_feature_dim = 9

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flow_attention: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attn_drop = float(attn_drop)
        self.use_flow_attention = bool(use_flow_attention)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.use_flow_attention:
            self.pair_bias_weight = nn.Parameter(torch.zeros(num_heads, self.pair_feature_dim))
            self.reset_pair_bias()
        else:
            self.register_parameter("pair_bias_weight", None)

    def reset_pair_bias(self) -> None:
        if self.pair_bias_weight is None:
            return
        with torch.no_grad():
            self.pair_bias_weight.zero_()
            self.pair_bias_weight[:, 1].fill_(-0.02)
            if self.num_heads == 1:
                self.pair_bias_weight[:, 0].fill_(0.02)
            else:
                half = self.num_heads // 2
                self.pair_bias_weight[:half, 0].fill_(0.02)
                self.pair_bias_weight[half:, 0].fill_(-0.02)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _pair_bias(
        self,
        q_pos: torch.Tensor,
        kv_pos: torch.Tensor,
        q_normals: torch.Tensor,
        kv_normals: torch.Tensor,
        q_log_area: torch.Tensor,
        kv_log_area: torch.Tensor,
        flow: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        q_pos = q_pos.to(dtype=dtype)
        kv_pos = kv_pos.to(dtype=dtype)
        q_normals = q_normals.to(dtype=dtype)
        kv_normals = kv_normals.to(dtype=dtype)
        q_log_area = q_log_area.to(dtype=dtype)
        kv_log_area = kv_log_area.to(dtype=dtype)
        flow = F.normalize(flow.to(dtype=dtype), dim=-1)
        rel = q_pos[:, :, None, :] - kv_pos[:, None, :, :]
        signed_stream = (rel * flow[:, None, None, :]).sum(dim=-1)
        rel_sq = (rel * rel).sum(dim=-1)
        transverse = (rel_sq - signed_stream * signed_stream).clamp_min(0.0).sqrt()
        normal_align = (q_normals[:, :, None, :] * kv_normals[:, None, :, :]).sum(dim=-1)
        q_incidence = (q_normals * flow[:, None, :]).sum(dim=-1)[:, :, None].expand_as(signed_stream)
        kv_incidence = (kv_normals * flow[:, None, :]).sum(dim=-1)[:, None, :].expand_as(signed_stream)
        q_normal_disp = (rel * q_normals[:, :, None, :]).sum(dim=-1)
        kv_normal_disp = (rel * kv_normals[:, None, :, :]).sum(dim=-1)
        q_area = q_log_area[:, :, 0][:, :, None].expand_as(signed_stream)
        kv_area = kv_log_area[:, :, 0][:, None, :].expand_as(signed_stream)
        features = torch.stack(
            [
                signed_stream,
                transverse,
                normal_align,
                q_incidence,
                kv_incidence,
                q_normal_disp,
                kv_normal_disp,
                q_area,
                kv_area,
            ],
            dim=-1,
        )
        return torch.einsum("bqkf,hf->bhqk", features, self.pair_bias_weight.to(dtype=dtype))

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
        flow: torch.Tensor,
    ) -> torch.Tensor:
        q = self._shape(self.q_proj(q_x))
        k = self._shape(self.k_proj(kv_x))
        v = self._shape(self.v_proj(kv_x))
        attn_bias = None
        if self.use_flow_attention:
            attn_bias = self._pair_bias(
                q_pos,
                kv_pos,
                q_normals,
                kv_normals,
                q_log_area,
                kv_log_area,
                flow,
                q.dtype,
            )
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(q_x.shape[0], q_x.shape[1], self.dim)
        return self.proj_drop(self.out_proj(x))


class QuerySurfaceAttention(nn.Module):
    pair_feature_dim = 3

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flow_attention: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attn_drop = float(attn_drop)
        self.use_flow_attention = bool(use_flow_attention)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.use_flow_attention:
            self.pair_bias_weight = nn.Parameter(torch.zeros(num_heads, self.pair_feature_dim))
            self.reset_pair_bias()
        else:
            self.register_parameter("pair_bias_weight", None)

    def reset_pair_bias(self) -> None:
        if self.pair_bias_weight is None:
            return
        with torch.no_grad():
            self.pair_bias_weight.zero_()
            self.pair_bias_weight[:, 1].fill_(-0.02)
            if self.num_heads == 1:
                self.pair_bias_weight[:, 0].fill_(0.02)
            else:
                half = self.num_heads // 2
                self.pair_bias_weight[:half, 0].fill_(0.02)
                self.pair_bias_weight[half:, 0].fill_(-0.02)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _pair_bias(
        self,
        query_pos: torch.Tensor,
        surface_pos: torch.Tensor,
        surface_log_area: torch.Tensor,
        flow: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_pos = query_pos.to(dtype=dtype)
        surface_pos = surface_pos.to(dtype=dtype)
        surface_log_area = surface_log_area.to(dtype=dtype)
        flow = F.normalize(flow.to(dtype=dtype), dim=-1)
        rel = query_pos[:, :, None, :] - surface_pos[:, None, :, :]
        signed_stream = (rel * flow[:, None, None, :]).sum(dim=-1)
        rel_sq = (rel * rel).sum(dim=-1)
        transverse = (rel_sq - signed_stream * signed_stream).clamp_min(0.0).sqrt()
        area = surface_log_area[:, :, 0][:, None, :].expand_as(signed_stream)
        features = torch.stack([signed_stream, transverse, area], dim=-1)
        return torch.einsum("bqkf,hf->bhqk", features, self.pair_bias_weight.to(dtype=dtype))

    def forward(
        self,
        query_x: torch.Tensor,
        surface_x: torch.Tensor,
        query_pos: torch.Tensor,
        surface_pos: torch.Tensor,
        surface_log_area: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        q = self._shape(self.q_proj(query_x))
        k = self._shape(self.k_proj(surface_x))
        v = self._shape(self.v_proj(surface_x))
        attn_bias = None
        if self.use_flow_attention:
            attn_bias = self._pair_bias(query_pos, surface_pos, surface_log_area, flow, q.dtype)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.permute(0, 2, 1, 3).contiguous().view(query_x.shape[0], query_x.shape[1], self.dim)
        return self.proj_drop(self.out_proj(x))


__all__ = [
    "MultiheadAttention",
    "CrossAttention",
    "CrossAttentionBlock",
    "SelfAttentionBlock",
    "FlowManifoldAttention",
    "QuerySurfaceAttention",
]
