from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiheadAttention


def build_downsampled_coords(
    pos_in: torch.Tensor,
    feat_in: torch.Tensor,
    down_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return pos_in[:, ::down_ratio], feat_in[:, ::down_ratio]


class PointCrossAttention(nn.Module):
    """Patch-wise iterative cross-attention used by legacy query models."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        down_ratio: int,
        patch_size: int,
        iters: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.iters = int(iters)
        self.dim = int(dim)
        self.down_ratio = int(down_ratio)
        self.patch_size = int(patch_size)
        self.attn = MultiheadAttention(dim, num_heads)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor, slots_init: torch.Tensor) -> torch.Tensor:
        batch, points, dim = inputs.shape
        slot_count = slots_init.shape[1]
        if not (points // slot_count == self.down_ratio or slot_count // points == self.down_ratio):
            raise ValueError("inputs and slots_init must differ by down_ratio.")
        if points % self.patch_size != 0:
            raise ValueError("Point count must be divisible by patch_size.")

        groups = points // self.patch_size
        inputs = inputs.view(batch, groups, self.patch_size, dim).flatten(0, 1)
        if points // slot_count == self.down_ratio:
            slots_init = slots_init.view(
                batch,
                groups,
                self.patch_size // self.down_ratio,
                dim,
            ).flatten(0, 1)
        else:
            slots_init = slots_init.view(
                batch,
                groups,
                self.patch_size * self.down_ratio,
                dim,
            ).flatten(0, 1)

        inputs = self.norm_inputs(inputs)
        slots = slots_init
        for _ in range(self.iters):
            slots_prev = slots
            updates = self.attn(self.norm_slots(slots), inputs, inputs)
            slots = self.gru(
                updates.reshape(-1, dim),
                slots_prev.reshape(-1, dim),
            ).reshape(batch * groups, -1, dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        if points // slot_count == self.down_ratio:
            return slots.view(batch, groups, self.patch_size // self.down_ratio, dim).flatten(1, 2)
        return slots.view(batch, groups, self.patch_size * self.down_ratio, dim).flatten(1, 2)


__all__ = ["PointCrossAttention", "build_downsampled_coords"]
