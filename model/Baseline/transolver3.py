"""Official Transolver-3, with independent surface and volume branches."""
from __future__ import annotations
import torch
from torch import nn
from model.third_party.transolver3 import Model
from .official_common import load_official_checkpoint, split_queries, pack_outputs, surface_query_normals


class Transolver3FieldModel(nn.Module):
    def __init__(self, hidden_dim=256, layers=12, heads=8, slices=64,
                 branch='both', tile_points=65536, mlp_ratio=2):
        super().__init__()
        if branch not in {'surface', 'volume', 'both'}:
            raise ValueError('branch must be surface, volume, or both.')
        self.branch, self.tile_points = branch, int(tile_points)
        common = dict(n_hidden=hidden_dim, n_layers=layers, fun_dim=0, n_head=heads,
                      mlp_ratio=mlp_ratio, slice_num=slices, unified_pos=False)
        self.surface = Model(space_dim=6, out_dim=1, **common) if branch != 'volume' else None
        self.volume = Model(space_dim=3, out_dim=4, **common) if branch != 'surface' else None

    def forward(self, surface_pos, query_pos, cond=None, routes=None, normals=None,
                area=None, flow_vec=None, geometry_meta=None):
        surface, volume, mask = split_queries(query_pos)
        sp = query_pos.new_empty(query_pos.shape[0], 0, 1)
        vp = query_pos.new_empty(query_pos.shape[0], 0, 4)
        if surface.shape[1]:
            if self.surface is None:
                raise ValueError('Surface queries require a surface branch.')
            qn = surface_query_normals(surface, surface_pos, normals, geometry_meta)
            x = torch.cat((surface, qn), dim=-1)
            sp = torch.cat(self.surface(list(x.split(self.tile_points, dim=1)),
                                       input_list=True, use_checkpoint=self.training), dim=1)
        if volume.shape[1]:
            if self.volume is None:
                raise ValueError('Volume queries require a volume branch.')
            vp = torch.cat(self.volume(list(volume.split(self.tile_points, dim=1)),
                                      input_list=True, use_checkpoint=self.training), dim=1)
        return pack_outputs(sp, vp, mask)

    @torch.no_grad()
    def forward_test(self, *args, query_chunk_size=None, **kwargs):
        # Official chunk statistics are shared across all input chunks.
        return self.forward(*args, **kwargs)

    def load_branch_state_dict(self, state, branch, strict=True):
        target = getattr(self, branch, None)
        if target is None:
            raise ValueError(f'No {branch} branch in this model.')
        return target.load_state_dict(state, strict=strict)

    def load_route_compatible_state_dict(self, checkpoint, **kwargs):
        return load_official_checkpoint(self, checkpoint, branch=self.branch, **kwargs)
