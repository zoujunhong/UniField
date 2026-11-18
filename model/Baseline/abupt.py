"""Official AB-UPT with the shared whole_field tensor interface.

The inherited state_dict is identical to the retained DrivAerNet++ experiment.
Input coordinates are physical coordinates unless positions_normalized=True.
"""
from __future__ import annotations
import torch
from model.third_party.abupt import AnchoredBranchedUPT
from .official_common import load_official_checkpoint, split_queries, pack_outputs


class ABUPTFieldModel(AnchoredBranchedUPT):
    def __init__(self, geometry_supernodes=4096, positions_normalized=False,
                 position_min=-12.0012, position_max=32.0032, **kwargs):
        defaults = dict(ndim=3, input_dim=3, output_dim_surface=1, output_dim_volume=4,
                        dim=192, geometry_depth=1, num_heads=3, blocks='pscscscs',
                        num_volume_blocks=4, num_surface_blocks=4, radius=1.0)
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.geometry_supernodes = int(geometry_supernodes)
        self.positions_normalized = bool(positions_normalized)
        self.position_min, self.position_max = float(position_min), float(position_max)

    def _coordinates(self, positions):
        if self.positions_normalized:
            return positions
        return (positions - self.position_min) / (self.position_max - self.position_min) * 1000.0

    def forward(self, surface_pos, query_pos, cond=None, routes=None, normals=None,
                area=None, flow_vec=None, geometry_meta=None):
        if surface_pos.shape[0] != 1:
            raise ValueError('The retained official AB-UPT implementation requires one geometry per GPU.')
        surface, volume, mask = split_queries(query_pos)
        meta = geometry_meta or {}
        volume_context = volume if volume.shape[1] else meta.get('volume_context')
        if not surface.shape[1] or volume_context is None or not volume_context.shape[1]:
            raise ValueError('AB-UPT requires surface anchors and volume anchors/context.')
        indices = meta.get('geometry_supernode_idx')
        if indices is None:
            indices = torch.randperm(surface_pos.shape[1], device=surface_pos.device)[:self.geometry_supernodes]
        indices = indices.reshape(-1).long()
        outputs = super().forward(
            geometry_position=self._coordinates(surface_pos[0]),
            geometry_supernode_idx=indices, geometry_batch_idx=None,
            surface_anchor_position=self._coordinates(surface),
            volume_anchor_position=self._coordinates(volume_context))
        sp = outputs['surface_anchor_pressure'].reshape(1, surface.shape[1], 1)
        vp = torch.cat((outputs['volume_anchor_totalpcoeff'], outputs['volume_anchor_velocity']), -1)
        vp = vp.reshape(1, volume_context.shape[1], 4)[:, :volume.shape[1]]
        return pack_outputs(sp, vp, mask)

    @torch.no_grad()
    def forward_test(self, *args, query_chunk_size=None, **kwargs):
        # Anchor attention couples queries. Arbitrary chunks change the operator.
        return self.forward(*args, **kwargs)

    def load_route_compatible_state_dict(self, checkpoint, **kwargs):
        return load_official_checkpoint(self, checkpoint, **kwargs)
