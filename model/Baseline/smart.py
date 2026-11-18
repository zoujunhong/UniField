"""Official SMART with the shared whole_field tensor interface."""
from __future__ import annotations
import torch
from model.third_party.smart import SMART
from .official_common import load_official_checkpoint, split_queries, pack_outputs


class SMARTFieldModel(SMART):
    def __init__(self, positions_normalized=False,
                 position_min=(-12.0012, -4.0004, -3.46979e-18),
                 position_max=(32.0032, 4.0004, 6.40064), **kwargs):
        defaults = dict(spatial_dim=3, surface_channels=1, volume_channels=4, parameter_channels=0,
                        latent_dim=256, latent_geometry_points=2048, subsampled_geometry_points=8192,
                        num_encoder_decoder_blocks=6, num_heads=8, pos_scale_factor=1000,
                        dropout=0.0, subregion_size=32768)
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.condition_dim = defaults['parameter_channels']
        self.positions_normalized = bool(positions_normalized)
        self.position_min, self.position_max = tuple(position_min), tuple(position_max)

    def _coordinates(self, positions):
        if self.positions_normalized:
            return positions
        lower, upper = positions.new_tensor(self.position_min), positions.new_tensor(self.position_max)
        return (positions - lower) / (upper - lower)

    def forward(self, surface_pos, query_pos, cond=None, routes=None, normals=None,
                area=None, flow_vec=None, geometry_meta=None):
        surface, volume, mask = split_queries(query_pos)
        sp, vp = super().forward(self._coordinates(surface_pos), self._coordinates(surface),
                                 self._coordinates(volume), cond if self.condition_dim else None)
        return pack_outputs(sp, vp, mask)

    @torch.no_grad()
    def forward_test(self, *args, query_chunk_size=None, **kwargs):
        return self.forward(*args, **kwargs)

    def load_route_compatible_state_dict(self, checkpoint, **kwargs):
        return load_official_checkpoint(self, checkpoint, **kwargs)
