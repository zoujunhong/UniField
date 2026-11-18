"""DSAPT adapter for train.py's shared surface/query model interface."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import torch
from torch import nn
from .network import DualEncoder, prepare_query_initial_features
from .decoder import QueryDecoder
from model.Baseline.official_common import split_queries, pack_outputs, surface_query_normals


class DSAPTFieldModel(nn.Module):
    """Batch-one dual-scale encoder and query decoder.

    The dataset supplies official grid-sampled encoder inputs in geometry_meta.point;
    surface_pos and all query xyz already share its centering and global scale.
    """
    def __init__(self, **config):
        super().__init__()
        self.config = deepcopy(config)
        self.encoder = DualEncoder(self.config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config['seed'])
            self.decoder = QueryDecoder(self.config)

    def forward(self, surface_pos, query_pos, cond=None, routes=None, normals=None,
                area=None, flow_vec=None, geometry_meta=None):
        if surface_pos.shape[0] != 1:
            raise ValueError('DSAPT currently requires one geometry per GPU.')
        meta = geometry_meta or {}
        if 'point' not in meta:
            raise ValueError('DSAPT requires official grid-sampled geometry_meta["point"].')
        point = dict(meta['point'])
        # Default DataLoader collation introduces one leading batch axis.
        for key in ('coord', 'grid_coord', 'feat'):
            if point[key].ndim == 3:
                point[key] = point[key][0]
        point['grid_coord'] = point['grid_coord'].int()
        point['offset'] = point['offset'].reshape(-1).long()
        surface, volume, mask = split_queries(query_pos)
        normal = surface_query_normals(surface, surface_pos, normals, meta)
        query = torch.cat((torch.cat((surface, normal), -1),
                           torch.cat((volume, torch.zeros_like(volume)), -1)), 1)
        features, positions, coarse_count, _ = self.encoder(point)
        initial = prepare_query_initial_features(features, positions, coarse_count, query, self.config)
        out = self.decoder(features, positions, None, query, initial, surface_count=surface.shape[1])
        vp = torch.cat((out['volume_p'], out['volume_u']), -1) if 'volume_p' in out else query.new_empty(1, 0, 4)
        return pack_outputs(out['surface_p'], vp, mask)

    @torch.no_grad()
    def forward_test(self, *args, query_chunk_size=None, **kwargs):
        return self.forward(*args, **kwargs)

    def load_historical_checkpoint(self, checkpoint, use_ema=False):
        """Load historical full states, optionally overlaying trainable EMA tensors."""
        state = {f'{part}.{name}': tensor for part in ('encoder', 'decoder')
                 for name, tensor in checkpoint[part].items()}
        if use_ema:
            state.update(checkpoint['ema']['shadow'])
            state.update(checkpoint['ema']['buffers'])
        return self.load_state_dict(state, strict=True)

    def load_route_compatible_state_dict(self, checkpoint, route_map=None,
                                         strict=False, partial_load=False):
        if route_map is not None or partial_load:
            raise ValueError('DSAPT requires an exact architecture; route/partial loading is unsupported.')
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            state = {f'{part}.{name}': tensor for part in ('encoder', 'decoder')
                     for name, tensor in checkpoint[part].items()}
        else:
            state = checkpoint.get('state_dict', checkpoint)
        state = {name.replace('_orig_mod.', '').removeprefix('module.'): tensor
                 for name, tensor in state.items()}
        # A partial load can silently discard the pretrained dual encoder.
        self.load_state_dict(state, strict=True)
        return dict(loaded=list(state), route_loaded=[], skipped=[])
