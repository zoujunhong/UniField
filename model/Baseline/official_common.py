"""Shared tensor packing for the retained official baseline implementations."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def split_queries(query_pos):
    if query_pos.ndim != 3 or query_pos.shape[-1] < 4:
        raise ValueError('Native field models require xyz plus a surface=0/volume=1 type channel.')
    mask = query_pos[..., -1] > 0.5
    if not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError('Each batch member must have the same surface/volume query layout.')
    return query_pos[:, ~mask[0], :3], query_pos[:, mask[0], :3], mask[0]


def pack_outputs(surface, volume, mask):
    width = max(surface.shape[-1], volume.shape[-1])
    output = surface.new_zeros(surface.shape[0], mask.numel(), width)
    output[:, ~mask] = F.pad(surface, (0, width - surface.shape[-1]))
    output[:, mask] = F.pad(volume, (0, width - volume.shape[-1]))
    return output


def surface_query_normals(surface_query, surface_pos, normals, geometry_meta):
    meta = geometry_meta or {}
    value = meta.get('query_normals', meta.get('surface_query_normals'))
    if value is not None:
        if value.shape != surface_query.shape:
            raise ValueError('query_normals must align exactly with surface query points.')
        return value.to(surface_query)
    if normals is not None and surface_query.shape == surface_pos.shape and torch.equal(surface_query, surface_pos):
        return normals.to(surface_query)
    if surface_query.shape[1] == 0:
        return surface_query
    raise ValueError('Surface query normals must be supplied by the dataset in geometry_meta["query_normals"].')


def load_official_checkpoint(model, checkpoint, *, branch=None, route_map=None,
                             strict=True, partial_load=False):
    """Load retained native baselines, including their original training containers.

    Tensor names are only stripped of known wrappers. Architecture mismatches
    are always errors so a historical checkpoint never becomes a partial model.
    """
    if route_map or partial_load:
        raise ValueError('Official baselines do not support route remapping or partial weight loading.')
    from utils.checkpoint import clean_state_dict
    if not isinstance(checkpoint, dict):
        raise TypeError('Expected a state dict or a historical checkpoint dictionary.')
    if 'models' in checkpoint:
        state = {f'{domain}.{key}': value for domain, tensors in checkpoint['models'].items()
                 for key, value in clean_state_dict(tensors).items()}
    else:
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = clean_state_dict(state)
        if branch is not None and not any(k.startswith(('surface.', 'volume.')) for k in state):
            if branch == 'both':
                raise ValueError('An unlabelled single-branch checkpoint requires branch=surface or volume.')
            state = {f'{branch}.{key.removeprefix("net.")}': value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return {'loaded': list(state), 'legacy_loaded': [], 'route_loaded': [],
            'partial_loaded': [], 'skipped': []}
