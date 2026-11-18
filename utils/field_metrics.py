"""Shared field-space conversion for evaluation and visualization."""
from __future__ import annotations

import torch


def normalization_spec(task_config: dict, field: str, domain: str | None = None) -> dict | None:
    settings = task_config.get('denormalize', {})
    if domain is not None and isinstance(settings.get(domain), dict):
        return settings[domain].get(field)
    return settings.get(field)


def denormalize(values: torch.Tensor, field: str, task_config: dict,
                batch: dict | None = None, domain: str | None = None) -> torch.Tensor:
    spec = normalization_spec(task_config, field, domain)
    if not spec or spec.get('type') == 'identity':
        return values
    if spec.get('type') == 'affine':
        shape = (*([1] * (values.ndim - 1)), -1)
        return values * values.new_tensor(spec['std']).view(shape) + values.new_tensor(spec['mean']).view(shape)
    if spec.get('type') == 'physical':
        dataset = spec.get('dataset', 'DrivAerNet')
        if dataset == 'DrivAerNet':
            u_ref = float(spec.get('u_ref', 30.0))
            if field == 'p':
                return values * (0.5 * u_ref * u_ref)
            if field == 'U':
                reference = values.new_tensor(spec.get('velocity_ref', [u_ref, 0., 0.])).view(1, 1, 3)
                return values * abs(u_ref) + reference
        if dataset in {'HighSpeedTrain', 'CRH450', 'Maglev'}:
            if batch is None:
                raise ValueError('HighSpeedTrain physical denormalization requires batch conditions.')
            cond = batch['cond']
            speed = torch.sqrt((cond[:, 0] * 100.).square() + (cond[:, 1] * 20.).square()).clamp_min(1e-12).view(-1, 1, 1)
            factors = dict(p=0.5 * speed.square(), U=speed, k=0.01 * speed.square(), omega=100. * speed, nut=0.01 * speed)
            if field in factors:
                return values * factors[field]
    raise ValueError(f'Unsupported denormalize spec for {domain}/{field}: {spec}')
