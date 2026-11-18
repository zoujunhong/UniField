"""DrivAerNet cache API. Build physical named fields in per-case directories."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from dataset.cache.drivaernet_fields import (
    RAW_ROOT, CACHE_ROOT, DrivAerNetPlusPlusCacheBuilder,
    DrivAerNetPlusPlusWallDistanceCacheBuilder, case_directory, atomic_json,
)
DRIVAERNET_ROOT = RAW_ROOT
DEFAULT_CSV_FILE = os.path.join(DRIVAERNET_ROOT, 'DrivAerNetPlusPlus_Cd_8k_Updated.csv')
DEFAULT_CACHE_ROOT = CACHE_ROOT
DEFAULT_SURFACE_CACHE_POINTS = 100_000
DEFAULT_VOLUME_CACHE_POINTS = 300_000


def cache_dir(cache_root, kind, points):
    """Legacy flat path, for explicit historical readers only; new builders never write it."""
    return os.path.join(str(cache_root), f'{kind}_{int(points)}_fp16')


def cache_path(cache_root, kind, points, case_name):
    return os.path.join(cache_dir(cache_root, kind, points), f'{case_name}.npy')


def wall_distance_cache_path(cache_root, points, case_name):
    return cache_path(cache_root, 'volume_wall_distance', points, case_name)


def _cached_cases(root, ids_file, profile):
    from dataset.utils.sampling import read_ids
    requested = read_ids(ids_file)
    names = sorted(p.name for p in Path(root).iterdir() if p.is_dir()
        and (case_directory(root, p.name, profile) / 'surface.npy').exists())
    if requested is not None:
        if requested - set(names):
            raise FileNotFoundError(f'{len(requested - set(names))} requested cases missing selected cache')
        names = [name for name in names if name in requested]
    if not names:
        raise ValueError('No selected cached cases')
    return names


def running_cache_stats(paths, columns, log_every=100):
    from dataset.field_cache import read_fields
    count, total, total_sq, low, high = 0, None, None, None, None
    for index, path in enumerate(paths):
        array = read_fields(path) if isinstance(columns, str) else np.load(path, mmap_mode='r')
        values = np.asarray(array[columns] if isinstance(columns, str) else array[:, columns], dtype=np.float64)
        values = values.reshape(len(array), -1)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite statistics input: {path}')
        if total is None:
            total = np.zeros(values.shape[1]); total_sq = total.copy()
            low = np.full(values.shape[1], np.inf); high = -low
        count += len(values); total += values.sum(0); total_sq += (values * values).sum(0)
        low, high = np.minimum(low, values.min(0)), np.maximum(high, values.max(0))
        if log_every and (index + 1) % log_every == 0:
            print(f'[{index + 1}/{len(paths)}] scanned {path}', flush=True)
    if not count:
        raise ValueError('Cannot compute statistics over an empty pool')
    mean = total / count
    return dict(count=count, mean=mean, std=np.sqrt(np.maximum(total_sq / count - mean * mean, 0)), min=low, max=high)


def compute_cache_statistics(root_dir=DRIVAERNET_ROOT, cache_root=None, surface_sampling='random',
        surface_cache_points=DEFAULT_SURFACE_CACHE_POINTS, volume_cache_points=DEFAULT_VOLUME_CACHE_POINTS,
        ids_file=None, log_every=100, cache_profile=None):
    root = cache_root or DEFAULT_CACHE_ROOT
    directories = [case_directory(root, name, cache_profile) for name in _cached_cases(root, ids_file, cache_profile)]
    return dict(pressure=running_cache_stats([d / 'surface.npy' for d in directories], 'pressure', log_every),
                velocity=running_cache_stats([d / 'volume.npy' for d in directories], 'velocity', log_every))


def coordinate_statistics_path(cache_root, split_name='train_center_x'):
    return os.path.join(str(cache_root), f'coordinate_statistics_{split_name}.json')


def compute_coordinate_statistics(root_dir=DRIVAERNET_ROOT, cache_root=None, surface_sampling='random',
        surface_cache_points=DEFAULT_SURFACE_CACHE_POINTS, volume_cache_points=DEFAULT_VOLUME_CACHE_POINTS,
        ids_file=None, center_x=True, include_volume=True, log_every=100, cache_profile=None):
    from dataset.field_cache import read_fields
    root = cache_root or DEFAULT_CACHE_ROOT
    names = _cached_cases(root, ids_file, cache_profile)
    count = 0; total = np.zeros(3); total_sq = np.zeros(3)
    minimum = np.full(3, np.inf); maximum = -minimum
    for name in names:
        directory = case_directory(root, name, cache_profile)
        surface = read_fields(directory / 'surface.npy')['xyz']
        shift = .5 * (float(surface[:, 0].min()) + float(surface[:, 0].max())) if center_x else 0.
        arrays = [surface]
        if include_volume: arrays.append(read_fields(directory / 'volume.npy')['xyz'])
        for array in arrays:
            values = np.asarray(array, dtype=np.float64).copy(); values[:, 0] -= shift
            if not np.isfinite(values).all(): raise ValueError(f'Nonfinite coordinates: {name}')
            count += len(values); total += values.sum(0); total_sq += (values * values).sum(0)
            minimum = np.minimum(minimum, values.min(0)); maximum = np.maximum(maximum, values.max(0))
    mean = total / count
    return dict(version=2, count=count, mean=mean.tolist(),
        std=np.sqrt(np.maximum(total_sq / count - mean * mean, 0)).tolist(),
        min=minimum.tolist(), max=maximum.tolist(), cache_root=str(root), cache_profile=cache_profile,
        ids_file=str(ids_file or ''), center_x=center_x, include_volume=include_volume)


def load_or_compute_coordinate_statistics(output_path=None, split_name='train_center_x', force=False, **kwargs):
    root = kwargs.get('cache_root') or DEFAULT_CACHE_ROOT
    path = Path(output_path or coordinate_statistics_path(root, split_name))
    if path.exists() and not force:
        payload = json.loads(path.read_text())
        if len(payload.get('mean', [])) == 3 and len(payload.get('std', [])) == 3:
            return payload
    result = compute_coordinate_statistics(**kwargs)
    atomic_json(path, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', default=DRIVAERNET_ROOT)
    parser.add_argument('--cache-root', default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--cache-profile', default='sampled_surface100000_volume300000_fp32')
    parser.add_argument('--surface-cache-points', type=int, default=DEFAULT_SURFACE_CACHE_POINTS, help='0: all source rows')
    parser.add_argument('--volume-cache-points', type=int, default=DEFAULT_VOLUME_CACHE_POINTS, help='0: all source rows')
    parser.add_argument('--surface-sampling', choices=['random', 'uniform'], default='random')
    parser.add_argument('--ids-file')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--include-wall-distance', action='store_true')
    parser.add_argument('--include-topology', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    options = vars(parser.parse_args()); workers = options.pop('num_workers')
    options['surface_cache_points'] = options['surface_cache_points'] or None
    options['volume_cache_points'] = options['volume_cache_points'] or None
    DrivAerNetPlusPlusCacheBuilder(**options).build_cache(num_workers=workers)
