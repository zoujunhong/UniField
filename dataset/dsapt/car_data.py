"""Full compact-surface Utonia geometry and sampled AB-UPT-normalized targets.

GeometryDataset(config,names,validation=False); EpochSampler(n,seed,epoch=0,skip=0).
Sampler yields (epoch,index) for deterministic sampling and exact mid-epoch resume.
Validation forces epoch=0, so changing worker scheduling never changes its samples.
"""
from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

HERE = Path(__file__).resolve().parent
from model.third_party.utonia_dsapt.transform import default as official_transform
from model.third_party.utonia_dsapt.data import collate_fn as official_collate
from dataset.field_cache import load_surface_fields, load_volume

STATS = {
    'surface_pressure_mean': [-93.4098], 'surface_pressure_std': [120.787],

}


def load_array(path):
    with open(path, 'rb', buffering=0) as stream:
        return np.load(stream, allow_pickle=False)


class GeometryDataset(Dataset):
    def __init__(self, config, names, validation=False):
        self.config = config
        self.names = list(names)
        self.root = Path(config['data_root'])
        self.validation = bool(validation)
        self.validation_query_filter = True
        self.validity = json.loads(Path(config["validity_file"]).read_text())
        assert self.validity["complete"] and self.validity["total_cases_audited"] == 8119
        self.scale = np.float32(config['global_scale'])
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('A fixed, positive training-cache global_scale is required')
        self.aux = Path(config.get('validation_aux_root', HERE / 'validation_aux'))
        self.surface_queries = 10000 if self.validation else int(config.get('surface_queries', 16384))
        self.volume_queries = int(config.get('volume_queries', 16384))
        self.transform = official_transform(scale=1.0, apply_z_positive=False, normalize_coord=False)
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in STATS.items()}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        epoch, index = int(epoch), int(index)
        base = self.config.get('validation_seed', 99173) if self.validation else self.config.get('seed', 200099)
        if self.validation:
            epoch = 0
        seed = int(np.random.SeedSequence([base, epoch, index]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        np.random.seed(seed)  # Official GridSample uses the legacy NumPy RNG.
        name = self.names[index]
        directory = self.root / name
        surface, normals = load_surface_fields(directory)
        if surface.shape != (65536, 4):
            raise ValueError(f'{name}: expected compact [65536,4] surface array')
        if surface.dtype != np.float16:
            raise ValueError(f'{name}: expected FP16 cache')
        if normals.shape != (65536, 3) or normals.dtype != np.float32:
            raise ValueError(f'{name}: expected aligned FP32 normals')
        excluded = self.validity['excluded'].get(name, {}).get('field_cache_rows', [])
        if excluded:
            keep = np.ones(len(surface), dtype=bool)
            keep[excluded] = False
            surface, normals = surface[keep], normals[keep]
        threshold = (float(self.config.get('pressure_filter_train', -5000)) if not self.validation else
                     -10.0 * 0.5 * float(self.config.get('u_ref', 30.0)) ** 2 if self.validation_query_filter else -np.inf)
        candidates = np.flatnonzero(surface[:, 3].astype(np.float32) >= threshold)
        if len(candidates) < self.surface_queries:
            raise ValueError(f'{name}: only {len(candidates)} valid surface queries; need {self.surface_queries}')
        sq = rng.choice(candidates, self.surface_queries, replace=False)
        xyz = surface[:, :3].astype(np.float32)
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        shift = np.asarray([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], dtype=np.float32)
        surface_rows = surface[sq].astype(np.float32)
        xyz = (xyz - shift) / self.scale
        surface_query = (surface_rows[:, :3] - shift) / self.scale
        point = self.transform(dict(coord=xyz, color=np.zeros_like(xyz), normal=normals))
        point = {key: point[key] for key in ('coord', 'grid_coord', 'feat', 'offset')}
        def normalize(values, field):
            return (values - self.stats[field + '_mean']) / self.stats[field + '_std']
        result = dict(point=point, name=name,
            query=torch.from_numpy(np.concatenate((surface_query, normals[sq]), axis=-1)),
            target_surface_p=torch.from_numpy(normalize(surface_rows[:, 3:4], 'surface_pressure')))
        if not all(torch.isfinite(value).all() for value in result.values() if isinstance(value, torch.Tensor)):
            raise ValueError(f'{name}: nonfinite queries/targets')
        if not torch.isfinite(point['feat']).all():
            raise ValueError(f'{name}: nonfinite encoder features')
        return result


def collate(batch):
    result = {key: torch.stack([row[key] for row in batch])
              for key in batch[0] if key not in ('point', 'name')}
    result['point'] = official_collate([row['point'] for row in batch])
    result['name'] = [row['name'] for row in batch]
    return result


def to_device(batch, device):
    return {key: ({k: value.to(device, non_blocking=True) for k, value in item.items()}
                  if key == 'point' else item.to(device, non_blocking=True)
                  if isinstance(item, torch.Tensor) else item) for key, item in batch.items()}


class EpochSampler(Sampler):
    def __init__(self, n, seed, epoch=0, skip=0):
        self.n, self.seed, self.epoch, self.skip = int(n), int(seed), int(epoch), int(skip)
        if not 0 <= self.skip <= self.n:
            raise ValueError('Sampler skip must be within epoch')

    def __len__(self):
        return self.n - self.skip

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.n, generator=generator).tolist()[self.skip:]
        return iter((self.epoch, index) for index in indices)
