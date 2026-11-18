"""Model-independent FP32 cache construction; execute only on an idle compute node."""
from __future__ import annotations
import hashlib
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader, Subset
from dataset.utils.load import list_vtk_files, read_drivaer_surface_pressure, surface_polydata
from dataset.utils.sampling import filter_names, read_ids

RAW_ROOT = '/lustre/group/project1/CFD/DrivAerNet++'
CACHE_ROOT = '/data/home/zdhs0017/zoujunhong/data/DrivAerNet/cases'


def coefficient_key(identifier):
    prefix, separator, number = identifier.rpartition('_')
    return (prefix, int(number)) if separator and number.isdigit() else identifier


def case_directory(root, case, profile=None):
    path = Path(root) / case
    if profile:
        if Path(profile).name != profile or profile in {'.', '..'}:
            raise ValueError('cache_profile must be one directory name')
        path = path / 'pools' / profile
    return path


def atomic_npy(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2) + '\n'); temporary.replace(path)


def record(path, values, fields):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return dict(file=Path(path).name, shape=list(values.shape), dtype=values.dtype.descr,
                fields=fields, sha256=digest.hexdigest())


def indices(xyz, count, rng, mode='random'):
    if count is None or count >= len(xyz):
        return np.arange(len(xyz), dtype=np.int64)
    if count <= 0:
        raise ValueError('Cache point count must be positive or None for all rows')
    if mode == 'random':
        return rng.choice(len(xyz), count, replace=False).astype(np.int64)
    # Select voxel representatives, then fill/truncate without duplicating rows.
    span = np.maximum(np.ptp(xyz, axis=0), 1e-12)
    cells = np.floor((xyz - xyz.min(axis=0)) / span * max(1, int(np.ceil(count ** (1 / 3))))).astype(np.int64)
    order = rng.permutation(len(xyz)); _, first = np.unique(cells[order], axis=0, return_index=True)
    chosen = order[first]
    if len(chosen) >= count:
        return rng.choice(chosen, count, replace=False).astype(np.int64)
    remaining = np.setdiff1d(np.arange(len(xyz)), chosen)
    return np.concatenate((chosen, rng.choice(remaining, count - len(chosen), replace=False))).astype(np.int64)


def identity_collate(batch):
    return batch


class DrivAerNetPlusPlusCacheBuilder(Dataset):
    """Write a distinct physical FP32 pool; no training normalization or thresholding.

    Surface area is the original point-lumped/face area, not a quadrature weight
    for a subsampled point set. Optional topology uses its own vertex indexing.
    """
    def __init__(self, root_dir=RAW_ROOT, cache_root=None, surface_cache_points=100000,
                 volume_cache_points=300000, surface_sampling='random', volume_location='point',
                 ids_file=None, overwrite=False, build_surface=True, build_volume=True,
                 cache_profile='sampled_surface100000_volume300000_fp32',
                 include_wall_distance=False, include_topology=False, seed=0):
        if surface_sampling not in {'random', 'uniform'}:
            raise ValueError('Choose random or uniform; store distinct sampling in distinct profiles')
        if volume_location not in {'point', 'cell'} or not (build_surface or build_volume):
            raise ValueError('Invalid domain or volume location')
        self.root_dir, self.cache_root = Path(root_dir), Path(cache_root or CACHE_ROOT)
        self.surface_cache_points, self.volume_cache_points = surface_cache_points, volume_cache_points
        self.surface_sampling, self.volume_location = surface_sampling, volume_location
        self.overwrite, self.build_surface, self.build_volume = overwrite, build_surface, build_volume
        self.cache_profile, self.seed = cache_profile, int(seed)
        self.include_wall_distance, self.include_topology = include_wall_distance, include_topology
        self.coefficients = {}
        csv_path = self.root_dir / 'DrivAerNetPlusPlus_Cd_8k_Updated.csv'
        if csv_path.exists():
            with csv_path.open() as stream:
                for row in csv.DictReader(stream):
                    self.coefficients[coefficient_key(row.get('ID', ''))] = row
        self.pressure_files = list_vtk_files(self.root_dir / 'Pressure') if build_surface or include_wall_distance or include_topology else {}
        self.cfd_files = list_vtk_files(self.root_dir / 'CFD') if build_volume else {}
        names = set(self.pressure_files) if build_surface else set(self.cfd_files)
        if build_surface and build_volume:
            names &= set(self.cfd_files)
        self.case_names = filter_names(sorted(names), ids_file)
        requested = read_ids(ids_file)
        if requested is not None and set(self.case_names) != requested:
            raise FileNotFoundError('Requested cases missing raw input')
        if not self.case_names:
            raise ValueError('No raw cases available')
        case_directory(self.cache_root, self.case_names[0], cache_profile)

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, index):
        import pyvista as pv
        name = self.case_names[index]
        case_seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
        surface_rng = np.random.default_rng(np.random.SeedSequence([self.seed, case_seed, 0]))
        volume_rng = np.random.default_rng(np.random.SeedSequence([self.seed, case_seed, 1]))
        directory = case_directory(self.cache_root, name, self.cache_profile)
        meta_path = directory / 'metadata.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        settings = dict(surface_cache_points=self.surface_cache_points, volume_cache_points=self.volume_cache_points,
                        surface_sampling=self.surface_sampling, volume_location=self.volume_location, seed=self.seed,
                        include_wall_distance=self.include_wall_distance, include_topology=self.include_topology)
        if meta.get('builder_settings') not in (None, settings) and not self.overwrite:
            raise ValueError(f'{directory}: profile already has different settings; select a new profile')
        for domain, enabled in [('surface', self.build_surface), ('volume', self.build_volume)]:
            if enabled and (directory / (domain + '.npy')).exists() and domain not in meta and not self.overwrite:
                raise ValueError(f'{directory}: existing {domain} has no metadata; explicitly rebuild with overwrite')
        meta.update(schema='field_case_v1', case=name, dataset='DrivAerNet++', coordinate_frame='raw',
                    coordinate_unit='m', pressure_reference=dict(kind='kinematic', u_ref=30., p_ref=0.,
                    cp_definition='(p-p_ref)/(0.5*u_ref^2)'), builder_settings=settings,
                    condition=dict(velocity_m_s=[30., 0., 0.]), coefficients=self.coefficients.get(coefficient_key(name), {}))
        meta.setdefault('sampling', {}); meta.setdefault('source', {})
        path = directory / 'surface.npy'
        if self.build_surface and (self.overwrite or not path.exists()):
            rows = read_drivaer_surface_pressure(self.pressure_files[name])
            raw_surface = pv.read(self.pressure_files[name])
            geometry = surface_polydata(raw_surface)
            association = 'point' if 'p' in raw_surface.point_data else 'cell'
            # Preserve genuine zero areas; the legacy parser fills isolated
            # vertices for neural inputs, which is not a physical area cache.
            if association == 'point':
                triangles = geometry.triangulate()
                if triangles.n_points != len(rows) or not triangles.faces.size:
                    raise ValueError(f'{name}: cannot align source surface point areas')
                faces = np.asarray(triangles.faces).reshape(-1, 4)[:, 1:]
                areas = triangles.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
                point_area = np.zeros(len(rows), dtype=np.float64)
                np.add.at(point_area, faces.reshape(-1), np.repeat(np.asarray(areas) / 3., 3))
                rows[:, 7] = point_area
            else:
                rows[:, 7] = geometry.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
            selected = indices(rows[:, :3], self.surface_cache_points, surface_rng, self.surface_sampling)
            values = np.empty(len(selected), dtype=[('xyz', '<f4', (3,)), ('pressure', '<f4'),
                              ('normal', '<f4', (3,)), ('area', '<f4')])
            sample = rows[selected]
            values['xyz'], values['pressure'] = sample[:, :3], sample[:, 3]
            values['normal'], values['area'] = sample[:, 4:7], sample[:, 7]
            atomic_npy(path, values); atomic_npy(directory / 'surface_source_indices.npy', selected)
            meta['surface'] = record(path, values, dict(xyz=dict(unit='m', components=3),
                pressure=dict(unit='m^2/s^2', kind='kinematic'), normal=dict(unit='1', components=3),
                area=dict(unit='m^2', definition='source point lumped triangle area or source face area; sampled rows do not integrate the whole mesh')))
            meta['sampling']['surface'] = dict(mode=self.surface_sampling, source_rows=len(rows),
                seed=self.seed, case_seed=case_seed, association=association, indices='surface_source_indices.npy')
            meta['source']['surface'] = self.pressure_files[name]
        path = directory / 'volume.npy'
        if self.build_volume and (self.overwrite or not path.exists()):
            mesh = pv.read(self.cfd_files[name]); association = self.volume_location
            if association == 'point':
                if 'p' not in mesh.point_data or 'U' not in mesh.point_data:
                    if 'p' not in mesh.cell_data or 'U' not in mesh.cell_data:
                        raise ValueError(f'{name}: p and U must share an association')
                    mesh = mesh.cell_data_to_point_data(pass_cell_data=False); association = 'cell_to_point'
                xyz, fields = np.asarray(mesh.points), mesh.point_data
            else:
                xyz, fields = np.asarray(mesh.cell_centers().points), mesh.cell_data
            if 'p' not in fields or 'U' not in fields:
                raise ValueError(f'{name}: volume p and U are required')
            selected = indices(xyz, self.volume_cache_points, volume_rng)
            dtype = [('xyz', '<f4', (3,)), ('pressure', '<f4'), ('velocity', '<f4', (3,))]
            if self.include_wall_distance:
                dtype.append(('wall_distance', '<f4'))
            values = np.empty(len(selected), dtype=dtype)
            values['xyz'], values['pressure'] = xyz[selected], np.asarray(fields['p']).reshape(-1)[selected]
            values['velocity'] = np.asarray(fields['U'])[selected]
            info = dict(xyz=dict(unit='m', components=3), pressure=dict(unit='m^2/s^2', kind='kinematic'),
                        velocity=dict(unit='m/s', components=3))
            if self.include_wall_distance:
                geometry = pv.read(self.pressure_files[name])
                values['wall_distance'] = cKDTree(np.asarray(geometry.points)).query(values['xyz'], workers=1)[0]
                info['wall_distance'] = dict(unit='m', definition='nearest raw surface vertex; not exact triangle signed distance')
            atomic_npy(path, values); atomic_npy(directory / 'volume_source_indices.npy', selected)
            meta['volume'] = record(path, values, info)
            meta['sampling']['volume'] = dict(mode='random_without_replacement', source_rows=len(xyz),
                seed=self.seed, case_seed=case_seed, association=association, indices='volume_source_indices.npy')
            meta['source']['volume'] = self.cfd_files[name]
        if self.include_topology and (self.overwrite or not (directory / 'geometry_points.npy').exists()):
            mesh = surface_polydata(pv.read(self.pressure_files[name])).triangulate()
            atomic_npy(directory / 'geometry_points.npy', np.asarray(mesh.points, dtype=np.float32))
            atomic_npy(directory / 'geometry_triangles.npy', np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:])
            meta['geometry'] = dict(points='geometry_points.npy', triangles='geometry_triangles.npy',
                indexing='triangles reference geometry_points, not sampled field rows', coordinate_frame='raw')
        atomic_json(meta_path, meta)
        return dict(case=name, directory=str(directory))

    def build_cache(self, start_index=0, limit=None, log_every=1, num_workers=0):
        end = len(self) if limit is None else min(len(self), start_index + limit)
        selected = list(range(start_index, end))
        rows = (self[i] for i in selected) if num_workers <= 0 else (batch[0] for batch in DataLoader(
            Subset(self, selected), batch_size=1, num_workers=num_workers, collate_fn=identity_collate))
        for i, result in enumerate(rows, 1):
            if log_every and (i % log_every == 0 or i == len(selected)):
                print(f"[{i}/{len(selected)}] {result['case']}", flush=True)


class DrivAerNetPlusPlusWallDistanceCacheBuilder(Dataset):
    def __init__(self, root_dir=RAW_ROOT, cache_root=None, volume_cache_points=300000,
                 ids_file=None, overwrite=False, cache_profile=None):
        self.root_dir, self.cache_root = Path(root_dir), Path(cache_root or CACHE_ROOT)
        self.cache_profile, self.overwrite = cache_profile, overwrite
        requested = read_ids(ids_file)
        self.case_names = sorted(p.name for p in self.cache_root.iterdir() if p.is_dir()
            and (requested is None or p.name in requested)
            and (case_directory(self.cache_root, p.name, cache_profile) / 'volume.npy').exists())
        if requested is not None and set(self.case_names) != requested:
            raise FileNotFoundError('Requested cases missing volume pool')
    def __len__(self):
        return len(self.case_names)
    def __getitem__(self, index):
        from dataset.field_cache import read_fields
        import pyvista as pv
        name = self.case_names[index]; directory = case_directory(self.cache_root, name, self.cache_profile)
        path = directory / 'volume.npy'; values = read_fields(path)
        if 'wall_distance' in values.dtype.names and not self.overwrite:
            return dict(case=name, skipped=True)
        geometry = pv.read(self.root_dir / 'Pressure' / (name + '.vtk'))
        distances = cKDTree(np.asarray(geometry.points)).query(values['xyz'], workers=1)[0]
        dtype = [item for item in values.dtype.descr if item[0] != 'wall_distance'] + [('wall_distance', '<f4')]
        result = np.empty(len(values), dtype=dtype)
        for field in values.dtype.names:
            if field != 'wall_distance': result[field] = values[field]
        result['wall_distance'] = distances
        meta = json.loads((directory / 'metadata.json').read_text())
        fields = meta['volume']['fields']
        # Converted dense input becomes structured; column offsets no longer apply.
        fields = {k: {a: b for a, b in v.items() if a != 'columns'} for k, v in fields.items()}
        fields['wall_distance'] = dict(unit='m', definition='nearest raw surface vertex distance')
        atomic_npy(path, result); meta['volume'] = record(path, result, fields)
        atomic_json(directory / 'metadata.json', meta)
        return dict(case=name, rows=len(result))
    build_cache = DrivAerNetPlusPlusCacheBuilder.build_cache
