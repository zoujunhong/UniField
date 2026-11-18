"""Build capped, physical-coordinate train/rocket caches without padding rows.

Run on a compute node: python -m dataset.cache.full_field --workers 8
The source data and the existing training caches are read-only inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from dataset.cache.high_speed_train import HIGH_SPEED_TRAIN_CACHE_ROOT, cache_path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = Path('/lustre/home/zdhs0017/data/HighSpeedTrain')
ROCKET = Path('/lustre/home/zdhs0017/data/rocket/extracted')
TRAIN_MANIFEST = ROOT/'dataset/splits/FullFieldCache/manifests/train500.csv'
ROCKET_MANIFEST = ROOT/'dataset/splits/FullFieldCache/manifests/rocket7281.json'
DEFAULT_DEST = Path.home()/'zoujunhong/data/FullFieldCache/surface100000_volume300000'
SCHEMA = 1
TRAIN_SURFACE = ['x','y','z','p','Ux','Uy','Uz','k','omega','nut','nx','ny','nz','area']
TRAIN_VOLUME = ['x','y','z','p','Ux','Uy','Uz','k','omega','nut','surface_distance']
ROCKET_SURFACE = ['x','y','z','pressure','Cp','nx','ny','nz','area']
ROCKET_VOLUME = ['x','y','z','p','Ux','Uy','Uz','k','omega','nut']


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def atomic_npy(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f:
        np.save(f, data, allow_pickle=False)
    os.replace(tmp, path)


def select_ids(count, cap, key, seed):
    """All rows below the cap; otherwise reproducible sampling without replacement."""
    if count <= 0 or cap <= 0:
        raise ValueError(f'Invalid row count/cap: {count}/{cap}')
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    return np.sort(rng.choice(count, cap, replace=False))


def source_signature(paths):
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths}


def validate_array(array, columns):
    if array.ndim != 2 or array.shape[1] != len(columns) or not len(array):
        raise ValueError(f'Unexpected shape {array.shape}; expected {columns}')
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Cache must contain finite float32 values')


def read_train_surface(path):
    """Vectorized polygon vertex means; avoid per-face Python/VTK object creation."""
    import pyvista as pv
    from vtk.util.numpy_support import vtk_to_numpy
    from dataset.utils.load import compute_cell_area, compute_cell_normals
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData) or mesh.n_cells != mesh.GetNumberOfPolys():
        raise ValueError(f'Expected polygon-only train surface: {path}')
    polys = mesh.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    counts = np.diff(offsets)
    if not np.all(counts > 0):
        raise ValueError(f'Empty polygon in {path}')
    points = np.asarray(mesh.points, dtype=np.float32)
    centers = np.empty((mesh.n_cells, 3), dtype=np.float32)
    # Group by polygon size so np.mean uses the same per-cell float32 reduction
    # as the legacy reader, with identical rounding and no Python face loop.
    for size in np.unique(counts):
        ids = np.flatnonzero(counts == size)
        vertices = connectivity[offsets[ids,None] + np.arange(size)]
        centers[ids] = points[vertices].mean(axis=1)
    fields = [np.asarray(mesh.cell_data[name], dtype=np.float32).reshape(mesh.n_cells,-1)
              for name in ['p','U','k','omega','nut']]
    return np.column_stack([centers, *fields, compute_cell_normals(mesh), compute_cell_area(mesh)])


def finish_case(dest, record, surface, volume, surface_ids, volume_ids):
    validate_array(surface, record['surface_columns'])
    validate_array(volume, record['volume_columns'])
    for side, array, ids in [('surface',surface,surface_ids), ('volume',volume,volume_ids)]:
        expected = min(record[f'{side}_available_points'], record[f'{side}_cap'])
        if len(array) != expected or len(ids) != expected or len(np.unique(ids)) != expected:
            raise ValueError(f'{side}: count or unique-index mismatch')
        if np.any(ids < 0) or np.any(ids >= record[f'{side}_source_points']):
            raise ValueError(f'{side}: source index out of bounds')
        record[f'{side}_points'] = len(array)
        atomic_npy(dest/f'{side}.npy', array)
        atomic_npy(dest/f'{side}_indices.npy', ids)
    record['ready'] = True
    atomic_json(dest/'record.json', record)
    return record


def existing_record(dest, signature, options):
    marker = dest/'record.json'
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if (record.get('schema_version') != SCHEMA or record.get('sources') != signature
            or any(record.get(k) != v for k, v in options.items())):
        raise ValueError(f'Existing cache policy/source changed: {dest}; choose a new output directory')
    for side in ['surface','volume']:
        a = np.load(dest/f'{side}.npy', mmap_mode='r')
        validate_array(a, record[f'{side}_columns'])
        ids = np.load(dest/f'{side}_indices.npy', mmap_mode='r')
        expected = min(record[f'{side}_available_points'], record[f'{side}_cap'])
        if len(a) != expected or len(ids) != expected or len(np.unique(ids)) != expected:
            raise ValueError(f'Incomplete existing cache {dest}/{side}')
    return record


def build_train(task):
    row, output, options = task
    source = Path(row['source'])
    volume_path = Path(cache_path(HIGH_SPEED_TRAIN_CACHE_ROOT, row['family'],
                                  'volume_random', 300000, row['case'], 'fp32'))
    signature = source_signature([source/'surface.vtk', volume_path])
    dest = Path(output)/'train'/row['family']/row['case']
    old = existing_record(dest, signature, options)
    if old:
        return old
    surface = read_train_surface(source/'surface.vtk')
    if len(surface) != int(row['wall_faces']):
        raise ValueError(f"Changed raw surface count: {row['case']}")
    validate_array(surface, TRAIN_SURFACE)
    volume = np.load(volume_path, mmap_mode='r')
    validate_array(volume, TRAIN_VOLUME)
    # The legacy volume sample is already unpadded (all raw meshes exceed 300k).
    if len(volume) != min(int(row['cells']), 300000):
        raise ValueError(f'Unexpected legacy volume size: {volume_path}')
    if len(np.unique(volume[:, :3], axis=0)) != len(volume):
        raise ValueError(f'Duplicate volume coordinates: {volume_path}')
    si = select_ids(len(surface), options['surface_cap'], row['case']+'/surface', options['seed'])
    vi = select_ids(len(volume), options['volume_cap'], row['case']+'/volume', options['seed'])
    record = dict(schema_version=SCHEMA, domain='train', case=row['case'], family=row['family'],
                  split=row['split'], source_case=str(source), sources=signature,
                  surface_columns=TRAIN_SURFACE, volume_columns=TRAIN_VOLUME,
                  surface_source_points=len(surface), surface_available_points=len(surface),
                  volume_source_points=len(volume), volume_available_points=len(volume),
                  volume_raw_mesh_points=int(row['cells']),
                  surface_index_reference='raw surface.vtk cell row',
                  volume_index_reference=str(volume_path),
                  volume_sampling='reuse verified existing 300000-row random cache; no padding',
                  coordinate_frame='source coordinates, metres; no rotation or normalization',
                  pressure_units='kinematic p, m^2/s^2 (OpenFOAM incompressible)',
                  velocity_units='m/s', distance_units='m',
                  distance_definition='existing nearest full-surface face-centre distance, not exact wall distance',
                  freestream_velocity=[float(row['Ux']),float(row['Uy']),0.], **options)
    return finish_case(dest, record, surface[si], np.asarray(volume[vi]), si, vi)


def build_rocket_geometry(task):
    geometry, rows, output, options = task
    source = ROCKET/geometry
    static_paths = [source/'mesh/surface_static.npz', source/'mesh/volume_static.npz']
    with np.load(static_paths[0]) as f:
        xyz_s, normals, area = f['cell_centers'], f['normals'], f['face_area']
    with np.load(static_paths[1]) as f:
        xyz_v = f['cell_centers']
    records = []
    for row in rows:
        case = row['case']
        csource = ROCKET/case
        dest = Path(output)/'rocket'/case
        signature = source_signature(static_paths + [csource/'surface_fields.npz',csource/'volume_fields.npz',
                                                     csource/'metadata.json',csource/'field_data.json'])
        old = existing_record(dest, signature, options)
        if old:
            records.append(old)
            continue
        metadata = json.loads((csource/'metadata.json').read_text())
        with np.load(csource/'surface_fields.npz') as f:
            valid = np.flatnonzero(f['valid_mask'])
            if len(f['valid_mask']) != len(xyz_s):
                raise ValueError(f'{case}: surface topology mismatch')
            if not np.isfinite(f['pressure'][valid]).all() or not np.isfinite(f['cp'][valid]).all():
                raise ValueError(f'{case}: nonfinite valid surface fields')
            si = valid[select_ids(len(valid), options['surface_cap'], case+'/surface', options['seed'])]
            surface = np.column_stack([xyz_s[si], f['pressure'][si], f['cp'][si], normals[si], area[si]]).astype(np.float32)
        vi = select_ids(len(xyz_v), options['volume_cap'], case+'/volume', options['seed'])
        with np.load(csource/'volume_fields.npz') as f:
            fields = []
            for name in ['p','U','k','omega','nut']:
                array = f[name]
                if len(array) != len(xyz_v) or not np.isfinite(array).all():
                    raise ValueError(f'{case}: invalid raw {name}')
                fields.append(array[vi])
            volume = np.column_stack([xyz_v[vi], *fields]).astype(np.float32)
        record = dict(row, schema_version=SCHEMA, domain='rocket', source_case=str(csource), sources=signature,
                      surface_columns=ROCKET_SURFACE, volume_columns=ROCKET_VOLUME,
                      surface_source_points=len(xyz_s), surface_available_points=len(valid),
                      volume_source_points=len(xyz_v), volume_available_points=len(xyz_v),
                      surface_index_reference='surface_static.npz / surface_fields.npz original face row',
                      volume_index_reference='volume_static.npz / volume_fields.npz original cell row (not cell_id)',
                      coordinate_frame='source coordinates, metres; no rotation or normalization',
                      pressure_units='Pa (source absolute pressure)', velocity_units='m/s',
                      freestream=metadata['openfoam']['freestream'],
                      freestream_velocity=metadata['openfoam']['directions']['velocity'], **options)
        atomic_json(dest/'source_metadata.json', metadata)
        atomic_json(dest/'source_field_data.json', json.loads((csource/'field_data.json').read_text()))
        records.append(finish_case(dest, record, surface, volume, si, vi))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--surface-cap', type=int, default=100000)
    parser.add_argument('--volume-cap', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--limit', type=int, help='Smoke test: first N train cases and rocket geometries')
    parser.add_argument('--domain', choices=['all','train','rocket'], default='all')
    args = parser.parse_args()
    if not 0 < args.volume_cap <= 300000 or args.surface_cap <= 0 or args.workers <= 0:
        parser.error('Positive caps/workers required; train volume reuse supports at most 300000')
    options = dict(surface_cap=args.surface_cap, volume_cap=args.volume_cap, seed=args.seed)
    with TRAIN_MANIFEST.open() as f:
        train_rows = list(csv.DictReader(f))
    rocket_rows = json.loads(ROCKET_MANIFEST.read_text())['cases']
    if len(train_rows) != 500 or len(rocket_rows) != 7281:
        raise ValueError('Unexpected campaign manifest count')
    excluded = [dict(domain='train', case=r['case'], reason=r['status'], source=r['source'])
                for r in train_rows if r['status'] != 'active']
    excluded += [dict(domain='rocket', case=r['case'], reason='previously reviewed exclusion',
                      excluded_review_id=r['excluded_review_id']) for r in rocket_rows if not r['included']]
    selected_train = [r for r in train_rows if r['status'] == 'active']
    groups = {}
    for r in rocket_rows:
        if r['included']:
            groups.setdefault(r['geometry'], []).append(r)
    group_list = sorted(groups)
    if args.limit:
        selected_train, group_list = selected_train[:args.limit], group_list[:args.limit]
    jobs = []
    if args.domain in ['all','train']:
        jobs += [(build_train,(r,str(args.output),options)) for r in selected_train]
    if args.domain in ['all','rocket']:
        jobs += [(build_rocket_geometry,(g,groups[g],str(args.output),options)) for g in group_list]
    contract = dict(schema_version=SCHEMA, **options, dtype='float32',
                    sampling='uniform random rows without replacement; retain all eligible rows below cap; no padding',
                    train_manifest=str(TRAIN_MANIFEST), rocket_manifest=str(ROCKET_MANIFEST),
                    manifests_signature=source_signature([TRAIN_MANIFEST, ROCKET_MANIFEST]))
    contract_path = args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Output contract mismatch; choose a new output directory')
    atomic_json(contract_path, contract)
    atomic_json(args.output/'excluded.json', excluded)
    start = time.time()
    records, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn,task): str(task[0])[:150] for fn,task in jobs}
        for done, future in enumerate(as_completed(futures),1):
            try:
                result = future.result()
                records.extend(result if isinstance(result,list) else [result])
            except Exception as exc:
                errors.append(dict(task=futures[future],error=repr(exc)))
                print('ERROR', errors[-1], flush=True)
            progress = dict(done_tasks=done,total_tasks=len(jobs),cases=len(records),
                            errors=errors,elapsed_seconds=round(time.time()-start,2))
            atomic_json(args.output/'progress.json',progress)
            print(json.dumps({k:v for k,v in progress.items() if k != 'errors'}),flush=True)
    records.sort(key=lambda r:(r['domain'],r['case']))
    complete = not errors and args.limit is None and args.domain == 'all' and len(records) == 7776
    index = dict(**contract, ready=complete, cases=records, excluded=excluded, errors=errors,
                 elapsed_seconds=time.time()-start)
    atomic_json(args.output/('index.json' if complete else 'partial_index.json'), index)
    fields = ['domain','case','split','surface_source_points','surface_available_points','surface_points',
              'volume_source_points','volume_available_points','volume_points']
    with (args.output/'cases.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    print('COMPLETE',len(records),'ready',complete,'errors',len(errors),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
