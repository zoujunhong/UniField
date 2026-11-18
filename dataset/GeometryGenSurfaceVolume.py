"""Rotated-world GeometryGen joint U/Cp/sqrt(distance) dataset and durable cache."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from dataset.validation import pressure_valid_rows
import time
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

ROOT = Path(os.environ.get('GEOMETRYGEN_ROOT', '/lustre/home/zdhs0017/data/GeometryGen')).expanduser()
SOURCE = Path(os.environ.get('GEOMETRYGEN_JOINT_SOURCE_ROOT', str(ROOT/'results'))).expanduser()
CACHE = Path(os.environ.get('GEOMETRYGEN_JOINT_CACHE_ROOT', '~/zoujunhong/data/GeometryGen/volume')).expanduser()
SCHEMA = 1
POOL_CAPS = {'surface.npy': 16384, 'volume.npy': 65536}


def compact_cached_case(destination, record, seed=200099):
    """Rebuild query pools from full caches without changing FPS inputs or distances."""
    destination = Path(destination)
    if record.get('query_pool_caps') == [8192, 16384, 65536] and record.get('source_indices_saved') is False:
        return record
    for name, cap in POOL_CAPS.items():
        rows = np.load(destination/name, mmap_mode='r')
        record.setdefault('complete_'+name.split('.')[0]+'_points', len(rows))
        key = f'{seed}:{record["case_id"]}:compact:{name}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
        if len(rows) > cap:
            chosen = np.array(rows[rng.choice(len(rows), cap, replace=False)], copy=True)
            del rows
            atomic_npy(destination/name, chosen)
            count = len(chosen)
        else:
            count = len(rows)
            del rows
        record[name.split('.')[0]+'_points'] = count
    # Explicitly requested removal of source indices; raw VTK remains recoverable.
    for name in ['surface_ids.npy', 'volume_ids.npy']:
        (destination/name).unlink(missing_ok=True)
    record.update(query_pool_caps=[8192,16384,65536], input_points=8192,
                  source_indices_saved=False, query_sampling='uniform random without replacement; all if below cap',
                  compact_sampling_seed=seed)
    atomic_json(destination/'record.json', record)
    atomic_json(destination/'metadata.json', dict(record, columns=['x','y','z','Ustar_x','Ustar_y','Ustar_z','Cp','raw_distance_m']))
    return record


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def atomic_npy(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temp.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temp, path)


def _build_case(case_path, cache_root, seed=200099):
    import pyvista as pv
    from scipy.spatial import cKDTree
    from dataset.GeometryGenSurface import _farthest_point_indices, _dense_surface_candidates
    from dataset.utils.geometry_centres import surface_cell_geometry, volume_cell_geometry

    source = Path(case_path); destination = Path(cache_root)/'cases'/source.name
    marker = destination/'record.json'
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing.get('schema_version') == SCHEMA and existing.get('status') == 'accepted':
            for name, cols in [('surface.npy',8),('volume.npy',8),('input.npy',6)]:
                a = np.load(destination/name, mmap_mode='r')
                if a.ndim != 2 or a.shape[1] != cols or not len(a):
                    raise ValueError(f'Invalid existing cache {destination/name}')
            return compact_cached_case(destination, existing, seed)
        if existing.get('schema_version') == SCHEMA and existing.get('status') == 'excluded':
            return existing
        raise ValueError(f'Unexpected cache schema at {marker}')
    start = time.monotonic()
    metadata = json.loads((source/'metadata.json').read_text())
    velocity = np.asarray(metadata['U_inf_world_m_s'], dtype=np.float64)
    if velocity.shape != (3,) or velocity[0] <= 0 or not np.allclose(velocity[1:],0):
        raise ValueError(f'Expected positive x-only freestream: {velocity}')
    ux = float(velocity[0])
    surface = pv.read(source/'surface_500.vtp')
    centres, areas, surface_ok = surface_cell_geometry(surface)
    normals = np.asarray(surface.cell_data['normal_world'],dtype=np.float64)
    surface_u = np.asarray(surface.cell_data['U'],dtype=np.float64)
    surface_cp = np.asarray(surface.cell_data['Cp'],dtype=np.float64).reshape(-1)
    surface_ok &= (np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals,axis=1)>0)
                   & np.isfinite(surface_u).all(axis=1) & np.isfinite(surface_cp))
    volume = pv.read(source/'volume_500.vtu')
    volume_xyz, volume_sizes, volume_ok, centre_audit = volume_cell_geometry(volume)
    volume_u = np.asarray(volume.cell_data['U'],dtype=np.float64)
    volume_cp = np.asarray(volume.cell_data['Cp'],dtype=np.float64).reshape(-1)
    volume_ok &= np.isfinite(volume_u).all(axis=1) & np.isfinite(volume_cp)
    if not np.any(surface_ok) or not np.any(volume_ok):
        raise ValueError('Case has no valid surface or volume points')
    record = dict(schema_version=SCHEMA, case_id=source.name, status='accepted',
                  source=str(source), mach=metadata['mach'], alpha_deg=metadata['alpha_deg'],
                  beta_deg=metadata['beta_deg'], ux_m_s=ux, scale=1.0,
                  surface_points=int(surface_ok.sum()), volume_points=int(volume_ok.sum()),
                  filtered_surface_points=int((~surface_ok).sum()),
                  filtered_volume_points=int((~volume_ok).sum()), centre_audit=centre_audit,
                  surface_cp_range=[float(surface_cp[surface_ok].min()),float(surface_cp[surface_ok].max())],
                  volume_cp_range=[float(volume_cp[volume_ok].min()),float(volume_cp[volume_ok].max())])
    ranges = record['surface_cp_range'] + record['volume_cp_range']
    if min(ranges) < -8 or max(ranges) > 5:
        record.update(status='excluded', reason='Cp outside [-8,5] AFTER independent centre repair and point filtering',
                      cache_seconds=time.monotonic()-start)
        atomic_json(marker,record)
        return record
    case_seed = int.from_bytes(hashlib.sha256(f'{seed}:{source.name}'.encode()).digest()[:8],'little')
    rng = np.random.default_rng(case_seed)
    triangles = surface.triangulate()
    faces = np.asarray(triangles.faces).reshape(-1,4)
    if np.any(faces[:,0] != 3):
        raise ValueError('Triangulation did not produce triangles')
    face_ids = faces[:,1:]
    vertices = np.asarray(triangles.points,dtype=np.float64)
    t = vertices[face_ids]
    triangle_areas = np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    triangle_normals = np.asarray(triangles.cell_data['normal_world'],dtype=np.float64)
    good = np.isfinite(triangle_areas) & (triangle_areas>0) & np.isfinite(triangle_normals).all(axis=1)
    candidate_xyz,candidate_normal = _dense_surface_candidates(vertices,face_ids[good],triangle_normals[good],
                                                               triangle_areas[good],32768,rng)
    selected = _farthest_point_indices(candidate_xyz,8192)
    selected_normals = candidate_normal[selected]
    selected_normals /= np.maximum(np.linalg.norm(selected_normals,axis=1,keepdims=True),1e-20)
    geometry_input = np.concatenate([candidate_xyz[selected],selected_normals],axis=1).astype(np.float32)
    xyz = volume_xyz[volume_ok]
    tree = cKDTree(centres[surface_ok])
    distance = np.empty(len(xyz),dtype=np.float64)
    for first in range(0,len(xyz),131072):
        distance[first:first+131072] = tree.query(xyz[first:first+131072],workers=1)[0]
    surface_rows = np.column_stack([centres[surface_ok],(surface_u[surface_ok]-velocity)/ux,
                                   surface_cp[surface_ok],np.zeros(surface_ok.sum())]).astype(np.float32)
    volume_rows = np.column_stack([xyz,(volume_u[volume_ok]-velocity)/ux,
                                  volume_cp[volume_ok],distance]).astype(np.float32)
    if not np.isfinite(surface_rows).all() or not np.isfinite(volume_rows).all() or np.any(distance<0):
        raise ValueError('Nonfinite or negative-distance cache')
    destination.mkdir(parents=True,exist_ok=True)
    for name,array in [('input.npy',geometry_input),('surface.npy',surface_rows),('volume.npy',volume_rows)]:
        atomic_npy(destination/name,array)
    record.update(cache_seconds=time.monotonic()-start,distance_range_m=[float(distance.min()),float(distance.max())],
                  surface_normalized_velocity_max_error=float(np.max(np.abs(surface_rows[:,3:6]-[-1,0,0]))))
    return compact_cached_case(destination, record, seed)


def build_cache(source_root=SOURCE,cache_root=CACHE,workers=2,limit=None,seed=200099):
    source_root,cache_root=Path(source_root).expanduser(),Path(cache_root).expanduser()
    cases=sorted(p for p in source_root.glob('case_*') if (p/'metadata.json').is_file()
                 and (p/'surface_500.vtp').is_file() and (p/'volume_500.vtu').is_file())
    full_count=len(cases)
    # Compact exports have no source VTK. Preserve them when extending local VTK caches.
    source_ids={p.name for p in cases}
    retained=[]
    previous_path=cache_root/'index.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        for record in previous.get('cases', [])+previous.get('excluded', []):
            if record['case_id'] in source_ids:
                continue
            if record.get('cfd_status')!='complete' or not record.get('raw_distance_stored'):
                raise ValueError(f"Existing cache has a missing VTK source: {record['case_id']}")
            folder=cache_root/'cases'/record['case_id']
            if not all((folder/name).is_file() for name in ['input.npy','surface.npy','volume.npy','record.json']):
                raise ValueError(f'Incomplete imported cache: {folder}')
            retained.append(record)
    if limit is not None: cases=cases[:int(limit)]
    records=[]
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures={pool.submit(_build_case,str(p),str(cache_root),seed):p for p in cases}
        for future in concurrent.futures.as_completed(futures):
            case=futures[future]
            # Structural/read errors stop the pipeline; do not silently reduce the cohort.
            record=future.result(); records.append(record)
            if len(records)%10==0 or len(records)==len(cases):
                counts={s:sum(r['status']==s for r in records) for s in ['accepted','excluded']}
                progress=dict(stage='cache',done=len(records),total=len(cases),counts=counts,
                              elapsed_seconds=time.monotonic()-started,last_case=record['case_id'])
                atomic_json(cache_root/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    records.extend(retained)
    records.sort(key=lambda r:r['case_id'])
    index=dict(schema_version=SCHEMA,ready=len(cases)==full_count,source_root=str(source_root),
               coordinate_frame='rotated CFD world; original metres; no coordinate normalization',
               condition=['Ux_m_s','scale=1'],input_points=8192,fields=['U','p','surface_distance'],
               columns=['x','y','z','Ustar_x','Ustar_y','Ustar_z','Cp','raw_distance_m'],
               distance_reference='ALL valid geometric face centres from full surface VTK',
               cases=[r for r in records if r['status']=='accepted'],
               excluded=[r for r in records if r['status']=='excluded'],
               total_source_cases=full_count+len(retained),source_available_cases=full_count+len(retained),
               cached_source_cases=len(records),raw_vtk_cases=full_count,cache_only_cases=len(retained),
               query_pool_caps=[8192,16384,65536],source_indices_saved=False,
               coverage_complete=len(cases)==full_count,elapsed_seconds=time.monotonic()-started)
    atomic_json(cache_root/('index.json' if index['ready'] else 'preview_index.json'),index)
    return index


class GeometryGenSurfaceVolumeDataset(TorchDataset):
    dataset_name='GeometryGenSurfaceVolume'

    def __init__(self,cache_root=CACHE,split='all',num_surface_points=8192,num_surface_query_points=2048,
                 num_query_points=8192,surface_input_list=('xyz','normal'),
                 surface_target_list=('U','p','surface_distance'),volume_target_list=('U','p','surface_distance'),
                 seed=200099,deterministic=False,route=0,repeat=1,limit=None,allow_preview=False,
                 surface_distance_target_transform='sqrt',split_file=None):
        self.cache_root=Path(cache_root).expanduser()
        name='preview_index.json' if allow_preview else 'index.json'
        index=json.loads((self.cache_root/name).read_text())
        if index['schema_version']!=SCHEMA or (not allow_preview and not index['ready']):
            raise ValueError('Cache is not complete/current')
        if index.get('coverage_complete') is False:
            warnings.warn(f"Cache covers {len(index['cases'])} accepted and "
                          f"{len(index.get('excluded', []))} excluded cases out of "
                          f"{index.get('source_available_cases', 'unknown')} raw cases; "
                          "uncached cases are not used for training.", RuntimeWarning, stacklevel=2)
        if split not in ('all','train','val'): raise ValueError(f'Unknown split: {split}')
        records=index['cases']
        if split!='all':
            if split_file is None: raise ValueError('train/val require an explicit frozen split_file')
            partition=json.loads(Path(split_file).expanduser().read_text())
            train_ids,val_ids=partition['train'],partition['val']
            if len(set(train_ids))!=len(train_ids) or len(set(val_ids))!=len(val_ids) or set(train_ids)&set(val_ids):
                raise ValueError('Split contains duplicate or overlapping case IDs')
            by_id={r['case_id']:r for r in records}
            if set(train_ids)|set(val_ids)!=set(by_id):
                raise ValueError('Accepted cohort changed since frozen split was created')
            records=[by_id[cid] for cid in partition[split]]
        if list(surface_input_list)!=['xyz','normal'] or int(num_surface_points)!=8192:
            raise ValueError('Expected 8192 xyz+normal FPS inputs')
        allowed={'U','p','surface_distance'}
        if not set(surface_target_list).issubset(allowed) or not set(volume_target_list).issubset(allowed):
            raise ValueError('Supported targets: U, p (Cp), surface_distance')
        self.surface_target_list=list(surface_target_list)
        self.volume_target_list=list(volume_target_list)
        if surface_distance_target_transform!='sqrt': raise ValueError('Distance target must be sqrt')
        self.records=records[:int(limit)] if limit is not None else records
        if not self.records: raise ValueError('No accepted cases')
        self.seed=int(seed);self.deterministic=bool(deterministic);self.route=int(route);self.repeat=int(repeat)
        self.num_surface_points=8192;self.num_surface_query_points=int(num_surface_query_points)
        self.num_query_points=int(num_query_points)
        width=lambda fs:sum(3 if f=='U' else 1 for f in fs)
        self.interface_signature=(8192,self.num_surface_query_points,self.num_query_points,6,width(surface_target_list),width(volume_target_list),2)

    def __len__(self):return len(self.records)*self.repeat

    def __getitem__(self,index):
        r=self.records[index%len(self.records)];p=self.cache_root/'cases'/r['case_id']
        rng=np.random.default_rng(self.seed+index) if self.deterministic else np.random.default_rng(int(np.random.randint(0, 2**32, dtype=np.uint32)))
        def sample(name,count,fields):
            rows=np.load(p/name,mmap_mode='r')
            rows=pressure_valid_rows(self,rows,rows[:,6])
            ids=rng.choice(len(rows),size=count,replace=len(rows)<count)
            chosen=np.array(rows[ids],dtype=np.float32,copy=True)
            values={'U':chosen[:,3:6], 'p':chosen[:,6:7], 'surface_distance':np.sqrt(np.maximum(chosen[:,7:8],0))}
            target=np.concatenate([values[f] for f in fields],axis=1) if fields else np.empty((count,0),dtype=np.float32)
            return torch.from_numpy(chosen[:,:3].copy()),torch.from_numpy(target)
        sq,st=sample('surface.npy',self.num_surface_query_points,self.surface_target_list)
        vq,vt=sample('volume.npy',self.num_query_points,self.volume_target_list)
        return (torch.from_numpy(np.load(p/'input.npy').copy()),sq,st,vq,vt,
                torch.tensor([r['ux_m_s'],1.0],dtype=torch.float32),torch.tensor(self.route,dtype=torch.long),
                torch.empty(0,dtype=torch.float32))


Dataset=GeometryGenSurfaceVolumeDataset

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=SOURCE)
    parser.add_argument('--cache-root',type=Path,default=CACHE)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    result=build_cache(**vars(args))
    print(json.dumps(dict(ready=result['ready'],accepted=len(result['cases']),excluded=len(result['excluded']))),flush=True)
