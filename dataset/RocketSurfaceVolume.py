"""Native rocket shared-mesh fields, uniformly fitted geometry and force means."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from dataset.GeometryGenSurfaceVolume import atomic_json, atomic_npy
from dataset.validation import pressure_valid_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/lustre/home/zdhs0017/data/rocket/extracted')
CACHE = Path.home()/'zoujunhong/data/RocketCFD/volume'
SCHEMA = 1
EXCLUDED = {'g086/M020/Ap10': 'C0476', 'g170/M020/Am02': 'C0799'}


def _seed(text, seed=200099):
    return int.from_bytes(hashlib.sha256(f'{seed}:{text}'.encode()).digest()[:8], 'little')


def force_record(case):
    p = SOURCE/case
    a = np.genfromtxt(p/'force_coefficients.csv', delimiter=',', names=True)
    if len(a)<500 or not np.isfinite(np.column_stack([a['time'],a['Cd'],a['Cl']])).all():
        raise ValueError(f'Invalid force history {case}')
    t=a['time'][-500:]
    if not np.allclose(np.diff(t),1): raise ValueError(f'Nonunit last-500 steps {case}')
    m=json.loads((p/'metadata.json').read_text())
    q=json.loads((p/'quality.json').read_text())
    row=dict(case=case, geometry=m['case']['geometry_id'], mach=m['case']['mach'],
             aoa=m['case']['aoa_deg'], quality_class=q['convergence_class'],
             included=case not in EXCLUDED, excluded_review_id=EXCLUDED.get(case,''),
             coefficient_target_definition='arithmetic_mean_last_500_records',
             coefficient_window_n=500, coefficient_first_iteration=float(t[0]),
             coefficient_last_iteration=float(t[-1]), geometry_hash=m['openfoam']['geometry']['source_sha256'])
    for field in ['Cd','Cl']:
        tail=a[field][-500:]
        row.update({f'{field}_mean_last500':float(tail.mean()), f'{field}_std_last500':float(tail.std()),
                    f'{field}_min_last500':float(tail.min()), f'{field}_max_last500':float(tail.max()),
                    f'{field}_final':float(a[field][-1])})
    return row


def prepare_manifest(cache_root=CACHE):
    cache_root=Path(cache_root)
    existing=cache_root/'manifest.json'
    if existing.exists():
        value=json.loads(existing.read_text())
        if value['schema_version']!=SCHEMA or value['excluded']!=EXCLUDED:
            raise ValueError('Stale manifest policy; use a new cache root')
        return value['cases']
    cases=sorted(str(p.parent.relative_to(SOURCE)) for p in SOURCE.glob('*/*/*/metadata.json'))
    if not cases: raise FileNotFoundError(f'No raw rocket cases under {SOURCE}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(force_record,cases))
    groups=sorted(set(r['geometry_hash'] for r in rows if r['included']))
    np.random.default_rng(200099).shuffle(groups)
    nv=round(len(groups)*.1);nt=nv
    splits={h:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,h in enumerate(groups)}
    for r in rows: r['split']=splits[r['geometry_hash']] if r['included'] else 'excluded'
    cache_root.mkdir(parents=True,exist_ok=True)
    with (cache_root/'all_cases_force_mean_last500.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(cache_root/'manifest.json',dict(schema_version=SCHEMA, source_root=str(SOURCE),
                seed=200099, excluded=EXCLUDED, group_definition='source_sha256', cases=rows))
    print('MANIFEST',len(rows),'included',sum(r['included'] for r in rows),flush=True)
    return rows


def build_geometry(args):
    geometry, rows, cache_root=args
    from scipy.spatial import cKDTree
    import pyvista as pv
    from dataset.GeometryGenSurface import _dense_surface_candidates,_farthest_point_indices
    cache_root=Path(cache_root);dest=cache_root/'geometries'/geometry
    marker=dest/'record.json'
    if marker.exists(): return json.loads(marker.read_text())
    start=time.time();source=SOURCE/geometry
    surface=pv.read(source/'mesh/vehicle_surface.vtp')
    vertices=np.asarray(surface.points,dtype=np.float64)
    with np.load(source/'mesh/surface_static.npz') as a:
        faces=a['faces'];centres=a['cell_centers'].astype(np.float64)
        normals=a['normals'].astype(np.float64);area=a['face_area'].astype(np.float64)
    lo=vertices.min(axis=0);hi=vertices.max(axis=0);center=(lo+hi)/2
    scale=float(np.min(np.array([2.,2.,1.])/(hi-lo)))
    points=(vertices-center)*scale
    np.testing.assert_array_less(np.max(np.abs(points),axis=0),np.array([1.,1.,.5])+1e-6)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    valid=np.isfinite(centres).all(axis=1)&np.isfinite(normals).all(axis=1)&(area>0)
    if not valid.all():raise ValueError(f'Invalid static surface {geometry}')
    rng=np.random.default_rng(_seed(geometry))
    candidate,norm=_dense_surface_candidates(points,faces,normals,area,32768,rng)
    ids=_farthest_point_indices(candidate,8192)
    inputs=np.column_stack([candidate[ids],norm[ids]]).astype(np.float32)
    xyz_surface=((centres-center)*scale).astype(np.float32)
    with np.load(source/'mesh/volume_static.npz') as a:
        xyz=((a['cell_centers'].astype(np.float64)-center)*scale).astype(np.float32)
        cell_id=a['cell_id'].copy()
    if not np.isfinite(xyz).all():raise ValueError(f'Invalid volume centres {geometry}')
    tree=cKDTree(xyz_surface)
    distance=tree.query(xyz,workers=1)[0].astype(np.float32)
    for name,array in [('input.npy',inputs),('surface_xyz.npy',xyz_surface),
                       ('volume_xyz.npy',xyz),('distance.npy',distance),('volume_cell_ids.npy',cell_id)]:
        atomic_npy(dest/name,array)
    case_records=[]
    for row in rows:
        case=row['case'];csource=SOURCE/case;cdest=cache_root/'cases'/case
        m=json.loads((csource/'metadata.json').read_text())
        v=np.asarray(m['openfoam']['directions']['velocity'],dtype=np.float64)
        speed=float(np.linalg.norm(v));direction=v/speed
        if abs(direction[1])>1e-8:raise ValueError('Unexpected nonzero sideslip')
        # Row vectors: v @ rotation = [speed,0,0]. Same rigid rotation for all data.
        rotation=np.array([[direction[0],0,-direction[2]],[0,1,0],[direction[2],0,direction[0]]])
        np.testing.assert_allclose(v@rotation,[speed,0,0],atol=1e-5)
        with np.load(csource/'surface_fields.npz') as a:
            cp=a['cp'].astype(np.float32);surface_mask=a['valid_mask'].astype(bool)
        if not surface_mask.any() or not np.isfinite(cp[surface_mask]).all():raise ValueError(case)
        surface_ids=np.flatnonzero(surface_mask).astype(np.int32)
        with np.load(csource/'volume_fields.npz') as a:
            u=(a['U'].astype(np.float64)@rotation/speed-np.array([1.,0.,0.])).astype(np.float32)
            free=m['openfoam']['freestream']
            volume_cp=((a['p'].astype(np.float64)-float(free['pressure']))/(.5*float(free['density'])*speed**2)).astype(np.float32)
        values=np.column_stack([u,volume_cp]).astype(np.float32)
        if len(values)!=len(xyz) or not np.isfinite(values).all():raise ValueError(case)
        aligned=xyz@rotation
        bounds=points@rotation;bmin=bounds.min(axis=0);bmax=bounds.max(axis=0)
        near=(distance<=.5)|np.all((aligned>=bmin-[.5,.5,.5])&(aligned<=bmax+[2.,.5,.5]),axis=1)
        near_ids=np.flatnonzero(near).astype(np.int32)
        if not len(near_ids):raise ValueError(f'Empty near pool {case}')
        for name,array in [('surface_cp.npy',cp),('surface_ids.npy',surface_ids),
                           ('volume_fields.npy',values),('near_ids.npy',near_ids)]: atomic_npy(cdest/name,array)
        record=dict(**row,scale=scale,center=center.tolist(),speed=speed,rotation=rotation.tolist(),
                    volume_points=len(xyz),surface_points=len(surface_ids),near_points=len(near_ids))
        atomic_json(cdest/'record.json',record)
        materialize_case(cache_root,record)
        case_records.append(record)
    result=dict(schema_version=SCHEMA, geometry=geometry, scale=scale, center=center.tolist(),
                fitted_bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
                cases=case_records,elapsed_seconds=time.time()-start)
    atomic_json(marker,result)
    print('CACHED',geometry,len(case_records),round(time.time()-start,1),flush=True)
    return result


def materialize_case(cache_root, record):
    """Write self-contained physical field rows without changing historical values."""
    cache_root=Path(cache_root);p=cache_root/'cases'/record['case'];g=cache_root/'geometries'/record['geometry']
    rot=np.asarray(record['rotation'],dtype=np.float32)
    if not (p/'surface.npy').exists():
        ids=np.load(p/'surface_ids.npy')
        xyz=np.load(g/'surface_xyz.npy',mmap_mode='r')[ids]@rot
        cp=np.load(p/'surface_cp.npy',mmap_mode='r')[ids]
        atomic_npy(p/'surface.npy',np.column_stack([xyz,cp]).astype(np.float32))
    if not (p/'volume.npy').exists():
        xyz=np.load(g/'volume_xyz.npy',mmap_mode='r')@rot
        fields=np.load(p/'volume_fields.npy',mmap_mode='r')
        distance=np.load(g/'distance.npy',mmap_mode='r')
        atomic_npy(p/'volume.npy',np.column_stack([xyz,fields,distance]).astype(np.float32))
    atomic_json(p/'metadata.json',dict(record,surface_columns=['x','y','z','Cp'],
        volume_columns=['x','y','z','Ustar_x','Ustar_y','Ustar_z','Cp','raw_distance_model_units'],
        source=str(SOURCE/record['case']),geometry_input=str(g/'input.npy'),coordinate_frame='freestream aligned'))


def build_cache(cache_root=CACHE,workers=4,limit=None):
    cache_root=Path(cache_root)
    rows=prepare_manifest(cache_root)
    by_geometry={}
    for row in rows:
        if row['included']:by_geometry.setdefault(row['geometry'],[]).append(row)
    groups=sorted(by_geometry)
    if limit:groups=groups[:limit]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(build_geometry,(g,by_geometry[g],str(cache_root))) for g in groups]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            atomic_json(cache_root/'progress.json',dict(done=len(results),total=len(groups),stage='cache'))
    records=sorted([r for g in results for r in g['cases']],key=lambda r:r['case'])
    index=dict(schema_version=SCHEMA,ready=len(groups)==len(by_geometry),cases=records,
               coefficient_target_definition='Cd_mean_last500, Cl_mean_last500',
               scale_definition='model_xyz=(source_xyz-center)*scale; then align incoming velocity to +x',
               condition=['speed_m_s','physical_to_model_scale'],source_root=str(SOURCE))
    atomic_json(cache_root/('index.json' if index['ready'] else 'preview_index.json'),index)
    print('CACHE_COMPLETE',len(records),'ready',index['ready'],flush=True)
    return index


class RocketSurfaceVolumeDataset(TorchDataset):
    dataset_name='RocketSurfaceVolume'

    def __init__(self,cache_root=CACHE,split='train',num_surface_points=8192,num_surface_query_points=2048,
                 num_query_points=8192,surface_input_list=('xyz','normal'),surface_target_list=('p','surface_distance'),
                 volume_target_list=('U','p','surface_distance'),coefficient_target_list=(),seed=200099,
                 deterministic=False,route=0,repeat=1,allow_preview=False):
        self.cache_root=Path(cache_root);self.seed=int(seed);self.deterministic=bool(deterministic)
        self._rng=None
        self.route=int(route);self.repeat=int(repeat)
        index=json.loads((self.cache_root/('preview_index.json' if allow_preview else 'index.json')).read_text())
        if index['schema_version']!=SCHEMA or (not allow_preview and not index['ready']):raise ValueError('Cache not ready')
        self.records=[r for r in index['cases'] if split=='all' or r['split']==split]
        if not self.records:raise ValueError(f'Empty split {split}')
        if list(surface_input_list)!=['xyz','normal'] or num_surface_points!=8192:raise ValueError('Expected FPS8192 xyz+normal')
        if not set(surface_target_list)<= {'p','surface_distance'} or not set(volume_target_list)<= {'U','p','surface_distance'}:raise ValueError('Unexpected fields')
        self.surface_target_list=list(surface_target_list);self.volume_target_list=list(volume_target_list)
        if list(coefficient_target_list) not in [[],['Cd','Cl']]:raise ValueError('Coefficients use Cd,Cl last500 means')
        self.coefficient_target_list=list(coefficient_target_list)
        self.ns=int(num_surface_query_points);self.nv=int(num_query_points)
        width=lambda fields:sum(3 if f=='U' else 1 for f in fields)
        self.interface_signature=(8192,self.ns,self.nv,6,width(surface_target_list),width(volume_target_list),2,len(self.coefficient_target_list))
        print(f'RocketSurfaceVolume: {len(self.records)} cases, split={split}',flush=True)

    def __len__(self):return len(self.records)*self.repeat

    def __getitem__(self,index):
        r=self.records[index%len(self.records)];g=self.cache_root/'geometries'/r['geometry'];p=self.cache_root/'cases'/r['case']
        if self.deterministic:
            rng=np.random.default_rng(_seed(f'{self.seed}:{index}'))
        else:
            if self._rng is None:self._rng=np.random.default_rng(torch.initial_seed())
            rng=self._rng
        rot=np.asarray(r['rotation'],dtype=np.float32)
        inputs=np.load(g/'input.npy').copy();inputs[:,:3]=inputs[:,:3]@rot;inputs[:,3:]=inputs[:,3:]@rot
        surface=np.load(p/'surface.npy',mmap_mode='r')
        surface=pressure_valid_rows(self,surface,surface[:,3])
        ids=rng.choice(len(surface),self.ns,replace=len(surface)<self.ns)
        selected=np.array(surface[ids],dtype=np.float32)
        sq=selected[:,:3].copy()
        sv={'p':selected[:,3:4],'surface_distance':np.zeros((self.ns,1),dtype=np.float32)}
        st=np.concatenate([sv[f] for f in self.surface_target_list],axis=1) if self.surface_target_list else np.empty((self.ns,0),dtype=np.float32)
        volume=np.load(p/'volume.npy',mmap_mode='r')
        if getattr(self,'validation_cp_min',None) is not None:
            # Validation is uniformly random over the complete admissible query pool.
            volume=pressure_valid_rows(self,volume,volume[:,6])
            vi=rng.choice(len(volume),self.nv,replace=len(volume)<self.nv)
        else:
            near=np.load(p/'near_ids.npy',mmap_mode='r');near_count=self.nv//2
            vi=np.concatenate([rng.choice(len(volume),self.nv-near_count,replace=len(volume)<self.nv-near_count),
                               rng.choice(near,near_count,replace=len(near)<near_count)])
            rng.shuffle(vi)
        selected=np.array(volume[vi],dtype=np.float32)
        vq=selected[:,:3].copy()
        vv={'U':selected[:,3:6],'p':selected[:,6:7],'surface_distance':np.sqrt(np.maximum(selected[:,7:8],0))}
        vt=np.concatenate([vv[f] for f in self.volume_target_list],axis=1) if self.volume_target_list else np.empty((self.nv,0),dtype=np.float32)
        coefficients=np.array([r[f'{f}_mean_last500'] for f in self.coefficient_target_list],dtype=np.float32)
        return (torch.from_numpy(inputs),torch.from_numpy(sq),torch.from_numpy(st),torch.from_numpy(vq),torch.from_numpy(vt),
                torch.tensor([r['speed'],r['scale']],dtype=torch.float32),torch.tensor(self.route),torch.from_numpy(coefficients))


Dataset=RocketSurfaceVolumeDataset

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,default=CACHE)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    build_cache(**vars(parser.parse_args()))
