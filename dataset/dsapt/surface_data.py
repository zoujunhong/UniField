from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler
from model.third_party.utonia_dsapt.transform import default as official_transform
from dataset.dsapt.car_data import GeometryDataset as CarDataset,collate,to_device
ROOT=Path(__file__).resolve().parents[2]
class GeometryDataset(Dataset):
 def __init__(self,c,names,validation=False,test_set=None):
  self.c=c;self.names=names;self.validation=validation;self.test_set=test_set
  if c['domain']=='drivaer':
   self.car=CarDataset(c,names,validation=validation)
   self.car.validation_query_filter=test_set is None
   if test_set is not None:
    group=json.loads((Path(c['test_point_sets_manifest'])).read_text())
    item=group['sets'][test_set];self.mp=Path(item['manifest']);m=json.loads(self.mp.read_text());self.qrecords={r['name']:r for r in m['records']}
    if test_set==0:
     self.nmp=Path(item['query_normal_manifest']);self.nrecords={r['name']:r for r in json.loads(self.nmp.read_text())['records']}
   return
  self.m=json.loads(Path(c['domain_manifest']).read_text());self.records={r['family']+'_'+r['name']:r for rows in self.m['splits'].values() for r in rows}
  self.transform=official_transform(scale=1.,apply_z_positive=False,normalize_coord=False)
 def __len__(self):return len(self.names)
 def __getitem__(self,key):
  epoch,index=key if isinstance(key,tuple) else (0,key)
  if self.validation:epoch=0
  c=self.c;name=self.names[index]
  if c['domain']=='drivaer':
   b=self.car[(epoch,index)]
   b['condition']=torch.zeros(2)
   if self.test_set is not None:
    from dataset.field_cache import load_surface_fields
    a,_=load_surface_fields(Path(c['data_root'])/name)
    excluded=self.car.validity['excluded'].get(name,{})
    keep=np.ones(len(a),bool);keep[excluded.get('field_cache_rows',[])]=False
    xyz=a[keep,:3].astype(np.float32);lo=xyz.min(0);hi=xyz.max(0);shift=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]],np.float32)
    with np.load(self.mp.parent/self.qrecords[name]['path']) as z:
     q=z['xyz'].copy();p=z['pressure'].copy();ids=z['indices'].copy()
     normal=z['normal'].copy() if self.test_set else np.load(self.nmp.parent/self.nrecords[name]['path'])
    b['query']=torch.from_numpy(np.concatenate(((q-shift)/c['global_scale'],normal),-1).astype(np.float32))
    b['target_surface_p']=torch.from_numpy((p.astype(np.float32)[:,None]-np.float32(-93.4098))/np.float32(120.787))
    b['surface_supervision_mask']=torch.from_numpy(~np.isin(ids,excluded.get('original_vertex_indices',[])))[:,None]
   return b
  r=self.records[name];s=self.m['stats'];domain=r['domain']
  seed=int(np.random.SeedSequence([c['validation_seed'] if self.validation else c['seed'],epoch,index]).generate_state(1)[0])
  rng=np.random.default_rng(seed);np.random.seed(seed)
  a=np.load(r['path'],mmap_mode='r');valid=np.flatnonzero(np.load(r['validity']))
  gi=rng.choice(valid,min(c['geometry_points'],len(valid)),replace=False)
  if self.test_set is None:
   candidates=valid
   if self.validation:
    cp=np.asarray(a[valid,3],np.float32)
    if domain=='train':cp=cp/(.5*sum(v*v for v in r['condition']))
    candidates=valid[cp>=-10.]
   nq=10000 if self.validation else c['surface_queries']
   if len(candidates)<nq:raise ValueError(f'{name}: insufficient valid queries ({len(candidates)} < {nq})')
   qi=rng.choice(candidates,nq,replace=False)
  else:
   with np.load(Path(c['test_points_root'])/domain/(name+'.npz')) as z:qi=z[f'set{self.test_set:02d}']
  normal_slice=slice(10,13) if domain=='train' else slice(4,7)
  xyz=np.array(a[gi,:3],np.float32);lo=xyz.min(0);hi=xyz.max(0)
  shift=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]],np.float32)
  factor=float(rng.uniform(.9,1.1)) if not self.validation and c['geometry_speed_augmentation'] else 1.
  xyz=(xyz-shift)/c['global_scale']*factor;q=(np.array(a[qi,:3],np.float32)-shift)/c['global_scale']*factor
  normal=np.array(a[gi,normal_slice],np.float32);qn=np.array(a[qi,normal_slice],np.float32)
  point=self.transform(dict(coord=xyz,normal=normal,color=np.zeros_like(xyz)))
  point={k:point[k] for k in ('coord','grid_coord','feat','offset')}
  pressure=np.array(a[qi,3:4],np.float32)
  if domain=='train':pressure/=.5*sum(v*v for v in r['condition'])
  condition=np.array(r['condition'],np.float32)
  if domain=='train':condition/=factor
  else:condition[0]/=factor
  condition=(condition-np.array(s['condition_mean'],np.float32))/np.array(s['condition_std'],np.float32)
  return dict(point=point,name=name,query=torch.from_numpy(np.concatenate((q,qn),-1)),condition=torch.from_numpy(condition),target_surface_p=torch.from_numpy(pressure),surface_supervision_mask=torch.ones((len(qi),1),dtype=torch.bool))
class EpochSampler(Sampler):
 def __init__(self,n,seed,rank=0,world=1):
  self.n=n;self.seed=seed;self.rank=rank;self.world=world;self.epoch=0;self.skip=0;self.length=(n+world-1)//world
 def __len__(self):return self.length-self.skip
 def __iter__(self):
  ids=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
  ids+=ids[:self.length*self.world-self.n]
  return iter((self.epoch,i) for i in ids[self.rank::self.world][self.skip:])
