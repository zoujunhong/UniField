from pathlib import Path
import hashlib,json
import numpy as np
import torch
from dataset.dsapt.surface_data import GeometryDataset as SurfaceDataset, EpochSampler
from dataset.dsapt.car_data import collate,to_device
from dataset.field_cache import load_surface_fields,read_fields
class GeometryDataset(SurfaceDataset):
 def __getitem__(self,key):
  b=super().__getitem__(key)
  if self.c.get("surface_only") or not self.c.get("volume_queries",0):return b
  epoch,index=key if isinstance(key,tuple) else (0,key)
  if self.validation:epoch=0
  c=self.c;name=self.names[index];directory=Path(c['data_root'])/name
  a,_=load_surface_fields(directory);keep=np.ones(len(a),bool)
  keep[self.car.validity['excluded'].get(name,{}).get('field_cache_rows',[])]=False
  xyz=a[keep,:3].astype(np.float32);lo=xyz.min(0);hi=xyz.max(0)
  shift=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]],np.float32)
  volume=read_fields(directory/'volume.npy')
  st=self.test_set
  if st is None:seed=[c['validation_seed'] if self.validation else c['seed'],epoch,index,7431];n=10000 if self.validation else c['volume_queries']
  else:seed=[c['volume_test_seed'],st,index,7431];n=c['volume_test_queries']
  candidates=np.arange(len(volume))
  if self.validation and st is None:
   candidates=candidates[np.asarray(volume['pressure'],np.float32)>=-10.*.5*float(c.get('u_ref',30.))**2]
  if len(candidates)<n:raise ValueError(f'{name}: insufficient valid volume queries ({len(candidates)} < {n})')
  ids=np.random.default_rng(np.random.SeedSequence(seed)).choice(candidates,n,replace=False)
  q=(volume['xyz'][ids].astype(np.float32)-shift)/np.float32(c['global_scale'])
  # Zero padding supplies no fictitious volume normal; xyz remains the only volume geometry attribute.
  b['query']=torch.cat((b['query'],torch.from_numpy(np.concatenate((q,np.zeros_like(q)),axis=-1))),dim=0)
  stats=c['label_stats']
  for field,source,stat in [('volume_p','pressure','volume_pressure'),('volume_u','velocity','volume_velocity')]:
   y=volume[source][ids].astype(np.float32).reshape(n,-1)
   b['target_'+field]=torch.from_numpy((y-np.array(stats[stat+'_mean'],np.float32))/np.array(stats[stat+'_std'],np.float32))
  if not all(torch.isfinite(b[k]).all() for k in ('query','target_volume_p','target_volume_u')):raise ValueError(name+': nonfinite joint-field data')
  if st is not None:
   dest=Path(c['volume_test_indices_root'])/f'set{st:02d}';dest.mkdir(parents=True,exist_ok=True);path=dest/(name+'.npy')
   if path.exists():
    if not np.array_equal(np.load(path),ids):raise ValueError('Volume test indices mismatch')
   else:
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,ids)
    tmp.replace(path)
  return b
