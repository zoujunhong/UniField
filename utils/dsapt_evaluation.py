"""Final DSAPT EMA benchmark: official test split, ten fixed query sets, cache only."""
from pathlib import Path
import sys,os,json,math,fcntl

def deny_mesh(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).suffix.lower() in {'.vtk','.vtp','.vtu','.pvtu','.stl'}:raise RuntimeError('Cache-only evaluation')

def aggregate(rows):
 import numpy as np
 n=sum(r['n'] for r in rows);sse=sum(r['sse'] for r in rows);sae=sum(r['sae'] for r in rows);points=sum(r['points'] for r in rows)
 sums=np.asarray([r['sum_components'] for r in rows]).sum(0);squares=np.asarray([r['square_components'] for r in rows]).sum(0)
 sst=float((squares-sums*sums/points).sum());mse=sse/n
 return dict(mse=mse,mae=sae/n,rmse=math.sqrt(mse),r2=1-sse/max(sst,1e-12),rel_l1=sum(r['rel_l1'] for r in rows)/len(rows),rel_l2=sum(r['rel_l2'] for r in rows)/len(rows),points=points)
def evaluate_dsapt(config):
 sys.addaudithook(deny_mesh)
 import numpy as np,torch
 from model.DSAPT.network import make_models,EMA,prepare_query_initial_features
 from dataset.dsapt.data import GeometryDataset,collate,to_device
 if int(config['dsapt'].get('volume_queries',0))==0:
  from dataset.dsapt.surface_data import GeometryDataset
 from utils.training_dsapt import atomic_json
 from utils.dsapt_metrics import FIELDS,STAT_KEYS
 FIELDS=FIELDS if int(config['dsapt'].get('volume_queries',0))>0 else ('surface_p',)
 c=dict(config['dsapt']);run=Path(config['train']['log_dir']);out=Path(config.get('test',{}).get('output_dir') or run/'evaluation');(out/'cases').mkdir(parents=True,exist_ok=True)
 c['volume_test_indices_root']=str(out/'volume_indices')
 checkpoint=Path(config.get('test',{}).get('checkpoint') or Path(config['train']['saveroot'])/'latest.pth').expanduser()
 lock=(out/'eval.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
 torch.set_num_threads(4);torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if config.get('test',{}).get('require_complete',True) and saved['step']!=int(config['train']['total_steps']):raise ValueError('Final DSAPT test requires the complete training budget')
 enc,dec=make_models(c);enc.load_state_dict(saved['encoder']);dec.load_state_dict(saved['decoder']);ema=EMA(enc,dec,c['ema_decay'])
 if 'ema' in saved:ema.load_state_dict(saved['ema'])
 elif saved.get('weights')!='EMA':raise ValueError('Final test requires EMA weights')
 saved_step=int(saved['step']);del saved
 import hashlib
 hasher=hashlib.sha256()
 with checkpoint.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):hasher.update(chunk)
 identity=dict(step=saved_step,weights='EMA',checkpoint_sha256=hasher.hexdigest(),volume_seed=c['volume_test_seed'],protocol='official_test_10_query_sets_no_cp_filter_v1')
 if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
 atomic_json(out/'identity.json',identity)
 from model.DSAPT import apply_optimization,window_block_compile
 apply_optimization.apply(enc,'all');window_block_compile.patch(enc);window_block_compile.sparse_fp16(enc)
 decf=torch.compile(dec,fullgraph=True,dynamic=True);enc.eval();dec.eval()
 names=json.loads(Path(c['manifest_file']).read_text())['test']
 if len(names)!=int(config.get('test',{}).get('expected_cases',1154)):raise ValueError(f'Expected complete official test split, got {len(names)} cases')
 datasets=[GeometryDataset(c,names,validation=True,test_set=s) for s in range(10)];rows=[]
 with torch.no_grad(),ema.apply():
  for i,name in enumerate(names):
   path=out/'cases'/(name+'.json')
   if path.exists():rows.append(json.loads(path.read_text()));continue
   b=to_device(collate([datasets[0][i]]),'cuda');torch.manual_seed(c['validation_seed']+i);torch.cuda.manual_seed_all(c['validation_seed']+i)
   with torch.autocast('cuda',dtype=torch.float16):f,p,n,_=enc(b)
   sets=[]
   for st,ds in enumerate(datasets):
    if st:b=to_device(collate([ds[i]]),'cuda')
    ns=b['target_surface_p'].shape[1]
    with torch.autocast('cuda',dtype=torch.float16):
     z=prepare_query_initial_features(f,p,n,b['query'],c);pred=decf(f,p,None,b['query'],z,ns)
    fields={}
    for field in FIELDS:
     y=b['target_'+field][0].double();pr=pred[field][0].double()
     if field=='surface_p':
      mask=b['surface_supervision_mask'][0,:,0];y=y[mask];pr=pr[mask]
     stat=STAT_KEYS[field];mean=y.new_tensor(c['label_stats'][stat+'_mean']);std=y.new_tensor(c['label_stats'][stat+'_std']);spaces={}
     for space,yy,pp in [('standard',y,pr),('raw',y*std+mean,pr*std+mean)]:
      e=pp-yy;v=dict(n=yy.numel(),points=len(yy),sse=float(e.square().sum()),sae=float(e.abs().sum()),sum_components=yy.sum(0).tolist(),square_components=yy.square().sum(0).tolist(),rel_l1=float(e.abs().sum()/yy.abs().sum().clamp_min(1e-12)),rel_l2=float(e.norm()/yy.norm().clamp_min(1e-12)))
      if not torch.isfinite(e).all():raise FloatingPointError(name+' '+field)
      spaces[space]=v
     fields[field]=spaces
    sets.append(fields)
   row=dict(name=name,sets=sets);atomic_json(path,row);rows.append(row);atomic_json(out/'progress.json',dict(cases=len(rows),total=len(names),sets=10))
   if (i+1)%25==0:print(json.dumps(dict(cases=len(rows),total=len(names))),flush=True)
 summary={}
 for field in FIELDS:
  summary[field]={}
  for space in ['raw','standard']:
   per=[aggregate([r['sets'][st][field][space] for r in rows]) for st in range(10)]
   summary[field][space]=dict(per_set=per,statistics={k:dict(mean=float(np.mean([v[k] for v in per])),min=float(min(v[k] for v in per)),max=float(max(v[k] for v in per)),std=float(np.std([v[k] for v in per],ddof=1))) for k in per[0]})
 atomic_json(out/'summary.json',dict(cases=len(rows),sampling_uncertainty='10 independent fixed query sets; ddof=1',fields=summary,velocity_r2='SSE pooled over xyz, SST centered per component'))
