"""State-compatible DSAPT training, launched exclusively through train.py --config."""
from pathlib import Path
import os,sys,json,math,time,random,signal,datetime,fcntl
from utils.checkpoint import atomic_torch_save

def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def worker_init(i):
 import torch
 torch.set_num_threads(1)
def train_dsapt(config):
 import numpy as np,torch,torch.distributed as dist
 from torch.utils.data import DataLoader
 from model.DSAPT.network import make_models,make_optimizer,EMA,training_loss,prepare_query_initial_features
 from dataset.dsapt.data import GeometryDataset,EpochSampler,collate,to_device
 if int(config['dsapt'].get('volume_queries',0))==0:
  from dataset.dsapt.surface_data import GeometryDataset
 c=dict(config['dsapt'])
 c['total_steps']=int(config['train']['total_steps'])
 validation_interval=int(config['train']['validation']['interval'])
 checkpoint_interval=int(config['train']['checkpoint_interval_steps'])
 rank=int(os.getenv('LOCAL_RANK',0));world=int(os.getenv('WORLD_SIZE',1));torch.cuda.set_device(rank)
 if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=int(config['runtime'].get('distributed',{}).get('timeout_seconds',14400))))
 # Partition the allocated physical CPUs between local ranks.
 cores={}
 for cpu in sorted(os.sched_getaffinity(0)):
  p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
  cores.setdefault(((p/'physical_package_id').read_text(),(p/'core_id').read_text()),cpu)
 selected=list(cores.values())[rank::world]
 if selected:os.sched_setaffinity(0,selected)
 torch.set_num_threads(min(4,len(selected) or 4));torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 torch._dynamo.config.cache_size_limit=128;torch._dynamo.config.accumulated_cache_size_limit=1024
 run=Path(config['train']['log_dir']);run.mkdir(parents=True,exist_ok=True)
 checkpoint_dir=Path(config['train']['saveroot']);checkpoint_dir.mkdir(parents=True,exist_ok=True)
 if rank==0:
  lock=(run/'train.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def seed(value):
  random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed(value)
 seed(c['seed']);enc,dec=make_models(c,device=f'cuda:{rank}');enc.train();dec.train()
 if rank==0 and not (run/'initialization.json').exists():atomic_json(run/'initialization.json',dict(initialization=c['initialization'],encoder_checkpoint=c['encoder_checkpoint'],new_step=0))
 from model.DSAPT import apply_optimization,window_block_compile
 changes=apply_optimization.apply(enc,'all');window_block_compile.patch(enc);window_block_compile.sparse_fp16(enc)
 opt=make_optimizer(enc,dec,c);params=[p for g in opt.param_groups for p in g['params']]
 if rank==0:atomic_json(run/'optimizer_groups.json',[dict(name=g['name'],weight_decay=g['weight_decay'],parameter_count=sum(p.numel() for p in g['params'])) for g in opt.param_groups])
 ema=EMA(enc,dec,c['ema_decay']);scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128.)
 forward=torch.compile(dec,fullgraph=True,dynamic=True)
 mf=json.loads(Path(c['manifest_file']).read_text());names=mf['train'];valnames=mf['validation']
 sampler=EpochSampler(len(names),c['seed'],rank,world)
 loader_gen=torch.Generator().manual_seed(c['seed']+rank+101)
 common=dict(batch_size=1,num_workers=2,collate_fn=collate,pin_memory=True,worker_init_fn=worker_init,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
 loader=DataLoader(GeometryDataset(c,names),sampler=sampler,generator=loader_gen,**common)
 vl=DataLoader(GeometryDataset(c,valnames,validation=True),shuffle=False,**common) if rank==0 else None
 epoch_totals=[0.,0.,0.,0.];step=epoch=offset=0;seed(c['seed']+10+rank)
 def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),loader=loader_gen.get_state())
 def restore(r):
  random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu());torch.cuda.set_rng_state(r['cuda'].cpu());loader_gen.set_state(r['loader'].cpu())
 resume=config['train'].get('resume_checkpoint')
 if resume:
  path=checkpoint_dir/'latest.pth' if resume=='latest' else Path(resume).expanduser()
  # Legacy DSAPT snapshots contain numpy RNG tuples and are trusted local files.
  s=torch.load(path,map_location='cpu',weights_only=False)
  if len(s['rng_by_rank'])!=world:raise ValueError('Exact DSAPT resume requires the original world size')
  for key in ('seed','total_steps','peak_lr','end_lr','warmup_steps','betas','ranger','weight_decay','global_scale','encoder_patch_size','grid_size','surface_queries','volume_queries','geometry_points','mode','label_stats','optimizer','encoder_stride','coarse_stride','fine_stride','decoder_dim','decoder_depth','heads','encoder_precision','use_query_normal','decoder_rope','idw_power','idw_eps','idw_residual','lora_rank','lora_alpha','lora_dropout'):
   if key in c and s['config'].get(key)!=c[key]:raise ValueError(f'DSAPT resume changed training state: {key}')
  if str(s['config'].get('loss_type','relative_l2'))!=str(c.get('loss_type','relative_l2')):raise ValueError('DSAPT resume changed its supervised objective')
  saved_fields=s['config'].get('field_loss_config',{field:dict(loss_type=s['config'].get('loss_type','relative_l2'),weight=1.0) for field in c.get('field_loss_config',{})})
  if saved_fields!=c.get('field_loss_config',{}):raise ValueError('DSAPT resume changed field objectives or weights')
  enc.load_state_dict(s['encoder']);dec.load_state_dict(s['decoder']);opt.load_state_dict(s['optimizer']);ema.load_state_dict(s['ema']);scaler.load_state_dict(s['scaler'])
  step,epoch,offset=s['step'],s['epoch'],s['offset'];epoch_totals=s.get('epoch_totals',[0.,0.,0.,0.]);restore(s['rng_by_rank'][rank]);del s
 log=(run/f'rank{rank}.jsonl').open('a',buffering=1);start=time.perf_counter()
 def emit(event,**kw):
  row=dict(time=datetime.datetime.now().astimezone().isoformat(),event=event,step=step,rank=rank,**kw);log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
 def status(state,**kw):
  if rank==0:atomic_json(run/'status.json',dict(status=state,step=step,epoch=epoch,offset=offset,total_steps=c['total_steps'],world_size=world,**kw))
 def barrier():
  if world>1:dist.barrier()
 def save():
  local=rng();states=[None]*world
  if world>1:dist.all_gather_object(states,local)
  else:states=[local]
  if rank==0:
   payload=dict(format='whole_field.dsapt.v1',encoder=enc.state_dict(),decoder=dec.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),step=step,epoch=epoch,offset=offset,config=c,rng_by_rank=states,epoch_totals=epoch_totals,global_scale=c['global_scale'])
   atomic_torch_save(payload,checkpoint_dir/'latest.pth')
   with ema.apply():
    atomic_torch_save(dict(encoder=enc.state_dict(),decoder=dec.state_dict(),config=c,global_scale=c['global_scale'],step=step,weights='EMA'),checkpoint_dir/'latest_ema.pth')
   emit('checkpoint')
  barrier()
 @torch.no_grad()
 def validate():
  from contextlib import nullcontext
  from utils.dsapt_metrics import validation_metrics
  barrier()
  if rank==0:
   state=rng();enc.eval();dec.eval()
   try:
    for weights in ('online','EMA'):
     values=[]
     with ema.apply() if weights=='EMA' else nullcontext():
      for j,b in enumerate(vl):
       torch.manual_seed(c['validation_seed']+j);b=to_device(b,f'cuda:{rank}')
       with torch.autocast('cuda',dtype=torch.float16):
        f,p,n,_=enc(b);z=prepare_query_initial_features(f,p,n,b['query'],c)
        pred=forward(f,p,None,b['query'],z,surface_count=b['target_surface_p'].shape[1])
       values.append(validation_metrics(pred,b,c))
      if not values:raise ValueError('Full validation split is empty')
      stats={k:float(np.mean([v[k] for v in values])) for k in values[0]}
      suffix='_ema' if weights=='EMA' else ''
      result=dict(step=step,epoch=epoch,metrics_standard=stats,cases=len(values),weights=weights,protocol='full_val_query10000_cp_ge_minus10_v1')
      emit('validation',**{k:v for k,v in result.items() if k not in ('step',)})
      atomic_json(run/f'validation_latest{suffix}.json',result)
      bestpath=run/f'best_metrics{suffix}.json';score=stats['joint_rel_l2_standard']
      if not math.isfinite(score):raise FloatingPointError('Non-finite validation metric')
      previous=json.loads(bestpath.read_text()) if bestpath.exists() else {}
      if previous.get('protocol')!=result['protocol'] or score<previous.get('value',float('inf')):
       metadata=dict(step=step,epoch=epoch,metric=c['best_metric'],space='field-standardized',value=score,direction='min',weights=weights,cases=len(values),protocol=result['protocol'])
       atomic_torch_save(dict(encoder=enc.state_dict(),decoder=dec.state_dict(),config=c,global_scale=c['global_scale'],**metadata),checkpoint_dir/f'best{suffix}.pth')
       atomic_json(bestpath,metadata);emit('best_checkpoint',validation_joint_rel_l2=score,weights=weights)
   finally:
    enc.train();dec.train();restore(state)
  barrier()
 stopping=[]
 def stop(sig,frame):stopping.append(sig)
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,stop)
 if rank==0:atomic_json(run/'runtime.json',dict(config=c,encoder_parameters=sum(p.numel() for p in enc.parameters()),encoder_trainable=sum(p.numel() for p in enc.parameters() if p.requires_grad),decoder_parameters=sum(p.numel() for p in dec.parameters()),gradient_sync='FP32 dense gradient all-reduce mean each update; NCCL',gpu=torch.cuda.get_device_name(),world_size=world))
 emit('start',epoch=epoch,offset=offset);status('running');recent=[];forward_samples=[];timing_events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
 try:
  while step<c['total_steps']:
   sampler.epoch=epoch;sampler.skip=offset;it=iter(loader)
   while offset<sampler.length and step<c['total_steps']:
    flag=torch.tensor(bool(stopping),device=rank,dtype=torch.int)
    if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    if flag.item():break
    torch.cuda.synchronize();tick=time.perf_counter();b=to_device(next(it),f'cuda:{rank}');wait=time.perf_counter()-tick
    warm=c['warmup_steps'];lr=c['peak_lr']*(step+1)/(warm+1) if step<warm else c['end_lr']+.5*(c['peak_lr']-c['end_lr'])*(1+math.cos(math.pi*(step-warm)/(c['total_steps']-warm-1)))
    for g in opt.param_groups:g['lr']=lr
    for retry in range(13):
     opt.zero_grad(set_to_none=True)
     measure=50<=step<250
     if measure:timing_events[0].record()
     with torch.autocast('cuda',dtype=torch.float16):
      f,p,n,counts=enc(b)
      if measure:timing_events[1].record()
      z=prepare_query_initial_features(f,p,n,b['query'],c)
      if measure:timing_events[2].record()
      pred=forward(f,p,None,b['query'],z,surface_count=b['target_surface_p'].shape[1])
      if measure:timing_events[3].record()
      loss,parts=training_loss(pred,b,c)
     bad=torch.tensor(not bool(torch.isfinite(loss)),device=rank,dtype=torch.int)
     if world>1:dist.all_reduce(bad,op=dist.ReduceOp.MAX)
     if bad.item():raise FloatingPointError('Nonfinite loss')
     scaler.scale(loss).backward();scaler.unscale_(opt)
     grads=[p.grad for p in params]
     if any(g is None for g in grads):raise RuntimeError('Missing trainable gradient')
     flat=torch._utils._flatten_dense_tensors(grads)
     bad=(~torch.isfinite(flat).all()).to(torch.int)
     if world>1:dist.all_reduce(bad,op=dist.ReduceOp.MAX)
     if bad.item():
      if not scaler.is_enabled():raise FloatingPointError('Nonfinite BF16 gradients')
      scaler.update(new_scale=scaler.get_scale()/2);emit('overflow_retry',retry=retry,scale=scaler.get_scale())
      if retry==12:raise FloatingPointError('Persistent FP16 overflow')
      continue
     if world>1:
      dist.all_reduce(flat);flat.div_(world)
      for g,v in zip(grads,torch._utils._unflatten_dense_tensors(flat,grads)):g.copy_(v)
     del flat,grads
     norm=torch.nn.utils.clip_grad_norm_(params,c['clip_grad_norm'],error_if_nonfinite=True)
     scaler.step(opt);scaler.update();ema.update();break
    vals=torch.stack([loss.detach(),parts['mse'].detach(),parts['mae'].detach(),loss.new_tensor(1.)])
    if world>1:dist.all_reduce(vals)
    epoch_totals=[a+b for a,b in zip(epoch_totals,vals.double().tolist())]
    step+=1;offset+=1;torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    if measure:
     forward_samples.append([timing_events[0].elapsed_time(timing_events[1]),timing_events[1].elapsed_time(timing_events[2]),timing_events[2].elapsed_time(timing_events[3]),timing_events[0].elapsed_time(timing_events[3]),elapsed*1000])
     if step%50==0:
      a=np.asarray(forward_samples);atomic_json(run/f'timing_rank{rank}.json',dict(step=step,samples=len(a),columns=['encoder_ms','idw_ms','decoder_ms','forward_ms','step_ms'],mean=a.mean(0).tolist(),p50=np.quantile(a,.5,axis=0).tolist(),p90=np.quantile(a,.9,axis=0).tolist(),peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
    recent.append(elapsed);recent=recent[-50:]
    if step<=3 or step%50==0:
     if rank==0:
      fields=', '.join(f"{k}: {float(v.detach()):.6f}" for k,v in parts.items() if k in ('surface_p','volume_p','volume_u'))
      line=f"[{epoch}][{offset}/{sampler.length}], lr: {lr:.6f}, time: {elapsed:.2f}, Loss: {loss.item():.6f}, {fields}"
      with (run/'train.log').open('a') as stream:stream.write(line+'\n')
      print(line,flush=True)
     emit('train',field_relative_l2={k:float(v.detach()) for k,v in parts.items() if k.endswith('_rel_l2')},loss=loss.item(),mse=parts['mse'].item(),mae=parts['mae'].item(),lr=lr,step_s=elapsed,data_wait_s=wait,mean_step_s=float(np.mean(recent)),epoch=epoch,offset=offset,stages=enc.stage_counts,grad_scale=scaler.get_scale())
     status('running',mean_step_s=float(np.mean(recent)),loss=loss.item())
    if step%validation_interval==0:validate()
    if step%checkpoint_interval==0:save()
   if stopping or (offset<sampler.length and step<c['total_steps']):break
   if offset==sampler.length:
    emit('epoch_average',epoch=epoch+1,cases=int(epoch_totals[3]),loss=epoch_totals[0]/epoch_totals[3],mse=epoch_totals[1]/epoch_totals[3],mae=epoch_totals[2]/epoch_totals[3]);epoch_totals=[0.,0.,0.,0.]
    epoch+=1;offset=0
  if step==c['total_steps'] and step%validation_interval:validate()
  save()
  if step==c['total_steps']:
   status('complete');emit('complete');
   if rank==0:atomic_json(run/'result.json',dict(complete=True,step=step,elapsed_s=time.perf_counter()-start))
  else:status('stopped');emit('stopped')
 except BaseException as exc:status('failed',error=repr(exc));emit('failed',error=repr(exc));raise
 finally:
  log.close()
  if world>1:dist.destroy_process_group()
