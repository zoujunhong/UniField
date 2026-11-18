"""Shared experiment defaults; run differences belong in log/<name>/<variant>."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SPLITS = PROJECT / 'dataset/splits/DrivAerNet'


def make_experiment(dataset, model, size, surface=('p',), volume=(), dataset_option=None):
    codes = {'p': 'P', 'U': 'U', 'normal': 'Normal', 'k': 'K', 'omega': 'Omega', 'nut': 'Nut', 'surface_distance': 'Distance'}
    encode = lambda fields: ''.join(codes[x] for x in fields) or 'X'
    name = f'{dataset}_{model}_{size}_Surface{encode(surface)}_Volume{encode(volume)}'
    option = dataset_option or ('surface_volume' if surface and volume else 'surface' if surface else 'volume')
    model_config = deepcopy(import_module(f'config.model.{model}.{size}').CONFIG)
    tasks = []
    widths = {'p': 1, 'U': 3, 'normal': 3, 'k': 1, 'omega': 1, 'nut': 1, 'surface_distance': 1}
    for domain, fields in [('surface', surface), ('volume', volume)]:
        offset = 0
        for field in fields:
            tasks.append(dict(domain=domain, field=field, pred_slice=(offset, offset+widths[field]), weight=1., loss_type='mse'))
            offset += widths[field]
    channels = max([sum(widths[f] for f in fs) for fs in (surface,volume)])
    if 'decoder_kwargs' in model_config:
        model_config['decoder_kwargs']['out_channels'] = channels
    if model == 'Transolver3':
        model_config['native_model_kwargs']['branch'] = 'both' if surface and volume else 'surface' if surface else 'volume'
    c = dict(experiment=dict(name=name, variant='default', description=''),
        dataset=dataset, dataset_option=option, model=model_config, model_size=size,
        task=dict(surface_input_list=['xyz','normal'], surface_target_list=list(surface), volume_target_list=list(volume),
            query_type_channel=True, losses=tasks, eval_fields=[{k:v for k,v in t.items() if k not in ('loss_type','weight')} for t in tasks]),
        runtime=dict(seed=200099, matmul_precision='high', amp=True, amp_dtype='bfloat16', compile=False,
            distributed=dict(gpu_num=1,start_gpu=0,port=45325,backend='nccl')),
        train=dict(batch_size=1, workers=2, optimizer='AdamW', lr=1e-4, weight_decay=.05, total_steps=200000,
            warmup_iters=10000, final_lr_ratio=.01, clip_grad=1., log_interval=50, checkpoint_interval_steps=20000,
            validation=dict(enabled=True,interval=None,query_points=10000,batch_size=1,workers=2),
            resume_checkpoint='',checkpoint='',ema={'enabled':True,'decay':.9999},auto_test=True),
        test=dict(batch_size=1,workers=2,query_chunk_size=10000,max_batches=0,checkpoint='',log_interval=20),
        visualization=dict(batch_size=1,workers=0,query_chunk_size=10000,max_points=20000,checkpoint='',output_dir=''))
    common = dict(surface_input_list=['xyz','normal'],surface_target_list=list(surface),volume_target_list=list(volume),
                  use_volume=bool(volume),num_surface_query_points=10000 if surface else 0,num_query_points=10000 if volume else 0)
    if dataset == 'DrivAerNet':
        c['task']['denormalize'] = {
            'surface': {'p': {'type': 'affine', 'mean': -93.4098, 'std': 120.787}},
            'volume': {'p': {'type': 'affine', 'mean': -62.4053, 'std': 94.2394},
                       'U': {'type': 'affine', 'mean': [21.8719,-.237778,.673902], 'std': [12.1079,3.97768,3.90113]}}}
        common.update(pressure_mean=-93.4098,pressure_std=120.787,volume_pressure_mean=-62.4053,volume_pressure_std=94.2394,
            velocity_mean=[21.8719,-.237778,.673902],velocity_std=[12.1079,3.97768,3.90113])
        if model == 'ABUPT':
            common.update(use_surface=True,use_volume=True,num_surface_query_points=10000,num_query_points=10000,unsupervised_surface_context=True)
        if model in {'UniField','VolumeUniField','TripNet'}:
            common.update(coordinate_frame='canonical',num_surface_points=8192)
        c['train']['dataset_kwargs'] = dict(ids_file=str(SPLITS/'train_design_ids.txt'))
        c['validation'] = dict(dataset_kwargs=dict(ids_file=str(SPLITS/'val_design_ids.txt')))
        c['test']['dataset_kwargs'] = dict(ids_file=str(SPLITS/'test_design_ids.txt'),deterministic=True)
    else:
        c['train']['dataset_kwargs'] = dict(split='train')
        c['validation'] = dict(dataset_kwargs=dict(split='val'))
        c['test']['dataset_kwargs'] = dict(split='test',deterministic=True)
    if dataset != 'MultiDomain':
        common['route'] = 0
    c['dataset_kwargs'] = common
    if model == 'DSAPT':
        from config.dsapt_defaults import CONFIG as RECIPE
        c['dsapt'] = deepcopy(RECIPE)
        c['dsapt'].update(surface_queries=8192 if surface else 0,volume_queries=8192 if volume else 0,surface_only=not bool(volume))
        c['train'].update(engine='dsapt',checkpoint_interval_steps=1000)
        c['runtime'].update(amp_dtype='float16')
    return c
