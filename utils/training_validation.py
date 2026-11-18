"""Full-split, single-query validation with the shared dataset exclusion protocol."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import build_dataset
from utils.checkpoint import capture_rng_state, restore_rng_state, unwrap_model
from utils.config import dataset_config_for_phase
from utils.distributed import is_main_process
from utils.metrics import MetricAccumulator
from utils.field_metrics import denormalize, normalization_spec
from utils.task import build_query, compute_query_loss, move_batch_to_device, prediction_field, target_field


def validation_due(config, step):
    settings = config['train']['validation']
    return bool(settings.get('enabled', True)) and step > 0 and step % int(settings['interval']) == 0


@torch.no_grad()
def validate_at_step(model, config, step, device, *, weights='online', force=False):
    settings = config['train']['validation']
    if not force and not validation_due(config, step):
        return None
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.barrier()
    result = None
    if is_main_process():
        core = unwrap_model(model)
        was_training = core.training
        rng = capture_rng_state()
        core.eval()
        try:
            dataset_config = dataset_config_for_phase(config, 'validation', {
                'split': 'val', 'validation': True, 'deterministic': True,
                'seed': int(config['runtime'].get('seed', 200099)),
                'num_surface_query_points': 10000, 'num_query_points': 10000, 'repeat': 1,
            })
            dataset = build_dataset(dataset_config)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=int(settings.get('batch_size', 1)), shuffle=False,
                num_workers=int(settings.get('workers', 2)), drop_last=False,
                generator=torch.Generator().manual_seed(int(config['runtime'].get('seed', 200099))),
            )
            sums, meters, raw_meters, count = {}, {}, {}, 0
            requested = str(config['runtime'].get('amp_dtype', 'float16')).lower()
            dtype = torch.bfloat16 if requested in ('bfloat16', 'bf16') or requested == 'auto' and torch.cuda.is_bf16_supported() else torch.float16
            for data in loader:
                batch = move_batch_to_device(data, device, config['task']['surface_input_list'])
                query = build_query(batch['surface_query'], batch['volume_query'], config['task'].get('query_type_channel', True))
                forward = getattr(core, 'forward_test', core.forward)
                kwargs = dict(routes=batch['routes'], normals=batch['normals'], area=batch['area'], geometry_meta=batch.get('geometry_meta'))
                if hasattr(core, 'forward_test'):
                    kwargs['query_chunk_size'] = int(settings.get('query_chunk_size', 8192))
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(config['runtime'].get('amp', True)) and device.type == 'cuda'):
                    pred = forward(batch['surface_pos'], query, batch['cond'], **kwargs)
                    loss, values = compute_query_loss(pred, batch, config['task'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite validation loss at step {step}')
                n = len(batch['cond'])
                count += n
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.) + float(value) * n
                for item in config['task'].get('eval_fields', []):
                    domain, field = item['domain'], item['field']
                    pred_field = prediction_field(pred, domain, field, config['task'], batch['surface_query'].shape[1])
                    target_key = 'coefficient' if domain == 'coefficients' else domain
                    target = target_field(batch[target_key + '_target'], config['task'][target_key + '_target_list'], field)
                    meters.setdefault(domain + '/' + field, MetricAccumulator()).update(pred_field, target)
                    if normalization_spec(config['task'], field, domain) is not None:
                        raw_meters.setdefault(domain + '/' + field, MetricAccumulator()).update(
                            denormalize(pred_field, field, config['task'], batch, domain),
                            denormalize(target, field, config['task'], batch, domain))
            if count != len(dataset) or count == 0:
                raise ValueError(f'Validation must cover the full nonempty split: {count}/{len(dataset)} cases')
            result = dict(step=step, cases=count, split='val', loss=sums['total'] / count,
                          fields={key: value / count for key, value in sums.items()},
                          field_metrics={key: meter.result() for key, meter in meters.items()}, weights=weights,
                          raw_field_metrics={key: meter.result() for key, meter in raw_meters.items()},
                          field_metrics_space='model_space', raw_space_transforms=config['task'].get('denormalize', {}),
                          protocol='full_val_query10000_cp_ge_minus10_v1')
            root = Path(config['train']['log_dir'])
            root.mkdir(parents=True, exist_ok=True)
            with (root / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            print('VALIDATION ' + json.dumps(result), flush=True)
        finally:
            core.train(was_training)
            restore_rng_state(rng)
    if distributed:
        objects = [result]
        torch.distributed.broadcast_object_list(objects, src=0)
        result = objects[0]
    return result
