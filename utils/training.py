"""Shared optimizer-step, validation/checkpoint and process lifecycle helpers."""
from __future__ import annotations

import json
import signal
from pathlib import Path

import torch

from utils.checkpoint import atomic_torch_save, save_training_checkpoint
from utils.distributed import is_main_process
from utils.finite import ensure_finite, named_parameter_tensors, optimizer_state_tensors
from utils.task import build_query, compute_query_loss, move_batch_to_device
from utils.training_validation import validate_at_step, validation_due


def resolve_amp_dtype(runtime):
    requested = str(runtime.get('amp_dtype', 'float16')).lower()
    aliases = {'float16': torch.float16, 'fp16': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16}
    if requested == 'auto':
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if requested not in aliases:
        raise ValueError('runtime.amp_dtype must be float16, bfloat16 or auto')
    return aliases[requested]


class StopRequest:
    def __init__(self):
        self.requested = False
        for kind in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(kind, self._request)

    def _request(self, signum, frame):
        self.requested = True

    def synchronize(self, device):
        flag = torch.tensor(int(self.requested), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())


def optimize_step(model, data, optimizer, scaler, config, device, progress):
    task, train, runtime = config['task'], config['train'], config['runtime']
    batch = move_batch_to_device(data, device, task['surface_input_list'])
    query = build_query(batch['surface_query'], batch['volume_query'], task.get('query_type_channel', True))
    context = dict(progress)
    context['config'] = config['_config_path']
    for retry in range(13):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=resolve_amp_dtype(runtime), enabled=bool(runtime.get('amp', True)) and device.type == 'cuda'):
            pred = model(batch['surface_pos'], query, batch['cond'], routes=batch['routes'],
                         normals=batch['normals'], area=batch['area'], geometry_meta=batch.get('geometry_meta'))
            loss, losses = compute_query_loss(pred, batch, task)
        ensure_finite({'loss': loss, 'fields': losses}, stage='loss', device=device,
                      saveroot=train['saveroot'], context=context, batch=batch,
                      save_batch=bool(train.get('finite_check', {}).get('save_batch', True)) if isinstance(train.get('finite_check'), dict) else True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
        finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
        bad = torch.tensor(int(not finite), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            if scaler.is_enabled() and retry < 12:
                scaler.update(new_scale=scaler.get_scale() / 2)
                continue
            ensure_finite(named_parameter_tensors(model, gradients=True), stage='gradient', device=device,
                          saveroot=train['saveroot'], context=context, batch=batch)
        torch.nn.utils.clip_grad_norm_(parameters, float(train.get('clip_grad', 1.0)), error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        keys = list(losses)
        values = torch.stack([losses[key].detach().float() for key in keys])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
            values /= torch.distributed.get_world_size()
        return {key: float(value) for key, value in zip(keys, values)}
    raise RuntimeError('Persistent mixed-precision gradient overflow')


def validate_and_save(model, optimizer, scaler, ema, config, progress, device, *, force_validation=False, save_latest=False):
    train = config['train']
    step = progress['global_step']
    due = (bool(train['validation'].get('enabled', True))
           and step != progress.get('last_validation_step')
           and (force_validation or validation_due(config, step)))
    if due or save_latest:
        ensure_finite({'model': named_parameter_tensors(model), 'optimizer': optimizer_state_tensors(optimizer)},
                      stage='before_checkpoint', device=device, saveroot=train['saveroot'], context=progress, save_batch=False)
    if due:
        result = validate_at_step(model, config, step, device, force=True)
        online_improved = result['loss'] < progress.get('best_loss', float('inf'))
        if online_improved:
            progress['best_loss'] = result['loss']
            if is_main_process():
                (Path(train['log_dir']) / 'best_validation.json').write_text(json.dumps(result, indent=2) + '\n')
        if ema is not None:
            with ema.apply():
                result = validate_at_step(model, config, step, device, weights='EMA', force=True)
                if result['loss'] < progress.get('best_ema_loss', float('inf')):
                    progress['best_ema_loss'] = result['loss']
                    if is_main_process():
                        atomic_torch_save({'state_dict': ema.shadow, 'global_step': step, 'weights': 'EMA'}, Path(train['saveroot']) / 'best_ema.pth')
                        (Path(train['log_dir']) / 'best_validation_ema.json').write_text(json.dumps(result, indent=2) + '\n')
        progress['last_validation_step'] = step
        if online_improved:
            save_training_checkpoint(model, optimizer, scaler, config, progress, filename='best.pth', ema=ema)
    if save_latest:
        save_training_checkpoint(model, optimizer, scaler, config, progress, ema=ema)
        if ema is not None and is_main_process():
            atomic_torch_save({'state_dict': ema.shadow, 'global_step': step, 'weights': 'EMA'}, Path(train['saveroot']) / 'latest_ema.pth')
    if (due or save_latest) and torch.distributed.is_initialized():
        torch.distributed.barrier()
