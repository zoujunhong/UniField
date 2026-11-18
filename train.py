"""The config-only entry point for data-driven field model training."""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset import build_dataset
from model import build_model
from utils.checkpoint import ModelEMA, load_model_weights, load_training_checkpoint
from utils.config import (apply_runtime_seed, configure_training_run, dataset_config_for_phase,
                          load_main_config)
from utils.dataloader import build_epoch_loader
from utils.distributed import cleanup_distributed, is_main_process, set_seed, setup_distributed
from utils.optim import build_optimizer
from utils.scheduler import cosine_scheduler
from utils.training import StopRequest, optimize_step, resolve_amp_dtype, validate_and_save
from utils.training_pipeline import run_training_pipeline, write_status


def train_worker(rank: int, config: dict) -> None:
    runtime, train = config['runtime'], config['train']
    torch.set_num_threads(int(runtime.get('cpu_threads', 4)))
    set_seed(int(runtime.get('seed', 200099)) + rank)
    torch.set_float32_matmul_precision(runtime.get('matmul_precision', 'high'))
    device_index = setup_distributed(rank, runtime.get('distributed', {}))
    device = torch.device('cuda', device_index) if torch.cuda.is_available() else torch.device('cpu')
    try:
        _train(config, device)
    finally:
        cleanup_distributed()


def _train(config: dict, device: torch.device) -> None:
    runtime, train = config['runtime'], config['train']
    if config.get('coordinate_statistics'):
        from dataset.cache.drivaernet import load_or_compute_coordinate_statistics
        if is_main_process():
            load_or_compute_coordinate_statistics(**config['coordinate_statistics'])
        if dist.is_initialized():
            dist.barrier()
    dataset = build_dataset(dataset_config_for_phase(config, 'train'))
    model = build_model(config['model']).to(device).float()
    resume = train.get('resume_checkpoint')
    if train.get('checkpoint') and not resume:
        load_model_weights(model, train['checkpoint'], route_map=train.get('route_map'), partial_load=bool(train.get('partial_load', False)), trusted_legacy=bool(train.get('trust_legacy_checkpoint', False)))
    if bool(runtime.get('compile', False)):
        model.compile()
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
    optimizer_kwargs = dict(train.get('optimizer_kwargs', {}))
    if train.get('optimizer', 'AdamW') in ('Adam', 'AdamW'):
        optimizer_kwargs.setdefault('eps', float(train.get('eps', 1e-7)))
    optimizer = build_optimizer(model, dict(name=train.get('optimizer', 'AdamW'), kwargs=optimizer_kwargs,
        lr=train.get('lr', 1e-4), weight_decay=train.get('weight_decay', .04),
        special_keyword=train.get('special_keyword', []), special_lr=train.get('special_lr', [])))
    scaler = torch.amp.GradScaler(enabled=bool(runtime.get('amp', True)) and device.type == 'cuda' and resolve_amp_dtype(runtime) == torch.float16)
    ema_settings = train.get('ema', {})
    if ema_settings is True:
        ema_settings = {'enabled': True}
    ema = ModelEMA(model, float(ema_settings.get('decay', .999))) if ema_settings and ema_settings.get('enabled', True) else None
    total_steps = int(train['total_steps'])
    schedule = cosine_scheduler(1., float(train.get('final_lr_ratio', .01)), 1, total_steps,
        warmup_iters=min(int(train.get('warmup_iters', 5000)), total_steps - 1))
    progress = dict(global_step=0, epoch=0, batch_offset=0, dataset_length=len(dataset))
    if resume:
        path = Path(train['saveroot']) / 'latest.pth' if resume == 'latest' else Path(resume).expanduser()
        progress = load_training_checkpoint(model, optimizer, scaler, config, path, ema=ema)
        if progress['dataset_length'] != len(dataset):
            raise ValueError('Dataset length changed since the resumable checkpoint.')
    if not 0 <= progress['global_step'] <= total_steps:
        raise ValueError('Checkpoint step lies outside the configured total_steps.')
    stop = StopRequest()
    if is_main_process():
        write_status(Path(train['log_dir']) / 'status.json', dict(status='running', step=progress['global_step'], total_steps=total_steps))
    if is_main_process():
        print(f"Training: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M parameters, step {progress['global_step']}/{total_steps}", flush=True)
    stopped = False
    while progress['global_step'] < total_steps and not stopped:
        epoch = progress['epoch']
        loader = build_epoch_loader(dataset, config, epoch=epoch, start_batch=progress['batch_offset'])
        epoch_batches = loader.batch_sampler.full_length
        if epoch_batches <= 0:
            raise ValueError('Training split is too small for the batch size and world size.')
        if 'steps_per_epoch' in progress and progress['steps_per_epoch'] != epoch_batches:
            raise ValueError('Batch size or sampler length changed since the resume checkpoint.')
        progress['steps_per_epoch'] = epoch_batches
        for data in loader:
            if stop.synchronize(device):
                stopped = True
                break
            tick = time.perf_counter()
            step = progress['global_step']
            for group in optimizer.param_groups:
                group['lr'] = float(schedule[step]) * group['base_lr']
            model.train()
            losses = optimize_step(model, data, optimizer, scaler, config, device, progress)
            if ema is not None:
                ema.update()
            progress['global_step'] += 1
            progress['batch_offset'] += 1
            offset = progress['batch_offset']
            if offset == epoch_batches:
                progress.update(epoch=epoch + 1, batch_offset=0)
            step = progress['global_step']
            if is_main_process() and (step == 1 or step % int(train.get('log_interval', 10)) == 0):
                fields = ', '.join(f'{key}: {value:.6f}' for key, value in losses.items() if key != 'total')
                line = f"[{epoch}][{offset}/{epoch_batches}], lr: {optimizer.param_groups[0]['lr']:.6f}, time: {time.perf_counter()-tick:.2f}, Loss: {losses['total']:.6f}"
                if fields:
                    line += ', ' + fields
                print(line, flush=True)
                with Path(train['log_path']).open('a') as stream:
                    stream.write(line + '\n')
            validate_and_save(model, optimizer, scaler, ema, config, progress, device,
                              save_latest=step % int(train['checkpoint_interval_steps']) == 0)
            if step >= total_steps or int(train.get('stop_after_step', 0)) and step >= int(train['stop_after_step']):
                stopped = step < total_steps
                break
    validate_and_save(model, optimizer, scaler, ema, config, progress, device, save_latest=True,
                      force_validation=progress['global_step'] == total_steps and total_steps % int(train['validation']['interval']) != 0)
    if is_main_process():
        write_status(Path(train['log_dir']) / 'status.json', dict(status='stopped' if stopped else 'complete', step=progress['global_step'], total_steps=total_steps))
        print(f"Training {'paused' if stopped else 'complete'} at optimizer step {progress['global_step']}/{total_steps}.", flush=True)


def dsapt_worker(rank: int, config: dict) -> None:
    distributed = config['runtime'].get('distributed', {})
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE=str(distributed.get('gpu_num', 1)),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(distributed.get('port', 45325)))
    from utils.training_dsapt import train_dsapt
    train_dsapt(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Executable experiment or log variant config.py')
    args = parser.parse_args()
    config = configure_training_run(load_main_config(args.config))
    apply_runtime_seed(config)
    saved = [Path(config['train']['saveroot']) / name for name in ('latest.pth', 'best.pth', 'latest_ema.pth', 'best_ema.pth')]
    existing = next((path for path in saved if path.exists()), None)
    if existing is not None and not config['train'].get('resume_checkpoint'):
        raise FileExistsError(f'{existing} already exists; use an explicit complete-state resume or a separate initialization variant.')
    run_training_pipeline(config, _launch_workers)


def _launch_workers(config):
    worker = dsapt_worker if config['train'].get('engine') == 'dsapt' else train_worker
    world = int(config['runtime'].get('distributed', {}).get('gpu_num', 1))
    if worker is dsapt_worker:
        start = int(config['runtime'].get('distributed', {}).get('start_gpu', 0))
        if start:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(start, start + world))
    if world > 1:
        context = mp.spawn(worker, nprocs=world, args=(config,), join=False)
        previous = {}
        def request_stop(signum, frame):
            for process in context.processes:
                if process.is_alive():
                    os.kill(process.pid, signal.SIGUSR1)
        try:
            for kind in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                previous[kind] = signal.signal(kind, request_stop)
            while not context.join():
                pass
        finally:
            for kind, handler in previous.items():
                signal.signal(kind, handler)
    else:
        worker(0, config)


if __name__ == '__main__':
    main()
