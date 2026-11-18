from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel nested ``torch.compile`` and DDP wrappers in either order."""

    # A compiled DDP module and a DDP compiled module expose these wrappers in
    # different orders.  Repeat until reaching the real module so checkpoint
    # keys are stable for both normal exports and recovery snapshots.
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        if hasattr(model, "module"):
            model = model.module
            continue
        break
    return model


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.module.", "")
        key = key.replace("_orig_mod.", "")
        key = key.replace("module.", "")
        cleaned[key] = value
    return cleaned


def _partial_leading_dim_load(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor | None:
    if src.ndim != dst.ndim or src.ndim == 0:
        return None
    if src.shape[0] > dst.shape[0] or tuple(src.shape[1:]) != tuple(dst.shape[1:]):
        return None
    merged = dst.clone()
    merged[: src.shape[0]] = src.to(device=dst.device, dtype=dst.dtype)
    return merged


def _partial_adaln_condition_load(name: str, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor | None:
    """Safely extend [speed, placement] AdaLN inputs with a zero radius column."""

    if not name.endswith(".adaLN_modulation.0.weight"):
        return None
    if src.ndim != 2 or dst.ndim != 2:
        return None
    if src.shape[0] != dst.shape[0] or src.shape[1] > dst.shape[1]:
        return None
    merged = torch.zeros_like(dst)
    merged[:, : src.shape[1]] = src.to(device=dst.device, dtype=dst.dtype)
    return merged


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    route_map: Sequence[int] | None = None,
    strict: bool = False,
    partial_load: bool = False,
    trusted_legacy: bool = False,
) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=not trusted_legacy)
    target = unwrap_model(model)
    if hasattr(target, "load_route_compatible_state_dict"):
        report = target.load_route_compatible_state_dict(
            checkpoint,
            route_map=route_map,
            strict=strict,
            partial_load=partial_load,
        )
        print(
            "Route-compatible load: "
            f"{len(report['loaded'])} exact, "
            f"{len(report.get('legacy_loaded', []))} legacy-remapped, "
            f"{len(report['route_loaded'])} route-adapted, "
            f"{len(report.get('partial_loaded', []))} partial, "
            f"{len(report['skipped'])} skipped",
            flush=True,
        )
        for item in report.get("partial_loaded", []):
            print(f"Partially loaded weight: {item}", flush=True)
        for item in report["skipped"]:
            print(f"Skipping weight: {item}", flush=True)
        return model

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)
    model_state = target.state_dict()
    filtered = {}
    for name, param in state_dict.items():
        if name not in model_state:
            if not strict:
                print(f"Skipping weight: {name} as it is not in the model", flush=True)
                continue
            raise KeyError(f"Checkpoint key not found in model: {name}")
        if model_state[name].shape != param.shape:
            if partial_load:
                merged = _partial_leading_dim_load(model_state[name], param)
                if merged is not None:
                    filtered[name] = merged
                    print(
                        f"Partially loaded weight: {name}: leading slice {tuple(param.shape)} -> "
                        f"{tuple(model_state[name].shape)}",
                        flush=True,
                    )
                    continue
                merged = _partial_adaln_condition_load(name, model_state[name], param)
                if merged is not None:
                    filtered[name] = merged
                    print(
                        f"Partially loaded weight: {name}: AdaLN condition slice {tuple(param.shape)} -> "
                        f"{tuple(model_state[name].shape)}",
                        flush=True,
                    )
                    continue
            if not strict:
                print(f"Skipping weight: {name} due to shape mismatch ({param.shape} vs {model_state[name].shape})", flush=True)
                continue
            raise RuntimeError(f"Shape mismatch for {name}: {param.shape} vs {model_state[name].shape}")
        filtered[name] = param
    if strict:
        target.load_state_dict(filtered, strict=True)
    else:
        model_state.update(filtered)
        target.load_state_dict(model_state, strict=False)
    return model


def resolve_eval_checkpoint(config: dict, phase: str) -> str:
    phase_config = config.get(phase, {})
    if "checkpoint" in phase_config and phase_config["checkpoint"] is None:
        return ""
    checkpoint = str(phase_config.get("checkpoint", ""))
    if checkpoint:
        return str(Path(checkpoint).expanduser())
    train_config = config.get("train", {})
    saveroot = train_config.get("saveroot")
    if not saveroot:
        return ""
    preferred = Path(saveroot) / "best.pth"
    return str(preferred if preferred.exists() else Path(saveroot) / "latest.pth")


def save_checkpoint(model: torch.nn.Module, save_dir: str | Path, epoch: int, filename: str | None = None) -> Path:
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or f"model_epoch_{int(epoch)}.pth"
    path = save_dir / filename
    # A recovery loop must never select a half-written checkpoint after a
    # scheduler/preemption event.  Replace atomically only once serialization
    # completed successfully on this filesystem.
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(unwrap_model(model).state_dict(), temporary)
    os.replace(temporary, path)
    return path


TRAINING_FORMAT = 'whole_field.training.v2'


def capture_rng_state() -> dict:
    import random
    import numpy as np

    state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    import random
    import numpy as np

    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_torch_save(payload, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def training_resume_contract(config):
    """Inputs that must remain fixed for an exact continuation."""
    train, runtime = config['train'], config['runtime']
    return {
        'dataset': config.get('dataset'), 'model': config.get('model'), 'task': config.get('task'),
        'dataset_kwargs': config.get('dataset_kwargs'), 'train_dataset_kwargs': train.get('dataset_kwargs'),
        'seed': runtime.get('seed', 200099), 'amp': runtime.get('amp', True),
        'amp_dtype': runtime.get('amp_dtype', 'float16'),
        'optimizer': {key: train.get(key) for key in ('optimizer', 'optimizer_kwargs', 'weight_decay', 'eps', 'clip_grad', 'special_keyword', 'special_lr')},
        'batch_size': train.get('batch_size', 1),
    }


def save_training_checkpoint(model, optimizer, scaler, config, progress, *, filename='latest.pth', ema=None):
    """Collect each rank's RNG and commit one atomic full-state checkpoint."""
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    rng_states = [None] * world
    if distributed:
        torch.distributed.all_gather_object(rng_states, capture_rng_state())
    else:
        rng_states[0] = capture_rng_state()
    if rank == 0:
        payload = {
            'format': TRAINING_FORMAT, 'state_dict': unwrap_model(model).state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scaler_state': scaler.state_dict(),
            'progress': dict(progress), 'rng_by_rank': rng_states, 'world_size': world,
            'schedule': {key: config['train'].get(key) for key in ('total_steps', 'warmup_iters', 'final_lr_ratio', 'lr')},
            'experiment': dict(config['experiment']), 'resume_contract': training_resume_contract(config),
            'ema_state': ema.state_dict() if ema is not None else None,
        }
        atomic_torch_save(payload, Path(config['train']['saveroot']) / filename)
    if distributed:
        torch.distributed.barrier()


def load_training_checkpoint(model, optimizer, scaler, config, path, *, ema=None) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != TRAINING_FORMAT:
        raise ValueError(f'{path} lacks complete resumable state; use train.checkpoint for a documented weights-only initialization.')
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    if payload['world_size'] != world:
        raise ValueError('Exact resume requires the checkpoint world_size; start an explicitly named warm-start variant to change it.')
    expected = {key: config['train'].get(key) for key in ('total_steps', 'warmup_iters', 'final_lr_ratio', 'lr')}
    if payload['schedule'] != expected:
        raise ValueError('Resume cannot change the saved learning-rate schedule; record a separate warm-start variant.')
    if payload.get('resume_contract') != training_resume_contract(config):
        raise ValueError('Resume changed model/task/dataset/sampling/optimizer inputs; use a separately named initialization variant.')
    unwrap_model(model).load_state_dict(clean_state_dict(payload['state_dict']), strict=True)
    optimizer.load_state_dict(payload['optimizer_state'])
    scaler.load_state_dict(payload['scaler_state'])
    if ema is not None:
        if payload.get('ema_state') is None:
            raise ValueError('EMA was enabled but the resume checkpoint contains no EMA state.')
        ema.load_state_dict(payload['ema_state'])
    elif payload.get('ema_state') is not None:
        raise ValueError('The resume checkpoint used EMA; keep train.ema enabled when resuming.')
    restore_rng_state(payload['rng_by_rank'][rank])
    return dict(payload['progress'])


class ModelEMA:
    """Keep a CPU shadow and temporarily apply it without a second GPU model."""
    def __init__(self, model, decay: float):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must lie in [0, 1).')
        self.model = model
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {name: value.detach().cpu().clone() for name, value in unwrap_model(model).state_dict().items()}

    @torch.no_grad()
    def update(self):
        for name, value in unwrap_model(self.model).state_dict().items():
            value = value.detach().to('cpu')
            if value.is_floating_point():
                self.shadow[name].lerp_(value, 1 - self.decay)
            else:
                self.shadow[name].copy_(value)
        self.updates += 1

    def state_dict(self):
        return {'decay': self.decay, 'updates': self.updates, 'shadow': self.shadow}

    def load_state_dict(self, state):
        if float(state['decay']) != self.decay:
            raise ValueError('EMA decay differs from the resume checkpoint.')
        self.updates = int(state['updates'])
        self.shadow = state['shadow']

    def apply(self):
        from contextlib import contextmanager

        @contextmanager
        def swap():
            core = unwrap_model(self.model)
            original = {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}
            core.load_state_dict(self.shadow, strict=True)
            try:
                yield
            finally:
                core.load_state_dict(original, strict=True)
        return swap()
