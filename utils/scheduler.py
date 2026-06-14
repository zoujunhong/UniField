from __future__ import annotations

import numpy as np


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def exp_scheduler(
    base_value: float,
    decay_rate: float,
    decay_steps: int,
    epochs: int,
    niter_per_ep: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value * (decay_rate ** (warmup_iters / decay_steps)), warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = base_value * (decay_rate ** (iters / decay_steps))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
