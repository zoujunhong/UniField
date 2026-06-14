from __future__ import annotations

import datetime as _datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def setup_distributed(local_rank: int, config: dict) -> int:
    start_gpu = int(config.get("start_gpu", 0))
    gpu_num = int(config.get("gpu_num", 1))
    device = local_rank + start_gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if gpu_num > 1:
        dist.init_process_group(
            backend=config.get("backend", "nccl"),
            init_method=f"tcp://127.0.0.1:{int(config.get('port', 45325))}",
            world_size=gpu_num,
            rank=local_rank,
            timeout=_datetime.timedelta(seconds=int(config.get("timeout_seconds", 300))),
        )
    return device


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
