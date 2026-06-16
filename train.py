from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset import build_dataset
from model import build_model
from utils.checkpoint import load_model_weights, save_checkpoint
from utils.config import dataset_config_for_phase, load_config, resolve_path
from utils.dataloader import MultiEpochsDataLoader
from utils.distributed import cleanup_distributed, is_main_process, set_seed, setup_distributed
from utils.metrics import AverageMeter
from utils.optim import build_optimizer
from utils.scheduler import cosine_scheduler
from utils.task import build_query, compute_query_loss, move_batch_to_device


def train_one_epoch(model, loader, optimizer, scaler, lr_schedule, epoch: int, device, config: dict) -> None:
    task_config = config["task"]
    train_config = config["train"]
    runtime_config = config["runtime"]
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    field_meters: dict[str, AverageMeter] = {}
    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    tic = time.time()
    max_iters = int(train_config.get("dry_run_iters", 0))
    for idx, batch in enumerate(loader):
        it = len(loader) * epoch + idx
        for group in optimizer.param_groups:
            group["lr"] = lr_schedule[it] * group["base_lr"]

        batch = move_batch_to_device(batch, device, task_config["surface_input_list"])
        query = build_query(
            batch["surface_query"],
            batch["volume_query"],
            add_type_channel=bool(task_config.get("query_type_channel", True)),
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(runtime_config.get("amp", True)) and torch.cuda.is_available(),
        ):
            pred = model(
                batch["surface_pos"],
                query,
                batch["cond"],
                routes=batch["routes"],
                normals=batch["normals"],
                area=batch["area"],
            )
            loss, losses = compute_query_loss(pred, batch, task_config)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("clip_grad", 0.1)))
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - tic)
        tic = time.time()
        loss_meter.update(loss.detach())
        for key, value in losses.items():
            if key == "total":
                continue
            field_meters.setdefault(key, AverageMeter()).update(value.detach())

        if is_main_process() and idx % int(train_config.get("log_interval", 10)) == 0:
            field_text = ", ".join(f"{key}: {meter.average():.6f}" for key, meter in field_meters.items())
            text = (
                f"[{epoch}][{idx}/{len(loader)}], lr: {lr_schedule[it]:.6f}, "
                f"time: {batch_time.average():.2f}, Loss: {loss_meter.average():.6f}"
            )
            if field_text:
                text = f"{text}, {field_text}"
            print(text, flush=True)
            os.makedirs("log", exist_ok=True)
            with open(f"log/{os.path.basename(train_config['saveroot'])}.txt", "a", encoding="utf-8") as f:
                print(text, file=f)

        if max_iters > 0 and idx + 1 >= max_iters:
            break


def main_worker(local_rank: int, config: dict) -> None:
    runtime_config = config["runtime"]
    train_config = config["train"]
    set_seed(int(runtime_config.get("seed", 200099)) + local_rank)
    torch.set_float32_matmul_precision(runtime_config.get("matmul_precision", "high"))

    device_index = setup_distributed(local_rank, runtime_config.get("distributed", {}))
    device = torch.device("cuda", device_index) if torch.cuda.is_available() else torch.device("cpu")

    if is_main_process():
        print("===== config loaded =====", flush=True)
        print("===== dataset loading ... =====", flush=True)
    dataset = build_dataset(dataset_config_for_phase(config, "train"))
    sampler = None
    shuffle = True
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=int(runtime_config.get("seed", 200099)))
        shuffle = False
    loader = MultiEpochsDataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 4)),
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=int(train_config.get("workers", 4)),
        drop_last=True,
    )
    if is_main_process():
        print("===== dataset loaded =====", flush=True)
        print("===== model loading ... =====", flush=True)

    model = build_model(config["model"])
    if is_main_process():
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量：{param_count / 1e6:.2f}M", flush=True)

    if train_config.get("checkpoint"):
        load_model_weights(model, train_config["checkpoint"], route_map=train_config.get("route_map"))
        if is_main_process():
            print("===== checkpoint loaded =====", flush=True)
    if int(train_config.get("resume_epoch", 0)) > 0:
        resume_path = os.path.join(train_config["saveroot"], f"model_epoch_{int(train_config['resume_epoch'])}.pth")
        load_model_weights(model, resume_path, strict=True)
        if is_main_process():
            print(f"===== resume from epoch {train_config['resume_epoch']} =====", flush=True)

    model = model.to(device).float()
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_index])
    if bool(runtime_config.get("compile", False)):
        model = torch.compile(model)

    optimizer_config = {
        "name": train_config.get("optimizer", "AdamW"),
        "lr": train_config.get("lr", 1e-4),
        "weight_decay": train_config.get("weight_decay", 0.04),
        "kwargs": {"eps": train_config.get("eps", 1e-7)},
        "special_keyword": train_config.get("special_keyword", ["adapt"]),
        "special_lr": train_config.get("special_lr", [train_config.get("lr", 1e-4)]),
    }
    optimizer = build_optimizer(model, optimizer_config)
    total_epoch = int(train_config.get("total_epoch", 200))
    lr_schedule = cosine_scheduler(
        1.0,
        float(train_config.get("final_lr_ratio", 0.01)),
        total_epoch,
        len(loader),
        warmup_iters=min(int(train_config.get("warmup_iters", 5000)), max(total_epoch * len(loader) - 1, 0)),
    )
    scaler = torch.amp.GradScaler(enabled=bool(runtime_config.get("amp", True)) and torch.cuda.is_available())

    start_epoch = int(train_config.get("resume_epoch", 0))
    for epoch in range(start_epoch, total_epoch):
        train_one_epoch(model, loader, optimizer, scaler, lr_schedule, epoch, device, config)
        if is_main_process() and (epoch + 1) % int(train_config.get("save_step", 50)) == 0:
            path = save_checkpoint(model, train_config["saveroot"], epoch + 1)
            print(f"saved checkpoint: {path}", flush=True)
        if int(train_config.get("dry_run_iters", 0)) > 0:
            break

    if is_main_process():
        print("Training Done!", flush=True)
    cleanup_distributed()


def _resolve_dataset_option_path(config: dict) -> str:
    dataset_name = config.get("dataset")
    dataset_option = config.get("dataset_option")
    if not dataset_name or not dataset_option:
        raise KeyError("Experiment config must define dataset and dataset_option.")
    return str(resolve_path(f"config/dataset/{dataset_name}/{dataset_option}.py", config.get("_config_dir")))


def _resolve_model_size_path(config: dict) -> str:
    model_name = config.get("model")
    model_size = config.get("model_size")
    if not model_name or not model_size:
        raise KeyError("Experiment config must define model and model_size.")
    return str(resolve_path(f"config/model/{model_name}/{model_size}.py", config.get("_config_dir")))


def load_main_config(path: str) -> dict:
    config = load_config(path)
    config["_dataset_name"] = config.get("dataset")
    config["_dataset_option"] = config.get("dataset_option")
    config["_model_name"] = config.get("model")
    config["_model_size"] = config.get("model_size")
    config["dataset"] = load_config(_resolve_dataset_option_path(config))
    config["model"] = load_config(_resolve_model_size_path(config))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven UniField training.")
    parser.add_argument("--config", required=True, help="Path to a Python config file exporting CONFIG.")
    args = parser.parse_args()
    config = load_main_config(args.config)
    gpu_num = int(config.get("runtime", {}).get("distributed", {}).get("gpu_num", 1))
    if gpu_num > 1:
        mp.spawn(main_worker, nprocs=gpu_num, args=(config,))
    else:
        main_worker(0, config)


if __name__ == "__main__":
    main()
