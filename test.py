from __future__ import annotations

import argparse

import torch

from dataset import build_dataset
from model import build_model
from train import load_main_config
from utils.checkpoint import load_model_weights
from utils.config import dataset_config_for_phase
from utils.metrics import MetricAccumulator, format_metrics
from utils.task import build_query, move_batch_to_device, prediction_field, target_field


def _high_speed_from_cond(cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ux = cond[:, 0] * 100.0
    uy = cond[:, 1] * 20.0
    free_stream = torch.stack([ux, uy, torch.zeros_like(ux)], dim=-1)
    speed = torch.linalg.norm(free_stream, dim=-1, keepdim=True).clamp_min(1e-12)
    return free_stream, speed


def denormalize(
    values: torch.Tensor,
    field: str,
    task_config: dict,
    batch: dict[str, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    spec = task_config.get("denormalize", {}).get(field)
    if not spec:
        return values
    if spec.get("type") == "affine":
        mean = values.new_tensor(spec["mean"]).view(*([1] * (values.ndim - 1)), -1)
        std = values.new_tensor(spec["std"]).view(*([1] * (values.ndim - 1)), -1)
        return values * std + mean

    if spec.get("type") == "physical":
        dataset = spec.get("dataset", "DrivAerNet")
        if dataset == "DrivAerNet":
            u_ref = float(spec.get("u_ref", 30.0))
            if field == "p":
                return values * (0.5 * u_ref * u_ref)
            if field == "U":
                velocity_ref = values.new_tensor(spec.get("velocity_ref", [u_ref, 0.0, 0.0])).view(1, 1, 3)
                return values * abs(u_ref) + velocity_ref
        if dataset in {"HighSpeedTrain", "CRH450", "Maglev"}:
            if batch is None:
                raise ValueError("HighSpeedTrain physical denormalization needs batch['cond'].")
            free_stream, speed = _high_speed_from_cond(batch["cond"])
            speed = speed.view(-1, 1, 1)
            if field == "p":
                return values * (0.5 * speed * speed)
            if field == "U":
                return values * speed + free_stream.view(-1, 1, 3)
            if field == "k":
                return values * speed * speed
            if field in {"omega", "nut"}:
                return values
    raise ValueError(f"Unsupported denormalize spec for {field}: {spec}")


@torch.no_grad()
def evaluate(config: dict) -> None:
    runtime_config = config["runtime"]
    test_config = config["test"]
    task_config = config["task"]
    torch.set_float32_matmul_precision(runtime_config.get("matmul_precision", "high"))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    ds_config = dataset_config_for_phase(
        config,
        "test",
        {
            "deterministic": True,
            "seed": int(runtime_config.get("seed", 200099)),
        },
    )
    dataset = build_dataset(ds_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(test_config.get("batch_size", 1)),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=int(test_config.get("workers", 4)),
        drop_last=False,
    )

    model = build_model(config["model"]).to(device).float().eval()
    if test_config.get("checkpoint"):
        load_model_weights(model, test_config["checkpoint"], strict=True)
        print("checkpoint loaded", flush=True)
    else:
        print("no checkpoint configured; running random-init smoke", flush=True)

    meters: dict[str, MetricAccumulator] = {}
    raw_meters: dict[str, MetricAccumulator] = {}
    max_batches = int(test_config.get("max_batches", 0))
    for idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device, task_config["surface_input_list"])
        query = build_query(
            batch["surface_query"],
            batch["volume_query"],
            add_type_channel=bool(task_config.get("query_type_channel", True)),
        )
        if bool(test_config.get("use_forward_test", True)):
            try:
                pred = model.forward_test(
                    batch["surface_pos"],
                    query,
                    batch["cond"],
                    routes=batch["routes"],
                    normals=batch["normals"],
                    area=batch["area"],
                    query_chunk_size=int(test_config.get("query_chunk_size", 50000)),
                )
            except NotImplementedError:
                pred = model(
                    batch["surface_pos"],
                    query,
                    batch["cond"],
                    routes=batch["routes"],
                    normals=batch["normals"],
                    area=batch["area"],
                )
        else:
            pred = model(
                batch["surface_pos"],
                query,
                batch["cond"],
                routes=batch["routes"],
                normals=batch["normals"],
                area=batch["area"],
            )

        surface_count = int(batch["surface_query"].shape[1])
        for item in task_config.get("eval_fields", task_config.get("losses", [])):
            domain = item["domain"]
            field = item["field"]
            key = f"{domain}_{field}"
            pred_field = prediction_field(pred, domain, field, {"losses": [item]}, surface_count)
            if domain == "surface":
                tgt = target_field(batch["surface_target"], task_config["surface_target_list"], field)
            elif domain == "volume":
                tgt = target_field(batch["volume_target"], task_config["volume_target_list"], field)
            else:
                raise ValueError(f"Unsupported domain: {domain}")
            meters.setdefault(key, MetricAccumulator()).update(pred_field, tgt)
            raw_meters.setdefault(key, MetricAccumulator()).update(
                denormalize(pred_field, field, task_config, batch),
                denormalize(tgt, field, task_config, batch),
            )

        if idx % int(test_config.get("log_interval", 20)) == 0:
            text = " | ".join(f"{key}: {format_metrics(meter.result())}" for key, meter in meters.items())
            print(f"[{idx}/{len(loader)}] {text}", flush=True)
        if max_batches > 0 and idx + 1 >= max_batches:
            break

    print("\nFinal model-space metrics", flush=True)
    for key, meter in meters.items():
        print(f"{key}: {format_metrics(meter.result())}", flush=True)
    print("\nFinal raw-space metrics", flush=True)
    for key, meter in raw_meters.items():
        print(f"{key}: {format_metrics(meter.result())}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven UniField evaluation.")
    parser.add_argument("--config", required=True, help="Path to a Python config file exporting CONFIG.")
    args = parser.parse_args()
    evaluate(load_main_config(args.config))


if __name__ == "__main__":
    main()
