from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset import build_dataset
from model import build_model
from utils.config import load_main_config, configure_training_run
from utils.checkpoint import load_model_weights, resolve_eval_checkpoint
from utils.config import apply_runtime_seed, dataset_config_for_phase
from utils.finite import ensure_finite
from utils.metrics import MetricAccumulator, format_metrics
from utils.field_metrics import denormalize, normalization_spec
from utils.task import build_query, move_batch_to_device, prediction_field, target_field


def _update_metrics(
    meters: dict[str, MetricAccumulator],
    raw_meters: dict[str, MetricAccumulator],
    key: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    field: str,
    task_config: dict,
    batch: dict[str, torch.Tensor | None],
    domain: str,
) -> None:
    meters.setdefault(key, MetricAccumulator()).update(pred, target)
    has_raw = normalization_spec(task_config, field, domain) is not None
    raw_pred = denormalize(pred, field, task_config, batch, domain)
    raw_target = denormalize(target, field, task_config, batch, domain)
    if has_raw:
        raw_meters.setdefault(key, MetricAccumulator()).update(raw_pred, raw_target)

    if field != "U" or pred.shape[-1] != 3:
        return
    component_names = ("Ux", "Uy", "Uz")
    for component, component_name in enumerate(component_names):
        component_key = f"{key}_{component_name}"
        meters.setdefault(component_key, MetricAccumulator()).update(
            pred[..., component : component + 1], target[..., component : component + 1]
        )
        if has_raw:
            raw_meters.setdefault(component_key, MetricAccumulator()).update(
                raw_pred[..., component : component + 1], raw_target[..., component : component + 1]
            )
    speed_key = f"{key}_speed"
    meters.setdefault(speed_key, MetricAccumulator()).update(
        torch.linalg.vector_norm(pred, dim=-1, keepdim=True),
        torch.linalg.vector_norm(target, dim=-1, keepdim=True),
    )
    if has_raw:
        raw_meters.setdefault(speed_key, MetricAccumulator()).update(
            torch.linalg.vector_norm(raw_pred, dim=-1, keepdim=True),
            torch.linalg.vector_norm(raw_target, dim=-1, keepdim=True),
        )


@torch.no_grad()
def evaluate(config: dict) -> dict[str, dict[str, dict[str, float]]]:
    if config.get('migration', {}).get('evaluation_requires_legacy_source'):
        raise ValueError('This historical checkpoint requires its archived exact model/evaluation source; the current model configuration cannot reproduce it.')
    runtime_config = config["runtime"]
    test_config = config["test"]
    task_config = config["task"]
    torch.set_float32_matmul_precision(runtime_config.get("matmul_precision", "high"))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    query_sets = int(test_config.get('query_sets', 10))
    if query_sets <= 0:
        raise ValueError('test.query_sets must be positive.')

    ds_config = dataset_config_for_phase(
        config,
        "test",
        {
            "deterministic": True,
            "seed": int(runtime_config.get("seed", 200099)),
            "repeat": query_sets,
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
    checkpoint = resolve_eval_checkpoint(config, "test")
    if checkpoint:
        load_model_weights(model, checkpoint, strict=True, trusted_legacy=bool(test_config.get("trust_legacy_checkpoint", False)))
        print(f"checkpoint loaded: {checkpoint}", flush=True)
    else:
        print("no checkpoint configured; running random-init smoke", flush=True)

    meters: dict[str, MetricAccumulator] = {}
    raw_meters: dict[str, MetricAccumulator] = {}
    max_batches = int(test_config.get("max_batches", 0))
    sampled_cases = 0
    for idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device, task_config["surface_input_list"])
        sampled_cases += len(batch['cond'])
        query = build_query(
            batch["surface_query"],
            batch["volume_query"],
            add_type_channel=bool(task_config.get("query_type_channel", True)),
        )
        finite_context = {
            "config": config.get("_config_path", ""),
            "phase": "test",
            "epoch": 0,
            "batch_index": idx,
            "global_step": idx,
            "cond": batch.get("cond"),
            "routes": batch.get("routes"),
        }
        ensure_finite(
            {
                "surface_pos": batch.get("surface_pos"),
                "surface_query": batch.get("surface_query"),
                "surface_target": batch.get("surface_target"),
                "volume_query": batch.get("volume_query"),
                "volume_target": batch.get("volume_target"),
                "cond": batch.get("cond"),
                "normals": batch.get("normals"),
                "query": query,
            },
            stage="test_input",
            device=device,
            saveroot=config.get("train", {}).get("saveroot"),
            context=finite_context,
            batch=batch,
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
                    geometry_meta=batch.get("geometry_meta"),
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
                    geometry_meta=batch.get("geometry_meta"),
                )
        else:
            pred = model(
                batch["surface_pos"],
                query,
                batch["cond"],
                routes=batch["routes"],
                normals=batch["normals"],
                area=batch["area"],
                geometry_meta=batch.get("geometry_meta"),
            )

        ensure_finite(
            pred,
            stage="test_forward",
            device=device,
            saveroot=config.get("train", {}).get("saveroot"),
            context=finite_context,
            batch=batch,
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
            elif domain == "coefficients":
                tgt = target_field(batch["coefficient_target"], task_config["coefficient_target_list"], field)
            else:
                raise ValueError(f"Unsupported domain: {domain}")
            ensure_finite(
                {"prediction": pred_field, "target": tgt},
                stage="test_metric_input",
                device=device,
                saveroot=config.get("train", {}).get("saveroot"),
                context={**finite_context, "field": key},
                batch=batch,
            )
            _update_metrics(meters, raw_meters, key, pred_field, tgt, field, task_config, batch, domain)

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
    summary = {
        "protocol": "deterministic_random_query_sets_v1",
        "query_sets": query_sets,
        "case_query_groups": sampled_cases,
        "complete": sampled_cases == len(dataset),
        "raw_space_transforms": task_config.get('denormalize', {}),
        "raw_space_note": "Only fields with an explicit inverse normalization are reported in raw_space.",
        "model_space": {key: meter.result() for key, meter in meters.items()},
        "raw_space": {key: meter.result() for key, meter in raw_meters.items()},
    }
    metrics_output = test_config.get("metrics_output", "")
    if metrics_output:
        output_path = Path(metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"metrics saved to: {output_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven UniField evaluation.")
    parser.add_argument("--config", required=True, help="Path to a Python config file exporting CONFIG.")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic evaluation seed and expand {seed} paths.")
    args = parser.parse_args()
    config = configure_training_run(load_main_config(args.config), for_training=False)
    apply_runtime_seed(config, args.seed)
    if config["train"].get("engine") == "dsapt":
        from utils.dsapt_evaluation import evaluate_dsapt
        evaluate_dsapt(config)
    else:
        evaluate(config)


if __name__ == "__main__":
    main()
