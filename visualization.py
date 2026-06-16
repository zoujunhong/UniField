from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from dataset import build_dataset
from model import build_model
from test import denormalize
from train import load_main_config
from utils.checkpoint import load_model_weights
from utils.config import dataset_config_for_phase
from utils.task import build_query, move_batch_to_device, prediction_field, target_field
from utils.visualization import save_prediction_npz, scatter_scalar, scatter_volume_slice, to_numpy


@torch.no_grad()
def run_visualization(config: dict) -> None:
    runtime_config = config["runtime"]
    vis_config = config["visualization"]
    task_config = config["task"]
    torch.set_float32_matmul_precision(runtime_config.get("matmul_precision", "high"))
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    ds_config = dataset_config_for_phase(
        config,
        "visualization",
        {
            "deterministic": True,
            "seed": int(runtime_config.get("seed", 200099)),
        },
    )
    dataset = build_dataset(ds_config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(vis_config.get("batch_size", 1)),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=int(vis_config.get("workers", 0)),
        drop_last=False,
    )

    model = build_model(config["model"]).to(device).float().eval()
    if vis_config.get("checkpoint"):
        load_model_weights(model, vis_config["checkpoint"], strict=True)
        print("checkpoint loaded", flush=True)
    else:
        print("no checkpoint configured; running random-init visualization smoke", flush=True)

    batch = move_batch_to_device(next(iter(loader)), device, task_config["surface_input_list"])
    query = build_query(
        batch["surface_query"],
        batch["volume_query"],
        add_type_channel=bool(task_config.get("query_type_channel", True)),
    )
    pred = model.forward_test(
        batch["surface_pos"],
        query,
        batch["cond"],
        routes=batch["routes"],
        normals=batch["normals"],
        area=batch["area"],
        query_chunk_size=int(vis_config.get("query_chunk_size", 50000)),
    )

    surface_count = int(batch["surface_query"].shape[1])
    output_dir = vis_config.get("output_dir")
    if not output_dir:
        output_dir = Path("temp") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_main_visualization" / "visualization"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "surface_query": to_numpy(batch["surface_query"][0]),
        "volume_query": to_numpy(batch["volume_query"][0]),
        "prediction": to_numpy(pred[0]),
        "surface_target": to_numpy(batch["surface_target"][0]),
        "volume_target": to_numpy(batch["volume_target"][0]),
    }
    save_prediction_npz(output_dir / "summary.npz", **arrays)

    max_points = int(vis_config.get("max_points", 20000))
    for item in task_config.get("eval_fields", task_config.get("losses", [])):
        domain = item["domain"]
        field = item["field"]
        pred_field = prediction_field(pred, domain, field, {"losses": [item]}, surface_count)
        if domain == "surface":
            xyz = to_numpy(batch["surface_query"][0])
            tgt = target_field(batch["surface_target"], task_config["surface_target_list"], field)
            pred_np = to_numpy(denormalize(pred_field, field, task_config, batch)[0])
            tgt_np = to_numpy(denormalize(tgt, field, task_config, batch)[0])
            if xyz.shape[0] > max_points:
                xyz = xyz[:max_points]
                pred_np = pred_np[:max_points]
                tgt_np = tgt_np[:max_points]
            scatter_scalar(output_dir / f"surface_{field}_target.png", xyz, tgt_np, f"surface {field} target")
            scatter_scalar(output_dir / f"surface_{field}_pred.png", xyz, pred_np, f"surface {field} pred")
            scatter_scalar(output_dir / f"surface_{field}_error.png", xyz, pred_np - tgt_np, f"surface {field} error")
        elif domain == "volume":
            xyz = to_numpy(batch["volume_query"][0])
            tgt = target_field(batch["volume_target"], task_config["volume_target_list"], field)
            pred_np = to_numpy(denormalize(pred_field, field, task_config, batch)[0])
            tgt_np = to_numpy(denormalize(tgt, field, task_config, batch)[0])
            if xyz.shape[0] > max_points:
                xyz = xyz[:max_points]
                pred_np = pred_np[:max_points]
                tgt_np = tgt_np[:max_points]
            channels = pred_np.shape[-1]
            for channel in range(channels):
                suffix = f"{field}{channel}" if channels > 1 else field
                scatter_volume_slice(output_dir / f"volume_{suffix}_target.png", xyz, tgt_np[:, channel], f"volume {suffix} target")
                scatter_volume_slice(output_dir / f"volume_{suffix}_pred.png", xyz, pred_np[:, channel], f"volume {suffix} pred")
                scatter_volume_slice(output_dir / f"volume_{suffix}_error.png", xyz, pred_np[:, channel] - tgt_np[:, channel], f"volume {suffix} error")

    print(f"visualization saved to: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven UniField visualization.")
    parser.add_argument("--config", required=True, help="Path to a Python config file exporting CONFIG.")
    args = parser.parse_args()
    run_visualization(load_main_config(args.config))


if __name__ == "__main__":
    main()
