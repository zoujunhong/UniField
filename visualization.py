from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from dataset import build_dataset
from dataset.utils.load import high_speed_type_dir
from dataset.utils.sampling import rng_for_index
from model import build_model
from test import denormalize
from train import load_main_config
from utils.checkpoint import load_model_weights, resolve_eval_checkpoint
from utils.config import dataset_config_for_phase
from utils.task import build_query, coefficient_prediction, move_batch_to_device, prediction_field, query_prediction, target_field
from utils.visualization import plot_volume_slice_panels, robust_limits, save_prediction_npz, scatter_scalar, scatter_volume_slice, to_numpy


def _mesh_center_from_bounds(bounds) -> tuple[float, float, float]:
    return (
        0.5 * (float(bounds[0]) + float(bounds[1])),
        0.5 * (float(bounds[2]) + float(bounds[3])),
        0.5 * (float(bounds[4]) + float(bounds[5])),
    )


def _crop_limits(surface_bounds) -> dict[str, tuple[float, float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in surface_bounds]
    x_limits = (xmin - 10.0, xmax + 100.0)
    y_limits = (ymin - 25.0, ymax + 45.0)
    z_min = max(0.0, zmin - 1.5)
    z_base_max = zmax + 6.0

    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    target_z_span = y_span * y_span / max(x_span + y_span, 1e-12)
    z_max = z_min + max(z_base_max - z_min, target_z_span)
    return {"x": x_limits, "y": y_limits, "z": (z_min, z_max)}


def _dataset_parts(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        offset = 0
        for subdataset in dataset.datasets:
            yield offset, subdataset
            offset += len(subdataset)
        return
    yield 0, dataset


def _dataset_at_index(dataset, index: int):
    for offset, subdataset in _dataset_parts(dataset):
        end = offset + len(subdataset)
        if offset <= index < end:
            return subdataset, index - offset
    raise IndexError(f"Visualization case_index={index} is out of range for dataset length {len(dataset)}.")


def _case_label(subdataset, local_idx: int) -> str:
    case_names = getattr(subdataset, "case_names", None)
    if case_names is None or len(case_names) == 0:
        return "<unknown>"
    case_name = case_names[int(local_idx) % len(case_names)]
    train_type = str(getattr(subdataset, "train_type", "")).strip()
    return f"{train_type}/{case_name}" if train_type else str(case_name)


def _resolve_visualization_sample(dataset, vis_config: dict) -> tuple[int, object, int]:
    case_name = str(vis_config.get("case_name", "")).strip()
    if case_name:
        for offset, subdataset in _dataset_parts(dataset):
            case_names = getattr(subdataset, "case_names", None)
            if case_names is None:
                continue
            if case_name in case_names:
                local_idx = list(case_names).index(case_name)
                return offset + local_idx, subdataset, local_idx

        if bool(vis_config.get("strict_case_name", False)):
            raise ValueError(f"Visualization case_name not found in dataset: {case_name}")

        sample_index = int(vis_config.get("case_index", 0))
        subdataset, local_idx = _dataset_at_index(dataset, sample_index)
        print(
            "Visualization case_name not found in current dataset: "
            f"{case_name}; fallback to case_index={sample_index} ({_case_label(subdataset, local_idx)}).",
            flush=True,
        )
        return sample_index, subdataset, local_idx

    sample_index = int(vis_config.get("case_index", 0))
    subdataset, local_idx = _dataset_at_index(dataset, sample_index)
    return sample_index, subdataset, local_idx


def _high_speed_x_center(dataset, case_name: str, local_idx: int) -> float:
    if not bool(getattr(dataset, "center_x", False)):
        return 0.0
    surface = getattr(dataset, "surface", None)
    if surface is None:
        return 0.0
    rng = rng_for_index(int(getattr(dataset, "seed", 0)), int(local_idx), bool(getattr(dataset, "deterministic", False)))
    surface_input, _, _ = surface.sample(case_name, rng)
    if surface_input.shape[0] == 0:
        return 0.0
    return float(0.5 * (surface_input[:, 0].max() + surface_input[:, 0].min()))


def _vtk_cell_array(slice_mesh, field: str) -> np.ndarray:
    if field in slice_mesh.cell_data:
        values = np.asarray(slice_mesh.cell_data[field], dtype=np.float32)
    elif field in slice_mesh.point_data:
        converted = slice_mesh.point_data_to_cell_data(pass_point_data=False)
        values = np.asarray(converted.cell_data[field], dtype=np.float32)
    else:
        raise KeyError(f"Field {field!r} not found in VTK slice.")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values.astype(np.float32, copy=False)


def _volume_eval_items(task_config: dict) -> list[dict]:
    items = []
    seen = set()
    for item in task_config.get("eval_fields", task_config.get("losses", [])):
        if item["domain"] != "volume":
            continue
        field = item["field"]
        if field in seen:
            continue
        seen.add(field)
        items.append(item)
    return items


def _dense_plot_spec(field: str, target: np.ndarray, pred: np.ndarray):
    if field == "U":
        target_scalar = np.linalg.norm(target, axis=-1)
        pred_scalar = np.linalg.norm(pred, axis=-1)
        return "speed", "speed", "viridis", target_scalar, pred_scalar
    return field, field, "coolwarm", target[:, 0], pred[:, 0]


POSITIVE_TURBULENCE_FIELDS = {"k", "omega", "nut"}


def _relative_error(pred: np.ndarray, target: np.ndarray, delta: float) -> np.ndarray:
    return np.abs(pred - target) / (np.abs(target) + float(delta))


def _relative_error_clim(values: list[np.ndarray] | tuple[np.ndarray, ...]) -> tuple[float, float]:
    lo, hi = robust_limits(values, low=0.0, high=99.0)
    return max(0.0, float(lo)), float(hi)


@torch.no_grad()
def _predict_volume_fields(
    model,
    batch: dict,
    task_config: dict,
    vis_config: dict,
    volume_query_xyz: np.ndarray,
    x_center: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    centered_query = volume_query_xyz.astype(np.float32, copy=True)
    centered_query[:, 0] -= float(x_center)
    volume_query = torch.from_numpy(centered_query).unsqueeze(0).to(device=device, dtype=torch.float32)
    surface_query = batch["surface_query"][:, :0, :]
    query = build_query(
        surface_query,
        volume_query,
        add_type_channel=bool(task_config.get("query_type_channel", True)),
    )
    pred = model.forward_test(
        batch["surface_pos"],
        query,
        batch["cond"],
        routes=batch["routes"],
        normals=batch["normals"],
        area=batch["area"],
        geometry_meta=batch.get("geometry_meta"),
        query_chunk_size=int(vis_config.get("query_chunk_size", 50000)),
    )
    fields = {}
    for item in _volume_eval_items(task_config):
        field = item["field"]
        pred_field = prediction_field(pred, "volume", field, {"losses": [item]}, surface_count=0)
        fields[field] = to_numpy(denormalize(pred_field, field, task_config, batch)[0])
    return fields


def _run_high_speed_vtk_volume_visualization(
    case_dataset,
    local_idx: int,
    model,
    batch: dict,
    task_config: dict,
    vis_config: dict,
    output_dir: Path,
    device: torch.device,
) -> bool:
    if case_dataset is None or not hasattr(case_dataset, "case_names") or not hasattr(case_dataset, "train_type"):
        return False
    if not case_dataset.case_names:
        return False

    case_name = case_dataset.case_names[int(local_idx) % len(case_dataset.case_names)]
    case_dir = high_speed_type_dir(case_dataset.root_dir, case_dataset.train_type) / case_name
    full_path = case_dir / "full_field.vtk"
    surface_path = case_dir / "surface.vtk"
    if not full_path.is_file() or not surface_path.is_file():
        print(f"VTK dense visualization skipped: missing {full_path} or {surface_path}", flush=True)
        return False

    import pyvista as pv

    pv.OFF_SCREEN = True
    planes = tuple(vis_config.get("volume_slice_planes", ("x", "y", "z")))
    max_points = int(vis_config.get("volume_slice_max_points", vis_config.get("max_points", 200000)))
    relative_error_delta = float(vis_config.get("relative_error_delta", 1e-8))
    x_center = _high_speed_x_center(case_dataset, case_name, local_idx)

    print(
        "VTK dense volume visualization: "
        f"{case_dataset.train_type}/{case_name}, planes={planes}, max_points_per_slice={max_points}",
        flush=True,
    )
    surface = pv.read(surface_path)
    center = _mesh_center_from_bounds(surface.bounds)
    limits = _crop_limits(surface.bounds)
    full = pv.read(full_path)

    target_by_field: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    pred_by_field: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    color_limits = {}
    slice_info = {}
    for plane in planes:
        slice_mesh = full.slice(normal=plane, origin=center).extract_surface(algorithm="dataset_surface").triangulate()
        centers = np.asarray(slice_mesh.cell_centers().points, dtype=np.float32)
        slice_info[plane] = {"points": int(slice_mesh.n_points), "cells": int(slice_mesh.n_cells)}
        pred_fields = _predict_volume_fields(model, batch, task_config, vis_config, centers, x_center, device)
        for item in _volume_eval_items(task_config):
            field = item["field"]
            target_values = _vtk_cell_array(slice_mesh, field)
            n = min(centers.shape[0], target_values.shape[0], pred_fields[field].shape[0])
            target_by_field.setdefault(field, {})[plane] = {"xyz": centers[:n], "values": target_values[:n]}
            pred_by_field.setdefault(field, {})[plane] = {"xyz": centers[:n], "values": pred_fields[field][:n]}

    for item in _volume_eval_items(task_config):
        field = item["field"]
        first_plane = next(iter(target_by_field[field].values()))
        del first_plane
        target_slices = {}
        pred_slices = {}
        error_slices = {}
        relative_error_slices = {}
        suffix = field
        title_field = field
        cmap = "coolwarm"
        for plane, target_values in target_by_field[field].items():
            pred_values = pred_by_field[field][plane]
            suffix, title_field, cmap, target_scalar, pred_scalar = _dense_plot_spec(
                field,
                target_values["values"],
                pred_values["values"],
            )
            target_slices[plane] = {"xyz": target_values["xyz"], "values": target_scalar}
            pred_slices[plane] = {"xyz": pred_values["xyz"], "values": pred_scalar}
            error_slices[plane] = {
                "xyz": target_values["xyz"],
                "values": pred_scalar - target_scalar,
            }
            relative_error_slices[plane] = {
                "xyz": target_values["xyz"],
                "values": _relative_error(pred_scalar, target_scalar, relative_error_delta),
            }
        target_pred_clim = robust_limits(
            [target_slices[plane]["values"] for plane in target_slices]
            + [pred_slices[plane]["values"] for plane in pred_slices]
        )
        if field in POSITIVE_TURBULENCE_FIELDS:
            error_clim = robust_limits(
                [error_slices[plane]["values"] for plane in error_slices],
                symmetric=True,
            )
            relative_error_clim = _relative_error_clim(
                [relative_error_slices[plane]["values"] for plane in relative_error_slices]
            )
            color_limits[suffix] = {
                "cmap": cmap,
                "target_pred_clim": [float(target_pred_clim[0]), float(target_pred_clim[1])],
                "symmetric_error_clim": [float(error_clim[0]), float(error_clim[1])],
                "relative_error_clim": [float(relative_error_clim[0]), float(relative_error_clim[1])],
            }
        else:
            shared_clim = robust_limits(
                [target_slices[plane]["values"] for plane in target_slices]
                + [pred_slices[plane]["values"] for plane in pred_slices]
                + [error_slices[plane]["values"] for plane in error_slices]
            )
            target_pred_clim = shared_clim
            error_clim = shared_clim
            relative_error_clim = _relative_error_clim(
                [relative_error_slices[plane]["values"] for plane in relative_error_slices]
            )
            color_limits[suffix] = {
                "cmap": cmap,
                "target_pred_error_clim": [float(shared_clim[0]), float(shared_clim[1])],
                "relative_error_clim": [float(relative_error_clim[0]), float(relative_error_clim[1])],
            }
        plot_volume_slice_panels(
            output_dir / f"volume_{suffix}_target.png",
            target_slices,
            f"volume {title_field} target",
            max_points_per_slice=max_points,
            cmap=cmap,
            limits=limits,
            clim=target_pred_clim,
            seed=17,
        )
        plot_volume_slice_panels(
            output_dir / f"volume_{suffix}_pred.png",
            pred_slices,
            f"volume {title_field} pred",
            max_points_per_slice=max_points,
            cmap=cmap,
            limits=limits,
            clim=target_pred_clim,
            seed=17,
        )
        plot_volume_slice_panels(
            output_dir / f"volume_{suffix}_error.png",
            error_slices,
            f"volume {title_field} error",
            max_points_per_slice=max_points,
            cmap=cmap,
            limits=limits,
            clim=error_clim,
            seed=17,
        )
        plot_volume_slice_panels(
            output_dir / f"volume_{suffix}_relative_error.png",
            relative_error_slices,
            f"volume {title_field} relative error",
            max_points_per_slice=max_points,
            cmap="magma",
            limits=limits,
            clim=relative_error_clim,
            seed=17,
        )

    metadata = {
        "case": case_name,
        "train_type": case_dataset.train_type,
        "case_dir": str(case_dir),
        "surface_vtk": str(surface_path),
        "full_vtk": str(full_path),
        "center": [float(v) for v in center],
        "surface_bounds": [float(v) for v in surface.bounds],
        "x_center_for_model_query": x_center,
        "slices": slice_info,
        "color_limits": color_limits,
        "relative_error_delta": relative_error_delta,
    }
    (output_dir / "vtk_slice_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return True


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
    sample_index, case_dataset, local_idx = _resolve_visualization_sample(dataset, vis_config)
    raw_batch = default_collate([dataset[sample_index]])
    if hasattr(case_dataset, "case_names"):
        case_name = case_dataset.case_names[int(local_idx) % len(case_dataset.case_names)]
        print(f"visualization case: {getattr(case_dataset, 'train_type', '')}/{case_name}", flush=True)

    model = build_model(config["model"]).to(device).float().eval()
    checkpoint = resolve_eval_checkpoint(config, "visualization")
    if checkpoint:
        load_model_weights(model, checkpoint, strict=True)
        print(f"checkpoint loaded: {checkpoint}", flush=True)
    else:
        print("no checkpoint configured; running random-init visualization smoke", flush=True)

    batch = move_batch_to_device(raw_batch, device, task_config["surface_input_list"])
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
        geometry_meta=batch.get("geometry_meta"),
        query_chunk_size=int(vis_config.get("query_chunk_size", 50000)),
    )
    query_pred = query_prediction(pred)
    coeff_pred = coefficient_prediction(pred)

    surface_count = int(batch["surface_query"].shape[1])
    output_dir = vis_config.get("output_dir")
    if not output_dir:
        output_dir = Path("temp") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_main_visualization" / "visualization"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "surface_query": to_numpy(batch["surface_query"][0]),
        "volume_query": to_numpy(batch["volume_query"][0]),
        "prediction": to_numpy(query_pred[0]),
        "surface_target": to_numpy(batch["surface_target"][0]),
        "volume_target": to_numpy(batch["volume_target"][0]),
        "coefficient_target": to_numpy(batch["coefficient_target"][0]),
    }
    if coeff_pred is not None:
        arrays["coefficient_prediction"] = to_numpy(coeff_pred[0])
    save_prediction_npz(output_dir / "summary.npz", **arrays)

    dense_volume_plotted = False
    if bool(vis_config.get("volume_use_vtk_slices", False)):
        dense_volume_plotted = _run_high_speed_vtk_volume_visualization(
            case_dataset,
            local_idx,
            model,
            batch,
            task_config,
            vis_config,
            output_dir,
            device,
        )

    max_points = int(vis_config.get("max_points", 20000))
    relative_error_delta = float(vis_config.get("relative_error_delta", 1e-8))
    for item in task_config.get("eval_fields", task_config.get("losses", [])):
        domain = item["domain"]
        field = item["field"]
        if dense_volume_plotted and domain == "volume":
            continue
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
            scatter_scalar(
                output_dir / f"surface_{field}_relative_error.png",
                xyz,
                _relative_error(pred_np, tgt_np, relative_error_delta),
                f"surface {field} relative error",
                cmap="magma",
            )
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
                scatter_volume_slice(
                    output_dir / f"volume_{suffix}_relative_error.png",
                    xyz,
                    _relative_error(pred_np[:, channel], tgt_np[:, channel], relative_error_delta),
                    f"volume {suffix} relative error",
                    cmap="magma",
                )
        elif domain == "coefficients":
            continue
        else:
            raise ValueError(f"Unsupported domain: {domain}")

    print(f"visualization saved to: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven UniField visualization.")
    parser.add_argument("--config", required=True, help="Path to a Python config file exporting CONFIG.")
    args = parser.parse_args()
    run_visualization(load_main_config(args.config))


if __name__ == "__main__":
    main()
