"""Fail-fast finite-value checks shared by training and evaluation.

The helpers deliberately live outside entry-point scripts so every numerical
failure has a small, portable JSON record instead of silently contaminating
metrics or later checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist

from utils.distributed import is_dist_avail_and_initialized


class NonFiniteTrainingError(RuntimeError):
    """Raised collectively when any distributed rank sees NaN or Inf."""


_MAX_STATS_ELEMENTS = 1_000_000


def _iter_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix or "tensor", value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_tensors(item, child)


def _requires_finite_check(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_floating_point() or tensor.is_complex())


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Return JSON-safe tensor metadata, including finite counts when needed."""

    data = tensor.detach()
    result: dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "numel": int(data.numel()),
    }
    if not _requires_finite_check(data):
        result["finite"] = True
        return result

    finite = torch.isfinite(data)
    finite_count = int(finite.sum().item())
    result["finite"] = finite_count == data.numel()
    result["finite_count"] = finite_count
    result["nan_count"] = int(torch.isnan(data).sum().item())
    result["inf_count"] = int((~finite & ~torch.isnan(data)).sum().item())
    # Full reductions over a 500M-parameter model add significant memory at
    # exactly the point where a failure may already have exhausted memory.
    if finite_count and data.numel() <= _MAX_STATS_ELEMENTS and not data.is_complex():
        finite_values = data[finite]
        result["min"] = float(finite_values.min().item())
        result["max"] = float(finite_values.max().item())
        result["max_abs"] = float(finite_values.abs().max().item())
    return result


def find_nonfinite_tensors(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """List non-finite floating tensors in a nested value tree."""

    bad: list[dict[str, Any]] = []
    for name, tensor in _iter_tensors(value, prefix):
        if _requires_finite_check(tensor) and not bool(torch.isfinite(tensor).all().item()):
            summary = tensor_summary(tensor)
            summary["name"] = name
            bad.append(summary)
    return bad


def tensor_tree_isfinite(value: Any) -> bool:
    return not find_nonfinite_tensors(value)


def _rank_and_world() -> tuple[int, int]:
    if is_dist_avail_and_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def distributed_any(local_value: bool, device: torch.device | str | None = None) -> bool:
    """Synchronize a boolean failure flag without assuming a backend."""

    if not is_dist_avail_and_initialized():
        return bool(local_value)
    tensor_device = device
    if tensor_device is None:
        tensor_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
    flag = torch.tensor(int(bool(local_value)), dtype=torch.int32, device=tensor_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().item())
        return tensor_summary(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _diagnostic_directory(saveroot: str | Path, context: Mapping[str, Any]) -> Path:
    step = int(context.get("global_step", 0))
    epoch = int(context.get("epoch", -1))
    batch_index = int(context.get("batch_index", -1))
    return (
        Path(saveroot)
        / "nonfinite"
        / f"step_{step:08d}_epoch_{epoch:04d}_batch_{batch_index:05d}"
    )


def _write_failure_record(
    saveroot: str | Path,
    stage: str,
    context: Mapping[str, Any],
    local_bad: list[dict[str, Any]],
    batch: Any | None,
    save_batch: bool,
) -> Path:
    rank, world_size = _rank_and_world()
    directory = _diagnostic_directory(saveroot, context)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "rank": rank,
        "world_size": world_size,
        "context": _json_safe(context),
        "local_nonfinite_tensors": local_bad,
    }
    record_path = directory / f"rank_{rank}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if save_batch and batch is not None:
        try:
            torch.save(batch, directory / f"batch_rank_{rank}.pt")
        except Exception as error:  # keep the primary finite report usable
            record["batch_save_error"] = repr(error)
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if is_dist_avail_and_initialized():
        dist.barrier()
    if rank == 0:
        manifest = {
            "stage": stage,
            "world_size": world_size,
            "context": _json_safe(context),
            "rank_records": [f"rank_{index}.json" for index in range(world_size)],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if is_dist_avail_and_initialized():
        dist.barrier()
    return directory


def ensure_finite(
    value: Any,
    *,
    stage: str,
    device: torch.device | str | None = None,
    saveroot: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
    batch: Any | None = None,
    save_batch: bool = True,
) -> None:
    """Raise collectively and persist a diagnostic if any rank is non-finite."""

    local_bad = find_nonfinite_tensors(value)
    any_bad = distributed_any(bool(local_bad), device)
    if not any_bad:
        return
    details = dict(context or {})
    details["stage"] = stage
    directory = None
    if saveroot:
        directory = _write_failure_record(saveroot, stage, details, local_bad, batch, save_batch)
    rank, _ = _rank_and_world()
    suffix = f"; diagnostic: {directory}" if directory is not None else ""
    raise NonFiniteTrainingError(
        f"Non-finite value detected at stage={stage} (rank={rank}, local_bad={len(local_bad)}){suffix}"
    )


def named_parameter_tensors(model: torch.nn.Module, *, gradients: bool = False) -> dict[str, torch.Tensor]:
    """Expose parameters or their populated gradients for a finite check."""

    if gradients:
        return {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    return {name: parameter for name, parameter in model.named_parameters()}


def optimizer_state_tensors(optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    """Expose tensor-valued optimizer state without retaining parameter refs."""

    tensors: dict[str, torch.Tensor] = {}
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value):
                tensors[f"parameter_{parameter_index}.{name}"] = value
    return tensors


__all__ = [
    "NonFiniteTrainingError",
    "distributed_any",
    "ensure_finite",
    "find_nonfinite_tensors",
    "named_parameter_tensors",
    "optimizer_state_tensors",
    "tensor_summary",
    "tensor_tree_isfinite",
]
