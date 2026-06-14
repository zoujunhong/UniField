from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def save_prediction_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def scatter_scalar(path: str | Path, xyz: np.ndarray, values: np.ndarray, title: str) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = values.reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=values, s=2, cmap="coolwarm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def scatter_volume_slice(
    path: str | Path,
    xyz: np.ndarray,
    values: np.ndarray,
    title: str,
    axis: int = 2,
    quantile_width: float = 0.08,
) -> Path:
    coord = xyz[:, axis]
    center = np.quantile(coord, 0.5)
    width = max(float(coord.max() - coord.min()) * float(quantile_width), 1e-12)
    mask = np.abs(coord - center) <= width
    if mask.sum() < 128:
        order = np.argsort(np.abs(coord - center))
        mask = np.zeros_like(coord, dtype=bool)
        mask[order[: min(2048, len(order))]] = True
    axes = [0, 1, 2]
    axes.remove(axis)
    return scatter_scalar(path, xyz[mask][:, axes], values[mask], title)
