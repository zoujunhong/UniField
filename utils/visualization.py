from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch


PLANE_AXES = {
    "x": (1, 2, "y", "z"),
    "y": (0, 2, "x", "z"),
    "z": (0, 1, "x", "y"),
}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def save_prediction_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def scatter_scalar(
    path: str | Path,
    xyz: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = "coolwarm",
    clim: tuple[float, float] | None = None,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = values.reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=values, s=2, cmap=cmap)
    if clim is not None:
        sc.set_clim(*clim)
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
    cmap: str = "coolwarm",
    clim: tuple[float, float] | None = None,
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
    return scatter_scalar(path, xyz[mask][:, axes], values[mask], title, cmap=cmap, clim=clim)


def robust_limits(
    values: list[np.ndarray] | tuple[np.ndarray, ...],
    low: float = 1.0,
    high: float = 99.0,
    symmetric: bool = False,
) -> tuple[float, float]:
    finite = []
    for value in values:
        arr = np.asarray(value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        return 0.0, 1.0

    data = np.concatenate(finite)
    if symmetric:
        bound = float(np.percentile(np.abs(data), high))
        if bound <= 0.0 or not np.isfinite(bound):
            bound = 1.0
        return -bound, bound

    lo, hi = np.percentile(data, [low, high])
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)) * 0.05, 1e-12)
        return float(lo - pad), float(hi + pad)
    return float(lo), float(hi)


def set_plane_limits(ax, normal: str, limits: dict[str, tuple[float, float]] | None) -> None:
    if not limits:
        return
    if normal == "x":
        ax.set_xlim(*limits["y"])
        ax.set_ylim(*limits["z"])
    elif normal == "y":
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["z"])
    elif normal == "z":
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])


def plot_volume_slice_panels(
    path: str | Path,
    slices: dict[str, dict[str, np.ndarray]],
    title: str,
    max_points_per_slice: int = 200_000,
    cmap: str = "coolwarm",
    limits: dict[str, tuple[float, float]] | None = None,
    clim: tuple[float, float] | None = None,
    symmetric: bool = False,
    point_size: float = 0.35,
    alpha: float = 0.65,
    seed: int = 42,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planes = [normal for normal in ("x", "y", "z") if normal in slices]
    if clim is None:
        clim = robust_limits([slices[normal]["values"] for normal in planes], symmetric=symmetric)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, len(planes), figsize=(6.3 * len(planes), 5.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    mappable = None
    for ax, normal in zip(axes, planes):
        xyz = np.asarray(slices[normal]["xyz"], dtype=np.float32)
        values = np.asarray(slices[normal]["values"], dtype=np.float32).reshape(-1)
        if xyz.shape[0] > int(max_points_per_slice):
            idx = rng.choice(xyz.shape[0], size=int(max_points_per_slice), replace=False)
            xyz = xyz[idx]
            values = values[idx]
        u_idx, v_idx, xlabel, ylabel = PLANE_AXES[normal]
        mappable = ax.scatter(
            xyz[:, u_idx],
            xyz[:, v_idx],
            c=values,
            s=point_size,
            alpha=alpha,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            linewidths=0,
            rasterized=True,
        )
        set_plane_limits(ax, normal, limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{normal}-normal slice")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path
