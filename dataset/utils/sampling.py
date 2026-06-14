from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ids(ids_file: str | Path | None) -> set[str] | None:
    if ids_file is None:
        return None
    with open(ids_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def filter_names(names: list[str], ids_file: str | Path | None = None) -> list[str]:
    ids = read_ids(ids_file)
    if ids is None:
        return names
    return [name for name in names if name in ids]


def sample_rows_np(arr: np.ndarray, num: int, rng: np.random.Generator) -> np.ndarray:
    num = int(num)
    if num < 0:
        raise ValueError("num must be non-negative.")
    if num == 0:
        return np.asarray(arr[:0]).copy()
    n = arr.shape[0]
    if n == 0:
        raise ValueError("Cannot sample from an empty array.")
    if n == num:
        return np.asarray(arr).copy()
    replace = n < num
    idx = rng.choice(n, size=num, replace=replace)
    return np.asarray(arr[idx]).copy()


def rng_for_index(seed: int, idx: int, deterministic: bool) -> np.random.Generator:
    if deterministic:
        return np.random.default_rng(int(seed) + int(idx))
    return np.random.default_rng()


def grid_keys(grid: np.ndarray) -> np.ndarray:
    grid = grid - grid.min(axis=0, keepdims=True)
    dims = grid.max(axis=0).astype(np.int64) + 1
    return grid[:, 0] + dims[0] * (grid[:, 1] + dims[1] * grid[:, 2])


def occupied_voxel_count(xyz: np.ndarray, mins: np.ndarray, cell_size: float) -> int:
    grid = np.floor((xyz - mins) / cell_size).astype(np.int64)
    return int(np.unique(grid_keys(grid)).size)


def voxel_uniform_sample_rows(
    arr: np.ndarray,
    num: int,
    rng: np.random.Generator,
    max_search_iter: int = 18,
) -> np.ndarray:
    num = int(num)
    if num <= 0:
        return np.asarray(arr[:0]).copy()
    n = arr.shape[0]
    if n <= num:
        return sample_rows_np(arr, num, rng)

    xyz = arr[:, :3].astype(np.float32, copy=False)
    mins = xyz.min(axis=0)
    extent = np.maximum(xyz.max(axis=0) - mins, 1e-6)

    large = float(extent.max() + 1e-6)
    small = large
    for _ in range(32):
        small *= 0.5
        if occupied_voxel_count(xyz, mins, small) >= num:
            break
    else:
        return sample_rows_np(arr, num, rng)

    for _ in range(max_search_iter):
        mid = 0.5 * (small + large)
        if occupied_voxel_count(xyz, mins, mid) >= num:
            small = mid
        else:
            large = mid

    grid = np.floor((xyz - mins) / small).astype(np.int64)
    key = grid_keys(grid)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(key_sorted)) + 1]
    ends = np.r_[starts[1:], len(order)]

    picks = np.empty(len(starts), dtype=np.int64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        picks[i] = order[start + rng.integers(end - start)]

    if picks.shape[0] >= num:
        picks = rng.choice(picks, size=num, replace=False)
        return np.asarray(arr[picks]).copy()

    remaining = np.setdiff1d(np.arange(n), picks, assume_unique=False)
    extra_count = num - picks.shape[0]
    extra = rng.choice(remaining, size=extra_count, replace=remaining.shape[0] < extra_count)
    return np.asarray(arr[np.concatenate([picks, extra])]).copy()
