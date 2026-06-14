from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.utils.load import (
    list_vtk_files,
    matched_vtk_names,
    read_drivaer_surface_pressure,
    read_drivaer_volume_velocity,
)
from dataset.utils.sampling import filter_names, sample_rows_np, voxel_uniform_sample_rows


DRIVAERNET_ROOT = "/data/group/project1/CFD/DrivAerNet++"
DEFAULT_CSV_FILE = os.path.join(DRIVAERNET_ROOT, "DrivAerNetPlusPlus_Cd_8k_Updated.csv")
DEFAULT_CACHE_ROOT = os.path.join(DRIVAERNET_ROOT, "cache")
DEFAULT_SURFACE_CACHE_POINTS = 100_000
DEFAULT_VOLUME_CACHE_POINTS = 300_000


def cache_dir(cache_root: str | Path, kind: str, points: int) -> str:
    return os.path.join(str(cache_root), f"{kind}_{int(points)}_fp16")


def cache_path(cache_root: str | Path, kind: str, points: int, case_name: str) -> str:
    return os.path.join(cache_dir(cache_root, kind, points), f"{case_name}.npy")


def identity_collate(batch):
    return batch


class DrivAerNetPlusPlusCacheBuilder(Dataset):
    """
    Build fp16 caches for DrivAerNet++.

    Surface rows:
        [x, y, z, p, nx, ny, nz, area]
    Volume rows:
        [x, y, z, Ux, Uy, Uz]
    """

    def __init__(
        self,
        root_dir: str = DRIVAERNET_ROOT,
        cache_root: str | None = None,
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        surface_sampling: str = "both",
        volume_location: str = "point",
        ids_file: str | None = None,
        overwrite: bool = False,
        build_surface: bool = True,
        build_volume: bool = True,
    ) -> None:
        if surface_sampling not in {"random", "uniform", "both"}:
            raise ValueError("surface_sampling must be 'random', 'uniform', or 'both'.")
        if volume_location not in {"point", "cell"}:
            raise ValueError("volume_location must be 'point' or 'cell'.")
        self.root_dir = root_dir
        self.pressure_dir = os.path.join(root_dir, "Pressure")
        self.cfd_dir = os.path.join(root_dir, "CFD")
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.surface_cache_points = int(surface_cache_points)
        self.volume_cache_points = int(volume_cache_points)
        self.surface_sampling = surface_sampling
        self.volume_location = volume_location
        self.overwrite = bool(overwrite)
        self.build_surface = bool(build_surface)
        self.build_volume = bool(build_volume)
        if not self.build_surface and not self.build_volume:
            raise ValueError("At least one of build_surface/build_volume must be True.")

        self.pressure_files = list_vtk_files(self.pressure_dir)
        self.cfd_files = list_vtk_files(self.cfd_dir)
        self.case_names = filter_names(matched_vtk_names(self.pressure_dir, self.cfd_dir), ids_file)
        if not self.case_names:
            raise RuntimeError("No DrivAerNet++ cases matched the requested ids_file.")

        kinds: list[str] = []
        if self.build_volume:
            kinds.append("volume_random")
        if self.build_surface and surface_sampling in {"random", "both"}:
            kinds.append("surface_random")
        if self.build_surface and surface_sampling in {"uniform", "both"}:
            kinds.append("surface_uniform")
        for kind in kinds:
            points = self.volume_cache_points if kind == "volume_random" else self.surface_cache_points
            os.makedirs(cache_dir(self.cache_root, kind, points), exist_ok=True)

        print(f"DrivAerNet++ cache builder: {len(self.case_names)} matched cases.")

    def __len__(self) -> int:
        return len(self.case_names)

    def _surface_paths(self, case_name: str) -> list[str]:
        paths = []
        if self.surface_sampling in {"random", "both"}:
            paths.append(cache_path(self.cache_root, "surface_random", self.surface_cache_points, case_name))
        if self.surface_sampling in {"uniform", "both"}:
            paths.append(cache_path(self.cache_root, "surface_uniform", self.surface_cache_points, case_name))
        return paths

    def _volume_path(self, case_name: str) -> str:
        return cache_path(self.cache_root, "volume_random", self.volume_cache_points, case_name)

    def __getitem__(self, idx: int) -> dict[str, str]:
        case_name = self.case_names[idx]
        rng = np.random.default_rng(idx)
        written: dict[str, str] = {}

        surface_paths = self._surface_paths(case_name)
        need_surface = self.build_surface and (self.overwrite or any(not os.path.exists(p) for p in surface_paths))
        if need_surface:
            surface = read_drivaer_surface_pressure(self.pressure_files[case_name])
            if self.surface_sampling in {"random", "both"}:
                path = cache_path(self.cache_root, "surface_random", self.surface_cache_points, case_name)
                if self.overwrite or not os.path.exists(path):
                    np.save(path, sample_rows_np(surface, self.surface_cache_points, rng).astype(np.float16))
                written["surface_random"] = path
            if self.surface_sampling in {"uniform", "both"}:
                path = cache_path(self.cache_root, "surface_uniform", self.surface_cache_points, case_name)
                if self.overwrite or not os.path.exists(path):
                    np.save(path, voxel_uniform_sample_rows(surface, self.surface_cache_points, rng).astype(np.float16))
                written["surface_uniform"] = path

        volume_path = self._volume_path(case_name)
        if self.build_volume and (self.overwrite or not os.path.exists(volume_path)):
            volume = read_drivaer_volume_velocity(self.cfd_files[case_name], location=self.volume_location)
            np.save(volume_path, sample_rows_np(volume, self.volume_cache_points, rng).astype(np.float16))
            written["volume_random"] = volume_path
        return written

    def build_cache(
        self,
        start_index: int = 0,
        limit: int | None = None,
        log_every: int = 1,
        num_workers: int = 0,
    ) -> None:
        end = len(self) if limit is None else min(len(self), int(start_index) + int(limit))
        indices = list(range(int(start_index), end))
        if num_workers <= 0:
            for i in indices:
                case_name = self.case_names[i]
                self[i]
                if log_every > 0 and ((i - start_index + 1) % log_every == 0 or i + 1 == end):
                    print(f"[{i + 1}/{len(self)}] cached {case_name}", flush=True)
            return

        subset = Subset(self, indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate,
        )
        for local_i, _ in enumerate(loader, start=1):
            global_i = int(start_index) + local_i - 1
            case_name = self.case_names[global_i]
            if log_every > 0 and (local_i % log_every == 0 or global_i + 1 == end):
                print(f"[{global_i + 1}/{len(self)}] cached {case_name}", flush=True)


def running_cache_stats(paths: list[str], columns: list[int], log_every: int = 100):
    count = 0
    total = np.zeros(len(columns), dtype=np.float64)
    total_sq = np.zeros(len(columns), dtype=np.float64)
    min_val = np.full(len(columns), np.inf, dtype=np.float64)
    max_val = np.full(len(columns), -np.inf, dtype=np.float64)
    for i, path in enumerate(paths):
        arr = np.load(path, mmap_mode="r")
        values = np.asarray(arr[:, columns], dtype=np.float64)
        count += values.shape[0]
        total += values.sum(axis=0)
        total_sq += (values * values).sum(axis=0)
        min_val = np.minimum(min_val, values.min(axis=0))
        max_val = np.maximum(max_val, values.max(axis=0))
        if log_every > 0 and ((i + 1) % log_every == 0 or i + 1 == len(paths)):
            print(f"[{i + 1}/{len(paths)}] scanned {Path(path).name}", flush=True)
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    return {"count": count, "mean": mean, "std": np.sqrt(var), "min": min_val, "max": max_val}


def compute_cache_statistics(
    root_dir: str = DRIVAERNET_ROOT,
    cache_root: str | None = None,
    surface_sampling: str = "random",
    surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
    volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
    ids_file: str | None = None,
    log_every: int = 100,
):
    cache_root = cache_root or os.path.join(root_dir, "cache")
    case_names = filter_names(
        matched_vtk_names(os.path.join(root_dir, "Pressure"), os.path.join(root_dir, "CFD")),
        ids_file,
    )
    surface_paths = [
        cache_path(cache_root, f"surface_{surface_sampling}", surface_cache_points, case_name)
        for case_name in case_names
    ]
    volume_paths = [cache_path(cache_root, "volume_random", volume_cache_points, case_name) for case_name in case_names]
    missing = [path for path in surface_paths + volume_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cache files. First missing: {missing[0]}")
    return {
        "pressure": running_cache_stats(surface_paths, [3], log_every=log_every),
        "velocity": running_cache_stats(volume_paths, [3, 4, 5], log_every=log_every),
    }
