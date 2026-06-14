from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.utils.load import list_high_speed_cases, read_high_speed_field
from dataset.utils.sampling import sample_rows_np, voxel_uniform_sample_rows


HIGH_SPEED_TRAIN_ROOT = "/data/home/zdhs0017/OpenFOAM/Cases/high_speed_train_debug_parallel_data"
DEFAULT_SURFACE_CACHE_POINTS = 100_000
DEFAULT_VOLUME_CACHE_POINTS = 300_000


def normalized_train_type(train_type: str) -> str:
    return "Maglev" if train_type.lower() == "maglev" else train_type


def cache_dir(cache_root: str | Path, train_type: str, kind: str, points: int) -> str:
    return os.path.join(str(cache_root), normalized_train_type(train_type), f"{kind}_{int(points)}_fp16")


def cache_path(cache_root: str | Path, train_type: str, kind: str, points: int, case_name: str) -> str:
    return os.path.join(cache_dir(cache_root, train_type, kind, points), f"{case_name}.npy")


def identity_collate(batch):
    return batch


class HighSpeedTrainCacheBuilder(Dataset):
    """
    Build fp16 caches for high_speed_train cases.

    Surface rows:
        [x, y, z, p, Ux, Uy, Uz, k, omega, nut, nx, ny, nz, area]
    Volume rows:
        [x, y, z, p, Ux, Uy, Uz, k, omega, nut]
    """

    def __init__(
        self,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
        train_type: str = "CRH450",
        cache_root: str | None = None,
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        surface_sampling: str = "random",
        overwrite: bool = False,
        build_surface: bool = True,
        build_volume: bool = True,
    ) -> None:
        if surface_sampling not in {"random", "uniform"}:
            raise ValueError("surface_sampling must be 'random' or 'uniform'.")
        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.surface_cache_points = int(surface_cache_points)
        self.volume_cache_points = int(volume_cache_points)
        self.surface_sampling = surface_sampling
        self.overwrite = bool(overwrite)
        self.build_surface = bool(build_surface)
        self.build_volume = bool(build_volume)
        if not self.build_surface and not self.build_volume:
            raise ValueError("At least one of build_surface/build_volume must be True.")

        self.case_dirs = list_high_speed_cases(self.root_dir, self.train_type)
        for kind, points in [
            (f"surface_{self.surface_sampling}", self.surface_cache_points),
            ("volume_random", self.volume_cache_points),
        ]:
            if kind.startswith("surface") and not self.build_surface:
                continue
            if kind.startswith("volume") and not self.build_volume:
                continue
            os.makedirs(cache_dir(self.cache_root, self.train_type, kind, points), exist_ok=True)
        print(f"HighSpeedTrain cache builder: {self.train_type}, {len(self.case_dirs)} cases.")

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        case_dir = self.case_dirs[idx]
        rng = np.random.default_rng(idx)
        written: dict[str, str] = {}

        if self.build_surface:
            kind = f"surface_{self.surface_sampling}"
            out_path = cache_path(self.cache_root, self.train_type, kind, self.surface_cache_points, case_dir.name)
            if self.overwrite or not os.path.exists(out_path):
                rows = read_high_speed_field(case_dir / "surface.vtk", include_geometry=True, location="cell")
                if self.surface_sampling == "uniform":
                    rows = voxel_uniform_sample_rows(rows, self.surface_cache_points, rng)
                else:
                    rows = sample_rows_np(rows, self.surface_cache_points, rng)
                np.save(out_path, rows.astype(np.float16))
            written[kind] = out_path

        if self.build_volume:
            out_path = cache_path(self.cache_root, self.train_type, "volume_random", self.volume_cache_points, case_dir.name)
            if self.overwrite or not os.path.exists(out_path):
                rows = read_high_speed_field(case_dir / "full_field.vtk", include_geometry=False, location="cell")
                rows = sample_rows_np(rows, self.volume_cache_points, rng)
                np.save(out_path, rows.astype(np.float16))
            written["volume_random"] = out_path
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
                self[i]
                if log_every > 0 and ((i - start_index + 1) % log_every == 0 or i + 1 == end):
                    print(f"[{i + 1}/{len(self)}] cached {self.case_dirs[i].name}", flush=True)
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
            if log_every > 0 and (local_i % log_every == 0 or global_i + 1 == end):
                print(f"[{global_i + 1}/{len(self)}] cached {self.case_dirs[global_i].name}", flush=True)
