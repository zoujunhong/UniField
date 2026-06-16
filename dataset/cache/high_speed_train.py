from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.utils.load import list_high_speed_cases, read_high_speed_field
from dataset.utils.sampling import sample_rows_np, voxel_uniform_sample_rows


HIGH_SPEED_TRAIN_ROOT = "/data/home/zdhs0017/OpenFOAM/Cases/high_speed_train_debug_parallel_data"
DEFAULT_SURFACE_CACHE_POINTS = 100_000
DEFAULT_VOLUME_CACHE_POINTS = 300_000
HIGH_SPEED_TRAIN_GEOMETRY_COUNTS = {
    "CRH450": 20,
    "Maglev": 30,
}


def normalized_train_type(train_type: str) -> str:
    return "Maglev" if train_type.lower() == "maglev" else train_type


def parse_high_speed_geo_id(path_or_name: str | Path) -> int | None:
    match = re.search(r"_geo_(\d+)_", Path(path_or_name).name)
    if match is None:
        return None
    return int(match.group(1))


def high_speed_case_sort_key(path_or_name: str | Path) -> tuple[int, int, str]:
    name = Path(path_or_name).name
    geo_id = parse_high_speed_geo_id(name)
    match = re.search(r"_geo_\d+_(\d+)_", name)
    run_id = int(match.group(1)) if match is not None else -1
    return (-1 if geo_id is None else geo_id, run_id, name)


def cache_dir(cache_root: str | Path, train_type: str, kind: str, points: int) -> str:
    return os.path.join(str(cache_root), normalized_train_type(train_type), f"{kind}_{int(points)}_fp16")


def cache_path(cache_root: str | Path, train_type: str, kind: str, points: int, case_name: str) -> str:
    return os.path.join(cache_dir(cache_root, train_type, kind, points), f"{case_name}.npy")


def identity_collate(batch):
    return batch


def valid_npy_cache(path: str | Path, min_cols: int = 1) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < int(min_cols):
            return False
    except Exception:
        return False
    return True


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def complete_high_speed_raw_case(case_dir: str | Path, build_surface: bool, build_volume: bool) -> bool:
    case_dir = Path(case_dir)
    if build_surface:
        surface_path = case_dir / "surface.vtk"
        if not surface_path.is_file() or surface_path.stat().st_size <= 0:
            return False
    if build_volume:
        volume_path = case_dir / "full_field.vtk"
        if not volume_path.is_file() or volume_path.stat().st_size <= 0:
            return False
    return True


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
        skip_incomplete: bool = True,
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
        self.skip_incomplete = bool(skip_incomplete)
        if not self.build_surface and not self.build_volume:
            raise ValueError("At least one of build_surface/build_volume must be True.")

        case_dirs = sorted(list_high_speed_cases(self.root_dir, self.train_type), key=high_speed_case_sort_key)
        if self.skip_incomplete:
            complete_case_dirs = [
                path
                for path in case_dirs
                if complete_high_speed_raw_case(path, self.build_surface, self.build_volume)
            ]
            skipped = len(case_dirs) - len(complete_case_dirs)
            if skipped > 0:
                print(
                    f"HighSpeedTrain cache builder: skipped {skipped} incomplete raw cases "
                    f"for {self.train_type}.",
                    flush=True,
                )
            case_dirs = complete_case_dirs
        self.case_dirs = case_dirs
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

    def _read_case_field(self, case_dir: Path, file_name: str, include_geometry: bool) -> np.ndarray | None:
        try:
            return read_high_speed_field(case_dir / file_name, include_geometry=include_geometry, location="cell")
        except Exception as exc:
            if not self.skip_incomplete:
                raise
            print(
                f"HighSpeedTrain cache builder: skipped incomplete case {case_dir.name}: {exc}",
                flush=True,
            )
            return None

    def __getitem__(self, idx: int) -> dict[str, str]:
        case_dir = self.case_dirs[idx]
        rng = np.random.default_rng(idx)
        written: dict[str, str] = {}

        if self.build_surface:
            kind = f"surface_{self.surface_sampling}"
            out_path = cache_path(self.cache_root, self.train_type, kind, self.surface_cache_points, case_dir.name)
            if self.overwrite or not os.path.exists(out_path):
                rows = self._read_case_field(case_dir, "surface.vtk", include_geometry=True)
                if rows is not None:
                    if self.surface_sampling == "uniform":
                        rows = voxel_uniform_sample_rows(rows, self.surface_cache_points, rng)
                    else:
                        rows = sample_rows_np(rows, self.surface_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(np.float16))
            elif not valid_npy_cache(out_path, min_cols=14):
                rows = self._read_case_field(case_dir, "surface.vtk", include_geometry=True)
                if rows is not None:
                    if self.surface_sampling == "uniform":
                        rows = voxel_uniform_sample_rows(rows, self.surface_cache_points, rng)
                    else:
                        rows = sample_rows_np(rows, self.surface_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(np.float16))
            if valid_npy_cache(out_path, min_cols=14):
                written[kind] = out_path

        if self.build_volume:
            out_path = cache_path(self.cache_root, self.train_type, "volume_random", self.volume_cache_points, case_dir.name)
            if self.overwrite or not os.path.exists(out_path):
                rows = self._read_case_field(case_dir, "full_field.vtk", include_geometry=False)
                if rows is not None:
                    rows = sample_rows_np(rows, self.volume_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(np.float16))
            elif not valid_npy_cache(out_path, min_cols=10):
                rows = self._read_case_field(case_dir, "full_field.vtk", include_geometry=False)
                if rows is not None:
                    rows = sample_rows_np(rows, self.volume_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(np.float16))
            if valid_npy_cache(out_path, min_cols=10):
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
                    print(f"[{i + 1}/{len(self)}] processed {self.case_dirs[i].name}", flush=True)
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
                print(f"[{global_i + 1}/{len(self)}] processed {self.case_dirs[global_i].name}", flush=True)
