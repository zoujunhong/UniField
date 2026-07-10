from __future__ import annotations

import os

import numpy as np

from dataset.cache.drivaernet import DEFAULT_VOLUME_CACHE_POINTS, cache_path, wall_distance_cache_path
from dataset.utils.normalization import (
    DEFAULT_U_REF,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VELOCITY_MEAN,
    VELOCITY_STD,
    normalize_drivaer_field,
)


WALL_DISTANCE_FIELDS = {"surface_distance", "wall_distance", "sdf"}


def sample_row_indices_np(n: int, num: int, rng: np.random.Generator) -> np.ndarray:
    num = int(num)
    if num < 0:
        raise ValueError("num must be non-negative.")
    if n <= 0:
        raise ValueError("Cannot sample from an empty array.")
    if num == 0:
        return np.empty((0,), dtype=np.int64)
    if n == num:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=num, replace=n < num).astype(np.int64, copy=False)


class DrivAerNetVolumeDataset:
    def __init__(
        self,
        cache_root: str,
        num_query_points: int = 8_192,
        volume_target_list: list[str] | tuple[str, ...] | None = ("U",),
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        normalization: str = "physical",
        u_ref: float = DEFAULT_U_REF,
        pressure_mean: float = PRESSURE_MEAN,
        pressure_std: float = PRESSURE_STD,
        velocity_mean=VELOCITY_MEAN,
        velocity_std=VELOCITY_STD,
        wall_distance_target_transform: str | None = None,
        mmap: bool = False,
    ) -> None:
        if wall_distance_target_transform not in {None, "identity", "sqrt"}:
            raise ValueError("wall_distance_target_transform must be None, 'identity', or 'sqrt'.")
        self.cache_root = cache_root
        self.num_query_points = int(num_query_points)
        self.volume_target_list = list(volume_target_list or [])
        self.volume_cache_points = int(volume_cache_points)
        self.normalization = normalization
        self.u_ref = float(u_ref)
        self.pressure_mean = float(pressure_mean)
        self.pressure_std = float(pressure_std)
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.wall_distance_target_transform = wall_distance_target_transform
        self.mmap_mode = "r" if mmap else None

    def _load_rows(self, case_name: str) -> np.ndarray:
        path = cache_path(self.cache_root, "volume_random", self.volume_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing DrivAerNet++ volume cache: {path}. "
                "Run DrivAerNetPlusPlusCacheBuilder.build_cache() first."
            )
        return np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)

    def _needs_wall_distance(self) -> bool:
        return any(field in WALL_DISTANCE_FIELDS for field in self.volume_target_list)

    def _load_wall_distance_rows(self, case_name: str) -> np.ndarray:
        path = wall_distance_cache_path(self.cache_root, self.volume_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing DrivAerNet++ wall-distance cache: {path}. "
                "Run DrivAerNetPlusPlusWallDistanceCacheBuilder.build_cache() first."
            )
        rows = np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        if rows.ndim != 2 or rows.shape[1] < 1:
            raise ValueError(f"Invalid DrivAerNet++ wall-distance cache shape for {case_name}: {rows.shape}.")
        return rows

    def _field(self, rows: np.ndarray, name: str, wall_distance_rows: np.ndarray | None = None) -> np.ndarray:
        if name == "xyz":
            return rows[:, :3]
        if name == "U":
            return normalize_drivaer_field(
                "U",
                rows[:, 3:6],
                self.normalization,
                u_ref=self.u_ref,
                pressure_mean=self.pressure_mean,
                pressure_std=self.pressure_std,
                velocity_mean=self.velocity_mean,
                velocity_std=self.velocity_std,
            )
        if name in WALL_DISTANCE_FIELDS:
            if wall_distance_rows is None:
                raise KeyError(
                    "DrivAerNet++ wall_distance target requested but no wall-distance rows were loaded."
                )
            distance = wall_distance_rows[:, :1]
            if self.wall_distance_target_transform == "sqrt":
                return np.sqrt(np.maximum(distance, 0.0)).astype(np.float32, copy=False)
            return distance
        raise KeyError(f"DrivAerNet++ volume field '{name}' is not available.")

    def _pack(
        self,
        rows: np.ndarray,
        fields: list[str],
        wall_distance_rows: np.ndarray | None = None,
    ) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field, wall_distance_rows) for field in fields], axis=1).astype(
            np.float32,
            copy=False,
        )

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del kwargs
        pool = self._load_rows(case_name)
        indices = sample_row_indices_np(pool.shape[0], self.num_query_points, rng)
        rows = np.asarray(pool[indices]).astype(np.float32, copy=False)
        wall_distance_rows = None
        if self._needs_wall_distance():
            wall_distance_pool = self._load_wall_distance_rows(case_name)
            if wall_distance_pool.shape[0] != pool.shape[0]:
                raise ValueError(
                    f"DrivAerNet++ wall-distance cache row count mismatch for {case_name}: "
                    f"{wall_distance_pool.shape[0]} != {pool.shape[0]}."
                )
            wall_distance_rows = np.asarray(wall_distance_pool[indices]).astype(np.float32, copy=False)
        return rows[:, :3].astype(np.float32, copy=True), self._pack(rows, self.volume_target_list, wall_distance_rows)
