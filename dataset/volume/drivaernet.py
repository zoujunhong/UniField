from __future__ import annotations

import os

import numpy as np

from dataset.cache.drivaernet import DEFAULT_VOLUME_CACHE_POINTS, cache_path
from dataset.utils.normalization import (
    DEFAULT_U_REF,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VELOCITY_MEAN,
    VELOCITY_STD,
    normalize_drivaer_field,
)
from dataset.utils.sampling import sample_rows_np


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
        mmap: bool = False,
    ) -> None:
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
        self.mmap_mode = "r" if mmap else None

    def _load_rows(self, case_name: str) -> np.ndarray:
        path = cache_path(self.cache_root, "volume_random", self.volume_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing DrivAerNet++ volume cache: {path}. "
                "Run DrivAerNetPlusPlusCacheBuilder.build_cache() first."
            )
        return np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)

    def _field(self, rows: np.ndarray, name: str) -> np.ndarray:
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
        raise KeyError(f"DrivAerNet++ volume field '{name}' is not available.")

    def _pack(self, rows: np.ndarray, fields: list[str]) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field) for field in fields], axis=1).astype(np.float32, copy=False)

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del kwargs
        pool = self._load_rows(case_name)
        rows = sample_rows_np(pool, self.num_query_points, rng)
        return rows[:, :3].astype(np.float32, copy=True), self._pack(rows, self.volume_target_list)
