from __future__ import annotations

import os

import numpy as np

from dataset.cache.drivaernet import DEFAULT_SURFACE_CACHE_POINTS, cache_path
from dataset.utils.load import ensure_xyz_prefix
from dataset.utils.normalization import (
    DEFAULT_U_REF,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VELOCITY_MEAN,
    VELOCITY_STD,
    normalize_drivaer_field,
)
from dataset.utils.sampling import sample_rows_np


class DrivAerNetSurfaceDataset:
    def __init__(
        self,
        cache_root: str,
        num_surface_points: int = 32_768,
        num_surface_query_points: int | None = None,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = ("p",),
        surface_sampling: str = "random",
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        normalization: str = "physical",
        u_ref: float = DEFAULT_U_REF,
        pressure_mean: float = PRESSURE_MEAN,
        pressure_std: float = PRESSURE_STD,
        velocity_mean=VELOCITY_MEAN,
        velocity_std=VELOCITY_STD,
        mmap: bool = False,
    ) -> None:
        if surface_sampling not in {"random", "uniform", "mixed"}:
            raise ValueError("surface_sampling must be 'random', 'uniform', or 'mixed'.")
        self.cache_root = cache_root
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = None if num_surface_query_points is None else int(num_surface_query_points)
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal", "area"))
        self.surface_target_list = list(surface_target_list or [])
        self.surface_sampling = surface_sampling
        self.surface_cache_points = int(surface_cache_points)
        self.normalization = normalization
        self.u_ref = float(u_ref)
        self.pressure_mean = float(pressure_mean)
        self.pressure_std = float(pressure_std)
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.mmap_mode = "r" if mmap else None

    def _choose_surface_mode(self, rng: np.random.Generator) -> str:
        if self.surface_sampling == "mixed":
            return "uniform" if rng.random() < 0.5 else "random"
        return self.surface_sampling

    def _load_rows(self, case_name: str, rng: np.random.Generator) -> np.ndarray:
        mode = self._choose_surface_mode(rng)
        path = cache_path(self.cache_root, f"surface_{mode}", self.surface_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing DrivAerNet++ surface cache: {path}. "
                "Run DrivAerNetPlusPlusCacheBuilder.build_cache() first."
            )
        return np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)

    def _field(self, rows: np.ndarray, name: str) -> np.ndarray:
        if name == "xyz":
            return rows[:, :3]
        if name == "p":
            values = rows[:, 3:4]
            return normalize_drivaer_field(
                "p",
                values,
                self.normalization,
                u_ref=self.u_ref,
                pressure_mean=self.pressure_mean,
                pressure_std=self.pressure_std,
                velocity_mean=self.velocity_mean,
                velocity_std=self.velocity_std,
            )
        if name in {"normal", "normals"}:
            if rows.shape[1] >= 7:
                return rows[:, 4:7]
            return np.zeros((rows.shape[0], 3), dtype=np.float32)
        if name == "area":
            if rows.shape[1] >= 8:
                return np.maximum(rows[:, 7:8], 1e-12)
            return np.ones((rows.shape[0], 1), dtype=np.float32)
        if name == "log_area":
            area = self._field(rows, "area")
            return np.log(np.maximum(area, 1e-12)).astype(np.float32)
        raise KeyError(f"DrivAerNet++ surface field '{name}' is not available.")

    def _pack(self, rows: np.ndarray, fields: list[str]) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field) for field in fields], axis=1).astype(np.float32, copy=False)

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del kwargs
        pool = self._load_rows(case_name, rng)
        surface_rows = sample_rows_np(pool, self.num_surface_points, rng)
        if self.num_surface_query_points is None:
            query_rows = surface_rows
        else:
            query_rows = sample_rows_np(pool, self.num_surface_query_points, rng)
        surface_input = self._pack(surface_rows, self.surface_input_list)
        surface_query = query_rows[:, :3].astype(np.float32, copy=True)
        surface_target = self._pack(query_rows, self.surface_target_list)
        return surface_input, surface_query, surface_target
