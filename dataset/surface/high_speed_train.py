from __future__ import annotations

import os

import numpy as np

from dataset.cache.high_speed_train import DEFAULT_SURFACE_CACHE_POINTS, cache_path, normalized_train_type
from dataset.utils.load import ensure_xyz_prefix, high_speed_free_stream, parse_high_speed_case_cond
from dataset.utils.normalization import normalize_high_speed_field
from dataset.utils.sampling import sample_rows_np


class HighSpeedTrainSurfaceDataset:
    def __init__(
        self,
        cache_root: str,
        train_type: str = "CRH450",
        num_surface_points: int = 32_768,
        num_surface_query_points: int | None = None,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = ("p",),
        surface_sampling: str = "random",
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        normalization: str = "physical",
        clamp_cp: tuple[float, float] | None = (-5.0, 1.05),
        mmap: bool = False,
    ) -> None:
        if surface_sampling not in {"random", "uniform", "mixed"}:
            raise ValueError("surface_sampling must be 'random', 'uniform', or 'mixed'.")
        self.cache_root = cache_root
        self.train_type = normalized_train_type(train_type)
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = None if num_surface_query_points is None else int(num_surface_query_points)
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal", "area"))
        self.surface_target_list = list(surface_target_list or [])
        self.surface_sampling = surface_sampling
        self.surface_cache_points = int(surface_cache_points)
        self.normalization = normalization
        self.clamp_cp = clamp_cp
        self.mmap_mode = "r" if mmap else None

    def _choose_surface_mode(self, rng: np.random.Generator) -> str:
        if self.surface_sampling == "mixed":
            return "uniform" if rng.random() < 0.5 else "random"
        return self.surface_sampling

    def _load_rows(self, case_name: str, rng: np.random.Generator) -> np.ndarray:
        mode = self._choose_surface_mode(rng)
        path = cache_path(self.cache_root, self.train_type, f"surface_{mode}", self.surface_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing HighSpeedTrain surface cache: {path}. "
                "Run HighSpeedTrainCacheBuilder.build_cache() first."
            )
        return np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)

    def _field(self, rows: np.ndarray, name: str, case_name: str) -> np.ndarray:
        cond, speed = parse_high_speed_case_cond(case_name)
        del cond
        free_stream = high_speed_free_stream(case_name)
        if name == "xyz":
            return rows[:, :3]
        if name == "p":
            return normalize_high_speed_field("p", rows[:, 3:4], self.normalization, free_stream, speed, self.clamp_cp)
        if name == "U":
            return normalize_high_speed_field("U", rows[:, 4:7], self.normalization, free_stream, speed, self.clamp_cp)
        if name == "k":
            return normalize_high_speed_field("k", rows[:, 7:8], self.normalization, free_stream, speed, self.clamp_cp)
        if name == "omega":
            return normalize_high_speed_field("omega", rows[:, 8:9], self.normalization, free_stream, speed, self.clamp_cp)
        if name == "nut":
            return normalize_high_speed_field("nut", rows[:, 9:10], self.normalization, free_stream, speed, self.clamp_cp)
        if name in {"normal", "normals"}:
            if rows.shape[1] >= 13:
                return rows[:, 10:13]
            return np.zeros((rows.shape[0], 3), dtype=np.float32)
        if name == "area":
            if rows.shape[1] >= 14:
                return np.maximum(rows[:, 13:14], 1e-12)
            return np.ones((rows.shape[0], 1), dtype=np.float32)
        if name == "log_area":
            area = self._field(rows, "area", case_name)
            return np.log(np.maximum(area, 1e-12)).astype(np.float32)
        raise KeyError(f"HighSpeedTrain surface field '{name}' is not available.")

    def _pack(self, rows: np.ndarray, fields: list[str], case_name: str) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field, case_name) for field in fields], axis=1).astype(
            np.float32,
            copy=False,
        )

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del kwargs
        pool = self._load_rows(case_name, rng)
        surface_rows = sample_rows_np(pool, self.num_surface_points, rng)
        if self.num_surface_query_points is None:
            query_rows = surface_rows
        else:
            query_rows = sample_rows_np(pool, self.num_surface_query_points, rng)
        surface_input = self._pack(surface_rows, self.surface_input_list, case_name)
        surface_query = query_rows[:, :3].astype(np.float32, copy=True)
        surface_target = self._pack(query_rows, self.surface_target_list, case_name)
        return surface_input, surface_query, surface_target
