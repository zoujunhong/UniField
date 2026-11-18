from __future__ import annotations

import os

import numpy as np

from dataset.cache.high_speed_train import (
    DEFAULT_CACHE_DTYPE,
    DEFAULT_VOLUME_CACHE_POINTS,
    HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL,
    cache_path,
    normalized_cache_dtype,
    normalized_train_type,
)
from dataset.utils.load import parse_high_speed_case_cond
from dataset.utils.normalization import normalize_high_speed_field
from dataset.utils.sampling import sample_rows_np
from dataset.validation import pressure_valid_rows


class HighSpeedTrainVolumeDataset:
    def __init__(
        self,
        cache_root: str,
        train_type: str = "CRH450",
        num_query_points: int = 8_192,
        volume_target_list: list[str] | tuple[str, ...] | None = ("U",),
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        cache_dtype: str = DEFAULT_CACHE_DTYPE,
        normalization: str = "physical",
        standard_stats=None,
        clamp_cp: tuple[float, float] | None = None,
        query_surface_distance: bool = False,
        query_surface_distance_clip: float | None = 1.0,
        surface_distance_target_transform: str | None = None,
        mmap: bool = False,
    ) -> None:
        if surface_distance_target_transform not in {None, "identity", "sqrt"}:
            raise ValueError("surface_distance_target_transform must be None, 'identity', or 'sqrt'.")
        self.cache_root = cache_root
        self.train_type = normalized_train_type(train_type)
        self.num_query_points = int(num_query_points)
        self.volume_target_list = list(volume_target_list or [])
        self.volume_cache_points = int(volume_cache_points)
        self.cache_dtype = normalized_cache_dtype(cache_dtype)
        self.normalization = normalization
        self.standard_stats = standard_stats
        self.clamp_cp = clamp_cp
        self.query_surface_distance = bool(query_surface_distance)
        self.query_surface_distance_clip = None if query_surface_distance_clip is None else float(query_surface_distance_clip)
        self.surface_distance_target_transform = surface_distance_target_transform
        self.mmap_mode = "r" if mmap else None

    def _load_rows(self, case_name: str) -> np.ndarray:
        path = cache_path(self.cache_root, self.train_type, "volume_random", self.volume_cache_points, case_name, self.cache_dtype)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing HighSpeedTrain volume cache: {path}. "
                "Run HighSpeedTrainCacheBuilder.build_cache() first."
            )
        return np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)

    def coordinate_reference(self, case_name: str) -> np.ndarray:
        """Fallback reference for volume-only configurations without a surface cache."""

        return self._load_rows(case_name)[:, :3].astype(np.float32, copy=True)

    def _field(self, rows: np.ndarray, name: str, case_name: str) -> np.ndarray:
        cond, speed = parse_high_speed_case_cond(case_name)
        del cond
        if name == "xyz":
            return rows[:, :3]
        if name == "p":
            return normalize_high_speed_field("p", rows[:, 3:4], self.normalization, speed, self.clamp_cp, self.standard_stats)
        if name == "U":
            return normalize_high_speed_field("U", rows[:, 4:7], self.normalization, speed, self.clamp_cp, self.standard_stats)
        if name == "k":
            return normalize_high_speed_field("k", rows[:, 7:8], self.normalization, speed, self.clamp_cp, self.standard_stats)
        if name == "omega":
            return normalize_high_speed_field("omega", rows[:, 8:9], self.normalization, speed, self.clamp_cp, self.standard_stats)
        if name == "nut":
            return normalize_high_speed_field("nut", rows[:, 9:10], self.normalization, speed, self.clamp_cp, self.standard_stats)
        if name in {"surface_distance", "wall_distance", "sdf"}:
            if rows.shape[1] <= HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL:
                raise KeyError(
                    "HighSpeedTrain volume cache does not include surface_distance. "
                    "Run append_surface_distance_to_volume_cache first."
                )
            distance = rows[:, HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL : HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL + 1]
            if self.surface_distance_target_transform == "sqrt":
                return np.sqrt(np.maximum(distance, 0.0)).astype(np.float32, copy=False)
            return distance
        raise KeyError(f"HighSpeedTrain volume field '{name}' is not available.")

    def _pack(self, rows: np.ndarray, fields: list[str], case_name: str) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field, case_name) for field in fields], axis=1).astype(
            np.float32,
            copy=False,
        )

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del kwargs
        pool = self._load_rows(case_name)
        pool = pressure_valid_rows(self, pool, pool[:, 3] / (0.5 * parse_high_speed_case_cond(case_name)[1] ** 2))
        rows = sample_rows_np(pool, self.num_query_points, rng)
        query = rows[:, :3].astype(np.float32, copy=True)
        if self.query_surface_distance:
            if rows.shape[1] <= HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL:
                raise ValueError(
                    "HighSpeedTrain volume cache does not include surface_distance. "
                    f"Expected column {HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL} in {self.train_type}/{case_name}."
                )
            distance = rows[:, HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL : HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL + 1].astype(
                np.float32,
                copy=True,
            )
            if self.query_surface_distance_clip is not None:
                distance = np.minimum(distance, float(self.query_surface_distance_clip)).astype(np.float32, copy=False)
            query = np.concatenate([query, distance], axis=1).astype(np.float32, copy=False)
        return query, self._pack(rows, self.volume_target_list, case_name)
