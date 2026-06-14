from __future__ import annotations

import os

import numpy as np

from dataset.cache.high_speed_train import DEFAULT_VOLUME_CACHE_POINTS, cache_path, normalized_train_type
from dataset.utils.load import high_speed_free_stream, parse_high_speed_case_cond
from dataset.utils.normalization import normalize_high_speed_field
from dataset.utils.sampling import sample_rows_np


class HighSpeedTrainVolumeDataset:
    def __init__(
        self,
        cache_root: str,
        train_type: str = "CRH450",
        num_query_points: int = 8_192,
        volume_target_list: list[str] | tuple[str, ...] | None = ("U",),
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        normalization: str = "physical",
        clamp_cp: tuple[float, float] | None = (-5.0, 1.05),
        mmap: bool = False,
    ) -> None:
        self.cache_root = cache_root
        self.train_type = normalized_train_type(train_type)
        self.num_query_points = int(num_query_points)
        self.volume_target_list = list(volume_target_list or [])
        self.volume_cache_points = int(volume_cache_points)
        self.normalization = normalization
        self.clamp_cp = clamp_cp
        self.mmap_mode = "r" if mmap else None

    def _load_rows(self, case_name: str) -> np.ndarray:
        path = cache_path(self.cache_root, self.train_type, "volume_random", self.volume_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing HighSpeedTrain volume cache: {path}. "
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
        rows = sample_rows_np(pool, self.num_query_points, rng)
        return rows[:, :3].astype(np.float32, copy=True), self._pack(rows, self.volume_target_list, case_name)
