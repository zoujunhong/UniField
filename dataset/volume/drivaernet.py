from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dataset.cache.drivaernet import DEFAULT_VOLUME_CACHE_POINTS, cache_path, wall_distance_cache_path
from dataset.validation import pressure_valid_rows
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
        clip_cp: bool = False,
        cache_profile: str | None = None,
        allow_legacy_flat_cache: bool = False,
    ) -> None:
        if wall_distance_target_transform not in {None, "identity", "sqrt"}:
            raise ValueError("wall_distance_target_transform must be None, 'identity', or 'sqrt'.")
        self.cache_root = cache_root
        if cache_profile is not None and (not cache_profile or cache_profile in {".", ".."} or Path(cache_profile).name != cache_profile):
            raise ValueError("cache_profile must be a single directory name or None")
        self.cache_profile = cache_profile
        self.allow_legacy_flat_cache = bool(allow_legacy_flat_cache)
        self.num_query_points = int(num_query_points)
        self.volume_target_list = list(volume_target_list or [])
        self.volume_cache_points = int(volume_cache_points)
        self.normalization = normalization
        self.clip_cp = bool(clip_cp)
        self.u_ref = float(u_ref)
        self.pressure_mean = float(pressure_mean)
        self.pressure_std = float(pressure_std)
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.wall_distance_target_transform = wall_distance_target_transform
        self.mmap_mode = "r" if mmap else None

    def _case_directory(self, case_name: str) -> Path:
        directory = Path(self.cache_root) / case_name
        return directory / "pools" / self.cache_profile if self.cache_profile else directory

    def _load_rows(self, case_name: str) -> np.ndarray:
        canonical = self._case_directory(case_name) / "volume.npy"
        if canonical.exists():
            from dataset.field_cache import read_fields
            return read_fields(canonical, required=("xyz",), mmap_mode=self.mmap_mode)
        if not self.allow_legacy_flat_cache or self.cache_profile is not None:
            raise FileNotFoundError(f"Missing DrivAerNet volume cache: {canonical}")
        path = cache_path(self.cache_root, "volume_random", self.volume_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing explicit legacy DrivAerNet volume cache: {path}")
        rows = np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)
        if rows.ndim != 2 or rows.shape[1] not in (3, 6, 7, 8):
            raise ValueError(f"Invalid legacy volume column schema: {path}: {rows.shape}")
        return rows

    def coordinate_reference(self, case_name: str) -> np.ndarray:
        """Fallback bbox reference for volume-only DrivAerNet configurations."""
        return self._raw_field(self._load_rows(case_name), "xyz").copy()

    @staticmethod
    def _raw_field(rows: np.ndarray, name: str) -> np.ndarray:
        key = {"p": "pressure", "U": "velocity"}.get(name, name)
        if rows.dtype.names:
            if key not in rows.dtype.names:
                raise ValueError(f"DrivAerNet volume cache lacks required field {key!r}")
            values = np.asarray(rows[key], dtype=np.float32)
            return values[:, None] if values.ndim == 1 else values
        columns = {"xyz": (0, 3), "velocity": (3, 6), "pressure": (6, 7), "wall_distance": (7, 8)}
        if key not in columns:
            raise KeyError(f"DrivAerNet volume field {key!r} is not available")
        begin, end = columns[key]
        if rows.ndim != 2 or rows.shape[1] < end:
            raise ValueError(f"DrivAerNet volume cache lacks required field {key!r}")
        return np.asarray(rows[:, begin:end], dtype=np.float32)

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
            return self._raw_field(rows, "xyz")
        if name == "p":
            return normalize_drivaer_field("p", self._raw_field(rows, "p"), self.normalization, u_ref=self.u_ref,
                pressure_mean=self.pressure_mean, pressure_std=self.pressure_std, clip_cp=self.clip_cp)
        if name == "U":
            return normalize_drivaer_field(
                "U",
                self._raw_field(rows, "U"),
                self.normalization,
                clip_cp=self.clip_cp,
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
        wall_distance_pool = None
        if self._needs_wall_distance():
            has_distance = "wall_distance" in names if (names := pool.dtype.names) else pool.shape[1] >= 8
            if has_distance:
                wall_distance_pool = self._raw_field(pool, "wall_distance")
            elif self.allow_legacy_flat_cache and self.cache_profile is None and not pool.dtype.names:
                wall_distance_pool = self._load_wall_distance_rows(case_name)
            else:
                raise ValueError("DrivAerNet volume cache lacks required field 'wall_distance'")
            if len(wall_distance_pool) != len(pool):
                raise ValueError(f"{case_name}: wall-distance cache row count differs from volume cache")
        if getattr(self, "validation_cp_min", None) is not None:
            cp = self._raw_field(pool, "p")[:, 0] / (0.5 * self.u_ref ** 2)
            # Filter paired fields together, including any explicit legacy sidecar.
            keep = pressure_valid_rows(self, np.arange(len(pool)), cp)
            pool = pool[keep]
            if wall_distance_pool is not None:
                wall_distance_pool = wall_distance_pool[keep]
        indices = sample_row_indices_np(len(pool), self.num_query_points, rng)
        rows = pool[indices]
        wall_distance_rows = None if wall_distance_pool is None else wall_distance_pool[indices]
        return self._raw_field(rows, "xyz").copy(), self._pack(rows, self.volume_target_list, wall_distance_rows)
