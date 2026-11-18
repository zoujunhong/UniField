from __future__ import annotations

import os
import warnings
from pathlib import Path

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
from dataset.validation import pressure_valid_rows


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
        clip_cp: bool = False,
        cache_profile: str | None = None,
        allow_legacy_flat_cache: bool = False,
        return_query_normals: bool = False,
    ) -> None:
        if surface_sampling not in {"random", "uniform", "mixed"}:
            raise ValueError("surface_sampling must be 'random', 'uniform', or 'mixed'.")
        self.cache_root = cache_root
        if cache_profile is not None and (not cache_profile or cache_profile in {".", ".."} or Path(cache_profile).name != cache_profile):
            raise ValueError("cache_profile must be a single directory name or None")
        self.cache_profile = cache_profile
        self.allow_legacy_flat_cache = bool(allow_legacy_flat_cache)
        self.return_query_normals = bool(return_query_normals)
        self._warned_unmatched_validity = False
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = None if num_surface_query_points is None else int(num_surface_query_points)
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal"))
        self.surface_target_list = list(surface_target_list or [])
        self.surface_sampling = surface_sampling
        self.surface_cache_points = int(surface_cache_points)
        self.normalization = normalization
        self.clip_cp = bool(clip_cp)
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

    def _case_directory(self, case_name: str) -> Path:
        directory = Path(self.cache_root) / case_name
        return directory / "pools" / self.cache_profile if self.cache_profile else directory

    def _apply_validity(self, case_name: str, rows: np.ndarray, directory: Path) -> np.ndarray:
        validity = getattr(self, "validity", None)
        if not validity:
            return rows
        from dataset.field_cache import case_metadata
        metadata = case_metadata(directory) if (directory / "metadata.json").is_file() else {}
        compact = metadata.get("surface", {}).get("row_order") == "compact_surface65536_v1"
        if not compact and self.cache_profile is None and not metadata:
            manifest_path = Path(self.cache_root) / "manifest.json"
            if manifest_path.is_file():
                import json
                compact = json.loads(manifest_path.read_text()).get("schema") == "standalone_fields_v2"
        matching = validity if compact else validity.get("profiles", {}).get(self.cache_profile, {})
        if not matching:
            if not self._warned_unmatched_validity:
                warnings.warn(
                    f"DrivAerNet cache profile {self.cache_profile!r} has no matching row-order validity mask; "
                    "compact field_cache_rows are not applied.", RuntimeWarning, stacklevel=2)
                self._warned_unmatched_validity = True
            return rows
        excluded = matching.get("excluded", {}).get(case_name, {}).get("field_cache_rows", [])
        if excluded:
            indices = np.asarray(excluded, dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= len(rows)):
                raise ValueError(f"{case_name}: validity indices lie outside cache profile {self.cache_profile!r}")
            keep = np.ones(len(rows), dtype=bool)
            keep[indices] = False
            rows = rows[keep]
        return rows

    def _load_rows(self, case_name: str, rng: np.random.Generator) -> np.ndarray:
        directory = self._case_directory(case_name)
        canonical = directory / "surface.npy"
        if canonical.exists():
            from dataset.field_cache import read_fields
            rows = read_fields(canonical, required=("xyz",), mmap_mode=self.mmap_mode)
            return self._apply_validity(case_name, rows, directory)
        if not self.allow_legacy_flat_cache or self.cache_profile is not None:
            raise FileNotFoundError(f"Missing DrivAerNet surface cache: {canonical}")
        mode = self._choose_surface_mode(rng)
        path = cache_path(self.cache_root, f"surface_{mode}", self.surface_cache_points, case_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing explicit legacy DrivAerNet surface cache: {path}")
        rows = np.load(path, mmap_mode=self.mmap_mode).astype(np.float32, copy=False)
        if rows.ndim != 2 or rows.shape[1] not in (3, 4, 7, 8):
            raise ValueError(f"Invalid legacy surface column schema: {path}: {rows.shape}")
        return rows

    def coordinate_reference(self, case_name: str) -> np.ndarray:
        """Return a cache-pool bbox reference independent of point sampling."""
        return self._raw_field(self._load_rows(case_name, np.random.default_rng(0)), "xyz").copy()

    @staticmethod
    def _raw_field(rows: np.ndarray, name: str) -> np.ndarray:
        key = {"p": "pressure", "normals": "normal"}.get(name, name)
        if rows.dtype.names:
            if key not in rows.dtype.names:
                raise ValueError(f"DrivAerNet surface cache lacks required field {key!r}")
            values = np.asarray(rows[key], dtype=np.float32)
            return values[:, None] if values.ndim == 1 else values
        columns = {"xyz": (0, 3), "pressure": (3, 4), "normal": (4, 7), "area": (7, 8)}
        if key not in columns:
            raise KeyError(f"DrivAerNet surface field {key!r} is not available")
        begin, end = columns[key]
        if rows.ndim != 2 or rows.shape[1] < end:
            raise ValueError(f"DrivAerNet surface cache lacks required field {key!r}")
        return np.asarray(rows[:, begin:end], dtype=np.float32)

    def _field(self, rows: np.ndarray, name: str) -> np.ndarray:
        if name == "xyz":
            return self._raw_field(rows, "xyz")
        if name == "p":
            values = self._raw_field(rows, "p")
            return normalize_drivaer_field(
                "p",
                values,
                self.normalization,
                clip_cp=self.clip_cp,
                u_ref=self.u_ref,
                pressure_mean=self.pressure_mean,
                pressure_std=self.pressure_std,
                velocity_mean=self.velocity_mean,
                velocity_std=self.velocity_std,
            )
        if name in {"normal", "normals"}:
            return self._raw_field(rows, "normal")
        if name == "area":
            return np.maximum(self._raw_field(rows, "area"), 1e-12)
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
        query_pool = pool
        count = self.num_surface_points if self.num_surface_query_points is None else self.num_surface_query_points
        if count and getattr(self, "validation_cp_min", None) is not None:
            query_pool = pressure_valid_rows(self, pool, self._raw_field(pool, "p")[:, 0] / (0.5 * self.u_ref ** 2))
        if self.num_surface_query_points is None and getattr(self, "validation_cp_min", None) is None:
            query_rows = surface_rows
        else:
            query_rows = sample_rows_np(query_pool, count, rng)
        has_normal = "normal" in rows_names if (rows_names := query_rows.dtype.names) else query_rows.shape[1] >= 7
        self.query_normals = (np.empty((0, 3), np.float32) if not len(query_rows) else
                              self._raw_field(query_rows, "normal").copy() if has_normal or self.return_query_normals else None)
        surface_input = self._pack(surface_rows, self.surface_input_list)
        surface_query = self._raw_field(query_rows, "xyz").copy()
        surface_target = self._pack(query_rows, self.surface_target_list)
        return surface_input, surface_query, surface_target
