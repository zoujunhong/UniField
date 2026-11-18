"""Cached surface sampler shared by the paper's aerodynamic datasets."""

from __future__ import annotations

from dataset.validation import pressure_valid_rows

from pathlib import Path

import numpy as np

from dataset.utils.aerodynamic_surface import (
    SURFACE_CACHE_COLUMNS,
    ensure_surface_input_fields,
    find_surface_cache_path,
    validate_surface_fields,
)
from dataset.utils.sampling import sample_rows_np, voxel_uniform_sample_rows


class AerodynamicSurfaceDataset:
    """Read ``xyz, Cp, normal, area`` arrays written by the cache builders."""

    def __init__(
        self,
        *,
        cache_root: str | Path,
        dataset_name: str,
        num_surface_points: int = 32_768,
        num_surface_query_points: int | None = None,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = ("p",),
        surface_sampling: str = "random",
        normalization: str = "physical",
        pressure_mean: float = 0.0,
        pressure_std: float = 1.0,
        mmap: bool = False,
    ) -> None:
        if surface_sampling not in {"random", "uniform", "area_random"}:
            raise ValueError("surface_sampling must be 'random', 'uniform', or 'area_random'.")
        if normalization not in {"physical", "none", "standard"}:
            raise ValueError("normalization must be 'physical', 'none', or 'standard'.")
        if float(pressure_std) <= 0.0:
            raise ValueError("pressure_std must be positive.")
        self.cache_root = Path(cache_root)
        self.dataset_name = str(dataset_name)
        self.num_surface_points = int(num_surface_points)
        # ``None`` has the same contract as the DrivAerNet and high-speed-train
        # surface readers: use the sampled encoder points as pressure queries.
        # Keeping the sentinel avoids an unnecessary O(N^2) decoder KNN path.
        self.num_surface_query_points = None if num_surface_query_points is None else int(num_surface_query_points)
        self.surface_input_list = ensure_surface_input_fields(surface_input_list)
        self.surface_target_list = list(surface_target_list or [])
        validate_surface_fields(self.surface_target_list)
        self.surface_sampling = surface_sampling
        self.normalization = normalization
        self.pressure_mean = float(pressure_mean)
        self.pressure_std = float(pressure_std)
        self.mmap_mode = "r" if mmap else None

    def cache_path(self, case_name: str) -> Path:
        return find_surface_cache_path(self.cache_root, self.dataset_name, case_name)

    def has_cache(self, case_name: str) -> bool:
        path = self.cache_path(case_name)
        if not path.is_file():
            return False
        try:
            rows = np.load(path, mmap_mode="r")
        except Exception:
            return False
        # Some legacy OpenFOAM surface caches contain only ``xyz, Cp``.  They
        # are still valid point-cloud supervision; normal and area requests
        # receive the documented zero/unit fallbacks below.
        return rows.ndim == 2 and rows.shape[0] > 0 and rows.shape[1] >= 4

    def coordinate_reference(self, case_name: str) -> np.ndarray:
        """Return the complete cached surface for a sample-independent bbox frame."""

        return self._load_rows(case_name)[:, 0:3].astype(np.float32, copy=True)

    def _load_rows(self, case_name: str) -> np.ndarray:
        path = self.cache_path(case_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {self.dataset_name} surface cache: {path}. "
                "Build it with `python -m dataset.cache.aerodynamic_surface --dataset ...`."
            )
        rows = np.load(path, mmap_mode=self.mmap_mode)
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 4:
            raise ValueError(
                f"Invalid {self.dataset_name} surface cache {path}: expected a non-empty "
                f"(N, >= 4) array containing xyz and Cp; got {rows.shape}."
            )
        return np.asarray(rows, dtype=np.float32)

    def _sample_rows(self, rows: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.surface_sampling == "uniform":
            return voxel_uniform_sample_rows(rows, count, rng)
        if self.surface_sampling == "area_random":
            if rows.shape[1] < 8:
                raise ValueError("area_random sampling requires xyz, Cp, normal, area cache rows.")
            count = int(count)
            if count < 0:
                raise ValueError("Surface sample count must be non-negative.")
            if count == 0:
                return np.asarray(rows[:0]).copy()
            area = np.asarray(rows[:, 7], dtype=np.float64)
            if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
                raise ValueError("area_random sampling requires finite positive face areas.")
            total_area = float(area.sum())
            indices = rng.choice(rows.shape[0], size=count, replace=True, p=area / total_area)
            sampled = np.asarray(rows[indices], dtype=np.float32).copy()
            # Horvitz-style equal weights make the area-proportional Monte Carlo
            # force integral unbiased, including when the cache has <8192 faces.
            sampled[:, 7] = total_area / count
            return sampled
        return sample_rows_np(rows, count, rng)

    def _field(self, rows: np.ndarray, name: str) -> np.ndarray:
        if name == "xyz":
            return rows[:, 0:3]
        if name == "p":
            cp = rows[:, 3:4].astype(np.float32, copy=True)
            if self.normalization == "standard":
                cp = (cp - self.pressure_mean) / self.pressure_std
            return cp
        if name in {"normal", "normals"}:
            if rows.shape[1] >= 7:
                return rows[:, 4:7]
            return np.zeros((rows.shape[0], 3), dtype=np.float32)
        if name == "area":
            if rows.shape[1] >= 8:
                return np.maximum(rows[:, 7:8], 1e-12)
            return np.ones((rows.shape[0], 1), dtype=np.float32)
        if name == "log_area":
            return np.log(np.maximum(self._field(rows, "area"), 1e-12)).astype(np.float32)
        raise KeyError(f"Unsupported aerodynamic surface field {name!r}.")

    def _pack(self, rows: np.ndarray, fields: list[str]) -> np.ndarray:
        if not fields:
            return np.zeros((rows.shape[0], 0), dtype=np.float32)
        return np.concatenate([self._field(rows, field) for field in fields], axis=1).astype(np.float32, copy=False)

    def sample(self, case_name: str, rng: np.random.Generator):
        pool = self._load_rows(case_name)
        input_rows = self._sample_rows(pool, self.num_surface_points, rng)
        query_pool = pressure_valid_rows(self, pool, pool[:, 3])
        query_rows = input_rows if self.num_surface_query_points is None else self._sample_rows(
            query_pool,
            self.num_surface_query_points,
            rng,
        )
        return (
            self._pack(input_rows, self.surface_input_list),
            query_rows[:, 0:3].astype(np.float32, copy=True),
            self._pack(query_rows, self.surface_target_list),
        )


__all__ = ["AerodynamicSurfaceDataset"]
