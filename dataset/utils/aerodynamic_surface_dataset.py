"""Composable surface-only PyTorch dataset for the paper's legacy datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.surface.aerodynamic import AerodynamicSurfaceDataset
from dataset.utils.aerodynamic_surface import (
    AerodynamicSurfaceCase,
    SURFACE_FIELD_DIMS,
    ensure_surface_input_fields,
    filter_cases_by_ids,
    select_cases_by_geometry,
    validate_surface_fields,
)
from dataset.utils.coordinate_frame import CanonicalCoordinateFrame, transform_packed_surface_input
from dataset.utils.sampling import rng_for_index
from dataset.volume.aerodynamic import EmptyAerodynamicVolumeDataset


class AerodynamicSurfaceOnlyDataset(Dataset):
    """Return the repository's standard eight-item batch tuple for surface data."""

    dataset_name = "aerodynamic_surface"
    supported_coefficient_fields: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        cases: Sequence[AerodynamicSurfaceCase],
        cache_root: str | Path,
        cache_dataset_name: str | None = None,
        use_surface: bool = True,
        use_volume: bool = False,
        num_surface_points: int = 32_768,
        num_surface_query_points: int | None = None,
        num_query_points: int = 0,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = ("p",),
        volume_target_list: list[str] | tuple[str, ...] | None = None,
        coefficient_target_list: list[str] | tuple[str, ...] | None = None,
        surface_sampling: str = "random",
        ids_file: str | None = None,
        split: str = "all",
        val_ratio: float = 0.1,
        skip_incomplete: bool = True,
        allow_empty: bool = False,
        repeat: int = 1,
        normalization: str = "physical",
        pressure_mean: float = 0.0,
        pressure_std: float = 1.0,
        center_x: bool = True,
        center_y: bool = True,
        ground_z: bool = True,
        rotate_geometry_by_aoa: bool = False,
        route: int = 0,
        mmap: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        placement_override: float | None = None,
    ) -> None:
        if not use_surface:
            raise ValueError(f"{self.dataset_name} only provides surface-pressure labels; use_surface must be True.")
        if use_volume:
            raise ValueError(f"{self.dataset_name} has no volume labels; use_volume must be False.")
        volume_target_list = list(volume_target_list or [])
        if volume_target_list:
            raise ValueError(f"{self.dataset_name} has no volume targets, got {volume_target_list}.")
        if int(num_query_points) not in {0}:
            raise ValueError(f"{self.dataset_name} has no volume queries; num_query_points must be 0.")
        if int(repeat) <= 0:
            raise ValueError("repeat must be positive.")

        self.cache_root = Path(cache_root)
        self.cache_dataset_name = str(cache_dataset_name or self.dataset_name)
        self.repeat = int(repeat)
        self.route = int(route)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.center_x = bool(center_x)
        self.center_y = bool(center_y)
        self.ground_z = bool(ground_z)
        self.rotate_geometry_by_aoa = bool(rotate_geometry_by_aoa)
        self.surface_input_list = ensure_surface_input_fields(surface_input_list)
        self.surface_target_list = list(surface_target_list or [])
        validate_surface_fields(self.surface_target_list)
        self.volume_target_list = []
        self.coefficient_target_list = [str(field).upper() for field in (coefficient_target_list or [])]
        unsupported_coefficients = [
            field for field in self.coefficient_target_list if field not in self.supported_coefficient_fields
        ]
        if unsupported_coefficients:
            raise KeyError(
                f"{self.dataset_name} does not provide coefficient fields {unsupported_coefficients}. "
                f"Available fields: {list(self.supported_coefficient_fields)}."
            )
        self.num_surface_points = int(num_surface_points)
        # Keep the resolved count in ``interface_signature`` so this dataset
        # remains concat-compatible with the other surface readers, but forward
        # the original None sentinel to the sampler to reuse encoder points.
        self.reuse_surface_points_as_queries = num_surface_query_points is None
        self.num_surface_query_points = (
            self.num_surface_points if self.reuse_surface_points_as_queries else int(num_surface_query_points)
        )
        self.num_query_points = 0
        self.interface_signature = (
            tuple(self.surface_input_list),
            tuple(self.surface_target_list),
            tuple(self.volume_target_list),
            tuple(self.coefficient_target_list),
            self.num_surface_points,
            self.num_surface_query_points,
            self.num_query_points,
            3,
        )

        selected = filter_cases_by_ids(cases, ids_file)
        selected = select_cases_by_geometry(selected, split, val_ratio)
        if placement_override is not None:
            placement = float(placement_override)
            if not np.isfinite(placement):
                raise ValueError("placement_override must be finite when provided.")
            if any(len(case.cond) != 2 for case in selected):
                raise ValueError("placement_override requires two-value condition vectors.")
            selected = [replace(case, cond=(float(case.cond[0]), placement)) for case in selected]
        self.surface = AerodynamicSurfaceDataset(
            cache_root=self.cache_root,
            dataset_name=self.cache_dataset_name,
            num_surface_points=self.num_surface_points,
            num_surface_query_points=None if self.reuse_surface_points_as_queries else self.num_surface_query_points,
            surface_input_list=self.surface_input_list,
            surface_target_list=self.surface_target_list,
            surface_sampling=surface_sampling,
            normalization=normalization,
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
            mmap=mmap,
        )
        self.volume = EmptyAerodynamicVolumeDataset(self.volume_target_list)
        if skip_incomplete:
            complete = [case for case in selected if self.surface.has_cache(case.name)]
            skipped = len(selected) - len(complete)
            if skipped:
                print(f"{self.dataset_name}: skipped {skipped} missing or invalid surface cache case(s).", flush=True)
            selected = complete
        self.case_records = list(selected)
        self.case_names = [case.name for case in self.case_records]
        if not self.case_records and not allow_empty:
            raise RuntimeError(
                f"No complete {self.dataset_name} cases are available. Build caches first or set allow_empty=True."
            )
        print(
            f"{self.dataset_name}: {len(self.case_records)} cases, split={split}, "
            f"repeat={self.repeat}, route={self.route}.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.case_records) * self.repeat

    def _aoa_for_case(self, case: AerodynamicSurfaceCase) -> float | None:
        if not self.rotate_geometry_by_aoa:
            return None
        if case.aoa_deg is None:
            raise ValueError(f"Cannot rotate {self.dataset_name} geometry: case {case.name} has no AoA metadata.")
        return float(case.aoa_deg)

    @staticmethod
    def _rotate_y(values: np.ndarray, aoa_deg: float | None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).copy()
        if aoa_deg is None:
            return values
        theta = np.deg2rad(float(aoa_deg))
        rotation = np.asarray(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float32,
        )
        return values @ rotation.T

    def _coordinate_frame(self, case: AerodynamicSurfaceCase, aoa_deg: float | None) -> CanonicalCoordinateFrame:
        reference = self._rotate_y(self.surface.coordinate_reference(case.name), aoa_deg)
        frame = CanonicalCoordinateFrame.from_surface_xyz(reference)
        return CanonicalCoordinateFrame(
            wind_heading_rad=frame.wind_heading_rad,
            xy_center=(frame.xy_center[0] if self.center_x else 0.0, frame.xy_center[1] if self.center_y else 0.0),
            z_min=frame.z_min if self.ground_z else 0.0,
        )

    def _transform_surface_input(
        self,
        surface_input: np.ndarray,
        aoa_deg: float | None,
        frame: CanonicalCoordinateFrame,
    ) -> np.ndarray:
        transformed = np.asarray(surface_input, dtype=np.float32).copy()
        offset = 0
        for field in self.surface_input_list:
            width = SURFACE_FIELD_DIMS[field]
            if field == "xyz":
                transformed[:, offset : offset + width] = self._rotate_y(
                    transformed[:, offset : offset + width], aoa_deg
                )
            elif field in {"normal", "normals"}:
                transformed[:, offset : offset + width] = self._rotate_y(
                    transformed[:, offset : offset + width], aoa_deg
                )
            offset += width
        return transform_packed_surface_input(transformed, self.surface_input_list, frame)

    def __getitem__(self, idx: int):
        if not self.case_records:
            raise IndexError(f"{self.dataset_name} is empty.")
        case = self.case_records[idx % len(self.case_records)]
        rng = rng_for_index(self.seed, idx, self.deterministic)
        surface_input, surface_query, surface_target = self.surface.sample(case.name, rng)
        volume_query, volume_target = self.volume.sample(case.name, rng)
        aoa_deg = self._aoa_for_case(case)
        frame = self._coordinate_frame(case, aoa_deg)
        surface_input = self._transform_surface_input(surface_input, aoa_deg, frame)
        surface_query = frame.transform_xyz(self._rotate_y(surface_query, aoa_deg))
        coefficient_target = np.asarray(
            [case.coefficients.get(field, np.nan) for field in self.coefficient_target_list], dtype=np.float32
        )
        return (
            torch.from_numpy(surface_input).float(),
            torch.from_numpy(surface_query).float(),
            torch.from_numpy(surface_target).float(),
            torch.from_numpy(volume_query).float(),
            torch.from_numpy(volume_target).float(),
            torch.tensor(case.cond, dtype=torch.float32),
            torch.tensor(self.route, dtype=torch.long),
            torch.from_numpy(coefficient_target).float(),
        )


__all__ = ["AerodynamicSurfaceOnlyDataset"]
