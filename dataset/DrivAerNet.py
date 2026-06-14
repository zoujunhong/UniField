from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.cache.drivaernet import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CSV_FILE,
    DEFAULT_SURFACE_CACHE_POINTS,
    DEFAULT_VOLUME_CACHE_POINTS,
    DRIVAERNET_ROOT,
    DrivAerNetPlusPlusCacheBuilder,
    compute_cache_statistics,
)
from dataset.surface.base import EmptySurfaceDataset
from dataset.surface.drivaernet import DrivAerNetSurfaceDataset
from dataset.utils.load import ensure_xyz_prefix, matched_vtk_names
from dataset.utils.normalization import (
    DEFAULT_PRESSURE_SCALE,
    DEFAULT_U_REF,
    DEFAULT_VELOCITY_SCALE,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VELOCITY_MEAN,
    VELOCITY_STD,
)
from dataset.utils.sampling import filter_names, rng_for_index
from dataset.volume.base import EmptyVolumeDataset
from dataset.volume.drivaernet import DrivAerNetVolumeDataset


class DrivAerNetPlusPlusDataset(Dataset):
    """
    Unified DrivAerNet++ surface/volume dataset.

    Returns:
        surface_input:  (Ns, Cinput), xyz plus configured surface features.
        surface_query:  (Nsurface, 3)
        surface_target: (Nsurface, Csurface)
        volume_query:   (Nvolume, 3)
        volume_target:  (Nvolume, Cvolume)
        cond:           (Ccond,)
        route:          scalar long tensor
    """

    def __init__(
        self,
        root_dir: str = DRIVAERNET_ROOT,
        cache_root: str | None = None,
        use_surface: bool = True,
        use_volume: bool = True,
        num_surface_points: int = 32_768,
        num_surface_query_points: int | None = None,
        num_query_points: int = 8_192,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = ("p",),
        volume_target_list: list[str] | tuple[str, ...] | None = ("U",),
        surface_sampling: str = "random",
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        ids_file: str | None = None,
        repeat: int = 1,
        normalization: str = "physical",
        u_ref: float = DEFAULT_U_REF,
        pressure_mean: float = PRESSURE_MEAN,
        pressure_std: float = PRESSURE_STD,
        velocity_mean: Iterable[float] = VELOCITY_MEAN,
        velocity_std: Iterable[float] = VELOCITY_STD,
        pressure_scale: float = DEFAULT_PRESSURE_SCALE,
        velocity_scale: float = DEFAULT_VELOCITY_SCALE,
        center_x: bool = True,
        cond: Iterable[float] = (0.3, 1.0),
        route: int = 0,
        mmap: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        return_surface_geometry: bool | None = None,
    ) -> None:
        del pressure_scale, velocity_scale, return_surface_geometry
        if not use_surface and not use_volume:
            raise ValueError("At least one of use_surface/use_volume must be True.")
        if normalization not in {"physical", "standard", "none"}:
            raise ValueError("normalization must be 'physical', 'standard', or 'none'.")

        self.root_dir = root_dir
        self.pressure_dir = os.path.join(root_dir, "Pressure")
        self.cfd_dir = os.path.join(root_dir, "CFD")
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.use_surface = bool(use_surface)
        self.use_volume = bool(use_volume)
        self.repeat = int(repeat)
        self.center_x = bool(center_x)
        self.cond = np.asarray(list(cond), dtype=np.float32)
        self.route = int(route)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal", "area"))
        self.surface_target_list = list(surface_target_list or [])
        self.volume_target_list = list(volume_target_list or [])
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = self.num_surface_points if num_surface_query_points is None else int(num_surface_query_points)
        self.num_query_points = int(num_query_points)
        self.interface_signature = (
            tuple(self.surface_input_list),
            tuple(self.surface_target_list),
            tuple(self.volume_target_list),
            self.num_surface_points,
            self.num_surface_query_points,
            self.num_query_points,
        )

        case_names = matched_vtk_names(self.pressure_dir, self.cfd_dir)
        self.case_names = filter_names(case_names, ids_file)
        if not self.case_names:
            raise RuntimeError("No DrivAerNet++ cases available for the requested ids_file.")

        common_kwargs = dict(
            cache_root=self.cache_root,
            normalization=normalization,
            u_ref=u_ref,
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
            velocity_mean=velocity_mean,
            velocity_std=velocity_std,
            mmap=mmap,
        )
        if self.use_surface:
            self.surface = DrivAerNetSurfaceDataset(
                num_surface_points=num_surface_points,
                num_surface_query_points=num_surface_query_points,
                surface_input_list=self.surface_input_list,
                surface_target_list=self.surface_target_list,
                surface_sampling=surface_sampling,
                surface_cache_points=surface_cache_points,
                **common_kwargs,
            )
        else:
            self.surface = EmptySurfaceDataset(self.surface_input_list, self.surface_target_list)

        if self.use_volume:
            self.volume = DrivAerNetVolumeDataset(
                num_query_points=num_query_points,
                volume_target_list=self.volume_target_list,
                volume_cache_points=volume_cache_points,
                **common_kwargs,
            )
        else:
            self.volume = EmptyVolumeDataset(self.volume_target_list)

        print(
            "DrivAerNet++ dataset: "
            f"{len(self.case_names)} cases, use_surface={self.use_surface}, "
            f"use_volume={self.use_volume}, repeat={self.repeat}."
        )

    def __len__(self) -> int:
        return len(self.case_names) * self.repeat

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx % len(self.case_names)]
        rng = rng_for_index(self.seed, idx, self.deterministic)

        surface_input, surface_query, surface_target = self.surface.sample(case_name, rng)
        volume_query, volume_target = self.volume.sample(case_name, rng)

        if self.center_x and surface_input.shape[0] > 0:
            x_center = 0.5 * (surface_input[:, 0].max() + surface_input[:, 0].min())
            surface_input = surface_input.copy()
            surface_input[:, 0] -= x_center
            if surface_query.shape[0] > 0:
                surface_query = surface_query.copy()
                surface_query[:, 0] -= x_center
            if volume_query.shape[0] > 0:
                volume_query = volume_query.copy()
                volume_query[:, 0] -= x_center

        return (
            torch.from_numpy(surface_input).float(),
            torch.from_numpy(surface_query).float(),
            torch.from_numpy(surface_target).float(),
            torch.from_numpy(volume_query).float(),
            torch.from_numpy(volume_target).float(),
            torch.from_numpy(self.cond.copy()).float(),
            torch.tensor(self.route, dtype=torch.long),
        )


DrivAerNetPlusPlusFieldDataset = DrivAerNetPlusPlusDataset
Dataset = DrivAerNetPlusPlusDataset


def get_field_dataset(**kwargs) -> DrivAerNetPlusPlusDataset:
    return DrivAerNetPlusPlusDataset(**kwargs)


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_CSV_FILE",
    "DEFAULT_SURFACE_CACHE_POINTS",
    "DEFAULT_VOLUME_CACHE_POINTS",
    "DEFAULT_U_REF",
    "DEFAULT_PRESSURE_SCALE",
    "DEFAULT_VELOCITY_SCALE",
    "DRIVAERNET_ROOT",
    "PRESSURE_MEAN",
    "PRESSURE_STD",
    "VELOCITY_MEAN",
    "VELOCITY_STD",
    "DrivAerNetPlusPlusCacheBuilder",
    "DrivAerNetPlusPlusDataset",
    "DrivAerNetPlusPlusFieldDataset",
    "Dataset",
    "compute_cache_statistics",
    "get_field_dataset",
]
