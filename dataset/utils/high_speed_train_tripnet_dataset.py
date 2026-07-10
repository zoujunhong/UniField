from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.cache.high_speed_train import (
    DEFAULT_CACHE_DTYPE,
    DEFAULT_VOLUME_CACHE_POINTS,
    HIGH_SPEED_TRAIN_ROOT,
    cache_numpy_dtype,
    cache_path,
    high_speed_case_sort_key,
    normalized_cache_dtype,
    normalized_train_type,
    valid_npy_cache,
)
from dataset.cache.tripnet_high_speed import (
    DEFAULT_TRIPLANE_CHANNELS,
    DEFAULT_TRIPLANE_RESOLUTION,
    HighSpeedTrainTriplaneCacheBuilder,
    triplane_cache_path,
    valid_triplane_cache,
)
from dataset.utils.high_speed_train_dataset import (
    _required_high_speed_cols,
    _split_high_speed_names_by_geometry,
    read_high_speed_coefficients,
    validate_high_speed_coefficient_fields,
)
from dataset.utils.load import high_speed_type_dir, list_high_speed_cases, parse_high_speed_case_cond
from dataset.utils.sampling import filter_names, rng_for_index
from dataset.volume.high_speed_train import HighSpeedTrainVolumeDataset


def _complete_tripnet_case(
    case_name: str,
    cache_root: str,
    triplane_cache_root: str,
    train_type: str,
    volume_cache_points: int,
    cache_dtype: str,
    volume_target_list: list[str],
    triplane_channels: int,
    triplane_resolution: int,
) -> bool:
    min_cols = _required_high_speed_cols(["xyz", *volume_target_list])
    volume_path = cache_path(cache_root, train_type, "volume_random", volume_cache_points, case_name, cache_dtype)
    if not valid_npy_cache(volume_path, min_cols=min_cols, dtype=cache_numpy_dtype(cache_dtype)):
        return False
    tri_path = triplane_cache_path(triplane_cache_root, train_type, case_name, triplane_channels, triplane_resolution)
    return valid_triplane_cache(tri_path, triplane_channels, triplane_resolution)


class HighSpeedTrainTripNetDatasetBase(Dataset):
    """
    High-speed-train TripNet dataset.

    Returns the same tuple as the point-cloud high-speed dataset plus a geometry
    metadata dict. surface_input is a raw triplane tensor with shape
    (3, C, H, W); surface query/target are empty because this v1 trains only
    volume fields.
    """

    def __init__(
        self,
        train_type: str,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
        cache_root: str | None = None,
        triplane_cache_root: str | None = None,
        use_surface: bool = True,
        use_volume: bool = True,
        num_surface_points: int = 0,
        num_surface_query_points: int = 0,
        num_query_points: int = 8_192,
        surface_input_list: list[str] | tuple[str, ...] | None = ("triplane",),
        surface_target_list: list[str] | tuple[str, ...] | None = None,
        volume_target_list: list[str] | tuple[str, ...] | None = ("U",),
        coefficient_target_list: list[str] | tuple[str, ...] | None = None,
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        triplane_channels: int = DEFAULT_TRIPLANE_CHANNELS,
        triplane_resolution: int = DEFAULT_TRIPLANE_RESOLUTION,
        cache_dtype: str = DEFAULT_CACHE_DTYPE,
        ids_file: str | None = None,
        split: str = "all",
        val_ratio: float = 0.1,
        geometry_count: int | None = None,
        skip_incomplete: bool = True,
        allow_empty: bool = False,
        repeat: int = 1,
        normalization: str = "physical",
        standard_stats=None,
        center_x: bool = True,
        route: int = 0,
        mmap: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        clamp_cp: tuple[float, float] | None = None,
        **unused,
    ) -> None:
        del use_surface, num_surface_points, num_surface_query_points, unused
        if not use_volume:
            raise ValueError("TripNet high-speed dataset currently requires use_volume=True.")
        if split not in {"all", "train", "val", "test"}:
            raise ValueError("split must be 'all', 'train', 'val', or 'test'.")
        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.triplane_cache_root = triplane_cache_root or self.cache_root
        self.repeat = int(repeat)
        self.center_x = bool(center_x)
        self.route = int(route)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.surface_input_list = list(surface_input_list or ["triplane"])
        if self.surface_input_list != ["triplane"]:
            raise ValueError("TripNet high-speed dataset requires surface_input_list=['triplane'].")
        self.surface_target_list = list(surface_target_list or [])
        if self.surface_target_list:
            raise ValueError("TripNet volume v1 does not provide surface targets.")
        self.volume_target_list = list(volume_target_list or [])
        self.coefficient_target_list = list(coefficient_target_list or [])
        validate_high_speed_coefficient_fields(self.coefficient_target_list)
        self.num_query_points = int(num_query_points)
        self.cache_dtype = normalized_cache_dtype(cache_dtype)
        self.volume_cache_points = int(volume_cache_points)
        self.triplane_channels = int(triplane_channels)
        self.triplane_resolution = int(triplane_resolution)
        self.mmap_mode = "r" if mmap else None
        self.interface_signature = (
            tuple(self.surface_input_list),
            tuple(self.surface_target_list),
            tuple(self.volume_target_list),
            tuple(self.coefficient_target_list),
            0,
            0,
            self.num_query_points,
        )

        case_dirs = list_high_speed_cases(root_dir, self.train_type)
        names = filter_names([path.name for path in case_dirs], ids_file)
        names = _split_high_speed_names_by_geometry(
            names,
            self.train_type,
            split,
            float(val_ratio),
            geometry_count,
        )
        split_case_count = len(names)
        if skip_incomplete:
            names = [
                name
                for name in names
                if _complete_tripnet_case(
                    name,
                    self.cache_root,
                    self.triplane_cache_root,
                    self.train_type,
                    self.volume_cache_points,
                    self.cache_dtype,
                    self.volume_target_list,
                    self.triplane_channels,
                    self.triplane_resolution,
                )
            ]
            skipped = split_case_count - len(names)
            if skipped > 0:
                print(
                    "HighSpeedTrain TripNet dataset: "
                    f"skipped {skipped} incomplete cached cases for {self.train_type}, split={split}.",
                    flush=True,
                )
        self.case_names = sorted(names, key=high_speed_case_sort_key)
        if not self.case_names:
            message = (
                "No complete HighSpeedTrain TripNet cases available for "
                f"train_type={self.train_type}, split={split}. cache_root={self.cache_root}. "
                "Build triplane caches first or set skip_incomplete=False."
            )
            if not allow_empty:
                raise RuntimeError(message)
            print(f"HighSpeedTrain TripNet dataset warning: {message}", flush=True)

        self.volume = HighSpeedTrainVolumeDataset(
            cache_root=self.cache_root,
            train_type=self.train_type,
            num_query_points=self.num_query_points,
            volume_target_list=self.volume_target_list,
            volume_cache_points=self.volume_cache_points,
            cache_dtype=self.cache_dtype,
            normalization=normalization,
            standard_stats=standard_stats,
            clamp_cp=clamp_cp,
            mmap=mmap,
        )
        print(
            "HighSpeedTrain TripNet dataset: "
            f"{self.train_type}, {len(self.case_names)} cases, repeat={self.repeat}.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.case_names) * self.repeat

    def _load_triplane(self, case_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        path = triplane_cache_path(
            self.triplane_cache_root,
            self.train_type,
            case_name,
            self.triplane_channels,
            self.triplane_resolution,
        )
        if not valid_triplane_cache(path, self.triplane_channels, self.triplane_resolution):
            raise FileNotFoundError(f"Missing or invalid TripNet triplane cache: {path}")
        data = np.load(path, mmap_mode=self.mmap_mode)
        triplane = np.asarray(data["triplane"], dtype=np.float32)
        bounds = np.asarray(data["bounds"], dtype=np.float32)
        x_center = np.asarray(data["x_center"], dtype=np.float32).reshape(1)
        if not self.center_x:
            bounds = bounds.copy()
            bounds[:, 0] += float(x_center[0])
        return triplane, {"bounds": bounds, "x_center": x_center}

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx % len(self.case_names)]
        rng = rng_for_index(self.seed, idx, self.deterministic)
        cond, _ = parse_high_speed_case_cond(case_name)

        triplane, meta = self._load_triplane(case_name)
        volume_query, volume_target = self.volume.sample(case_name, rng)
        if self.center_x:
            volume_query = volume_query.copy()
            volume_query[:, 0] -= float(meta["x_center"][0])

        type_dir = high_speed_type_dir(self.root_dir, self.train_type)
        coefficient_target = read_high_speed_coefficients(type_dir / case_name, self.coefficient_target_list)
        surface_query = np.zeros((0, 3), dtype=np.float32)
        surface_target = np.zeros((0, 0), dtype=np.float32)
        geometry_meta = {
            "bounds": torch.from_numpy(meta["bounds"]).float(),
            "x_center": torch.from_numpy(meta["x_center"]).float(),
        }
        return (
            torch.from_numpy(triplane).float(),
            torch.from_numpy(surface_query).float(),
            torch.from_numpy(surface_target).float(),
            torch.from_numpy(volume_query).float(),
            torch.from_numpy(volume_target).float(),
            torch.from_numpy(cond.copy()).float(),
            torch.tensor(self.route, dtype=torch.long),
            torch.from_numpy(coefficient_target).float(),
            geometry_meta,
        )


__all__ = ["HighSpeedTrainTripNetDatasetBase", "HighSpeedTrainTriplaneCacheBuilder"]
