from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.cache.high_speed_train import (
    DEFAULT_SURFACE_CACHE_POINTS,
    DEFAULT_VOLUME_CACHE_POINTS,
    HIGH_SPEED_TRAIN_ROOT,
    normalized_train_type,
)
from dataset.surface.base import EmptySurfaceDataset
from dataset.surface.high_speed_train import HighSpeedTrainSurfaceDataset
from dataset.utils.load import ensure_xyz_prefix, list_high_speed_cases, parse_high_speed_case_cond
from dataset.utils.sampling import filter_names, rng_for_index
from dataset.volume.base import EmptyVolumeDataset
from dataset.volume.high_speed_train import HighSpeedTrainVolumeDataset


class HighSpeedTrainDatasetBase(Dataset):
    def __init__(
        self,
        train_type: str,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
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
        split: str = "all",
        val_ratio: float = 0.1,
        repeat: int = 1,
        normalization: str = "physical",
        center_x: bool = True,
        route: int = 0,
        mmap: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        clamp_cp: tuple[float, float] | None = (-5.0, 1.05),
    ) -> None:
        if not use_surface and not use_volume:
            raise ValueError("At least one of use_surface/use_volume must be True.")
        if split not in {"all", "train", "val", "test"}:
            raise ValueError("split must be 'all', 'train', 'val', or 'test'.")

        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or os.path.join(root_dir, "cache")
        self.use_surface = bool(use_surface)
        self.use_volume = bool(use_volume)
        self.repeat = int(repeat)
        self.center_x = bool(center_x)
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

        case_dirs = list_high_speed_cases(root_dir, self.train_type)
        names = filter_names([path.name for path in case_dirs], ids_file)
        n_val = max(1, int(round(len(names) * float(val_ratio)))) if names else 0
        if split == "train":
            names = names[n_val:]
        elif split in {"val", "test"}:
            names = names[:n_val]
        self.case_names = names
        if not self.case_names:
            raise RuntimeError(f"No HighSpeedTrain cases available for train_type={self.train_type}, split={split}.")

        if self.use_surface:
            self.surface = HighSpeedTrainSurfaceDataset(
                cache_root=self.cache_root,
                train_type=self.train_type,
                num_surface_points=num_surface_points,
                num_surface_query_points=num_surface_query_points,
                surface_input_list=self.surface_input_list,
                surface_target_list=self.surface_target_list,
                surface_sampling=surface_sampling,
                surface_cache_points=surface_cache_points,
                normalization=normalization,
                clamp_cp=clamp_cp,
                mmap=mmap,
            )
        else:
            self.surface = EmptySurfaceDataset(self.surface_input_list, self.surface_target_list)

        if self.use_volume:
            self.volume = HighSpeedTrainVolumeDataset(
                cache_root=self.cache_root,
                train_type=self.train_type,
                num_query_points=num_query_points,
                volume_target_list=self.volume_target_list,
                volume_cache_points=volume_cache_points,
                normalization=normalization,
                clamp_cp=clamp_cp,
                mmap=mmap,
            )
        else:
            self.volume = EmptyVolumeDataset(self.volume_target_list)

        print(
            "HighSpeedTrain dataset: "
            f"{self.train_type}, {len(self.case_names)} cases, "
            f"use_surface={self.use_surface}, use_volume={self.use_volume}, repeat={self.repeat}."
        )

    def __len__(self) -> int:
        return len(self.case_names) * self.repeat

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx % len(self.case_names)]
        rng = rng_for_index(self.seed, idx, self.deterministic)
        cond, _ = parse_high_speed_case_cond(case_name)

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
            torch.from_numpy(cond.copy()).float(),
            torch.tensor(self.route, dtype=torch.long),
        )
