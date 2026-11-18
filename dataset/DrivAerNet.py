from __future__ import annotations

import os
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.cache.drivaernet import (
    DEFAULT_CACHE_ROOT,
    cache_path,
    DEFAULT_CSV_FILE,
    DEFAULT_SURFACE_CACHE_POINTS,
    DEFAULT_VOLUME_CACHE_POINTS,
    DRIVAERNET_ROOT,
    DrivAerNetPlusPlusCacheBuilder,
    DrivAerNetPlusPlusWallDistanceCacheBuilder,
    compute_cache_statistics,
    compute_coordinate_statistics,
    coordinate_statistics_path,
    load_or_compute_coordinate_statistics,
)
from dataset.surface.base import EmptySurfaceDataset
from dataset.surface.drivaernet import DrivAerNetSurfaceDataset
from dataset.utils.load import ensure_xyz_prefix
from dataset.utils.coordinate_frame import CanonicalCoordinateFrame, transform_packed_surface_input
from dataset.utils.normalization import (
    DEFAULT_PRESSURE_SCALE,
    DEFAULT_U_REF,
    DEFAULT_VELOCITY_SCALE,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VELOCITY_MEAN,
    VELOCITY_STD,
)
from dataset.utils.sampling import filter_names, read_ids, rng_for_index
from dataset.volume.base import EmptyVolumeDataset
from dataset.volume.drivaernet import DrivAerNetVolumeDataset


DEFAULT_REYNOLDS_SIMILARITY_REFERENCE_SPEED = 30.0


def _reynolds_similarity_speeds(speeds: Iterable[float] | None) -> tuple[float, ...]:
    """Validate and preserve the configured discrete similarity speeds."""

    if speeds is None:
        return ()
    values = tuple(float(speed) for speed in speeds)
    if not values:
        raise ValueError("reynolds_similarity_speeds must not be empty when enabled.")
    if any(not np.isfinite(speed) or speed <= 0.0 for speed in values):
        raise ValueError("reynolds_similarity_speeds must contain finite positive speeds in m/s.")
    if len(set(values)) != len(values):
        raise ValueError("reynolds_similarity_speeds must not contain duplicate speeds.")
    return values


def _reynolds_similarity_factor_range(
    values: Iterable[float] | None,
) -> tuple[float, float] | None:
    """Validate a continuous speed multiplier / reciprocal length range."""

    if values is None:
        return None
    result = tuple(float(value) for value in values)
    if len(result) != 2:
        raise ValueError("reynolds_similarity_factor_range must contain [low, high].")
    low, high = result
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        raise ValueError(
            "reynolds_similarity_factor_range must contain finite values with 0 < low < high."
        )
    return low, high


class DrivAerNetPlusPlusDataset(Dataset):
    """
    Unified DrivAerNet++ surface/volume dataset.

    ``cache_profile=None`` reads the frozen compact cohort at ``cache_root/case``.
    A named profile reads ``cache_root/case/pools/profile``. Without ids it uses
    that pool's actual coverage; with ids every requested case/domain must exist.
    ``surface_cache_profile`` / ``volume_cache_profile`` override the shared
    profile independently, combining their coverage by intersection without ids.
    Normal, area and pressure are required only when consumed by the task; none
    are synthesized. Historical flat arrays require ``allow_legacy_flat_cache``.

    Returns:
        surface_input:  (Ns, Cinput), xyz plus configured surface features.
        surface_query:  (Nsurface, 3)
        surface_target: (Nsurface, Csurface)
        volume_query:   (Nvolume, 3)
        volume_target:  (Nvolume, Cvolume)
        cond:           (Ccond,)
        route:          scalar long tensor
        coefficient_target: (Ccoeff,), always empty for DrivAerNet++.
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
        coefficient_target_list: list[str] | tuple[str, ...] | None = None,
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
        wall_distance_target_transform: str | None = None,
        center_x: bool = True,
        center_y: bool = True,
        ground_z: bool = True,
        cond: Iterable[float] | None = None,
        reynolds_similarity_speeds: Iterable[float] | None = None,
        reynolds_similarity_factor_range: Iterable[float] | None = None,
        reynolds_similarity_reference_speed: float = DEFAULT_REYNOLDS_SIMILARITY_REFERENCE_SPEED,
        route: int = 0,
        mmap: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        return_surface_geometry: bool | None = None,
        clip_cp: bool = False,
        volume_pressure_mean: float | None = None,
        volume_pressure_std: float | None = None,
        coordinate_frame: str = "canonical",
        validity_file: str | None = None,
        cache_profile: str | None = None,
        surface_cache_profile: str | None = None,
        volume_cache_profile: str | None = None,
        allow_legacy_flat_cache: bool = False,
    ) -> None:
        del pressure_scale, velocity_scale
        self.return_surface_geometry = bool(return_surface_geometry)
        self.coordinate_frame = coordinate_frame
        if coordinate_frame not in {"canonical", "raw"}:
            raise ValueError("coordinate_frame must be canonical or raw")
        if not use_surface and not use_volume:
            raise ValueError("At least one of use_surface/use_volume must be True.")
        if normalization not in {"physical", "standard", "none"}:
            raise ValueError("normalization must be 'physical', 'standard', or 'none'.")

        self.root_dir = root_dir
        self.pressure_dir = os.path.join(root_dir, "Pressure")
        self.cfd_dir = os.path.join(root_dir, "CFD")
        self.cache_root = str(Path(cache_root or DEFAULT_CACHE_ROOT).expanduser())
        self.surface_cache_profile = cache_profile if surface_cache_profile is None else surface_cache_profile
        self.volume_cache_profile = cache_profile if volume_cache_profile is None else volume_cache_profile
        for profile in (cache_profile, self.surface_cache_profile, self.volume_cache_profile):
            if profile is not None and (not profile or profile in {".", ".."} or Path(profile).name != profile):
                raise ValueError("cache profiles must be a single directory name or None")
        self.cache_profile = cache_profile
        self.allow_legacy_flat_cache = bool(allow_legacy_flat_cache)
        self.use_surface = bool(use_surface)
        self.use_volume = bool(use_volume)
        self.repeat = int(repeat)
        self.center_x = bool(center_x)
        self.center_y = bool(center_y)
        self.ground_z = bool(ground_z)
        self.cond = np.asarray([float(u_ref) / 100.0, 0.0] if cond is None else list(cond), dtype=np.float32)
        if self.cond.shape != (2,):
            raise ValueError("DrivAerNet++ cond must be the canonical two-vector [U_inf / 100, placement].")
        self.reynolds_similarity_speeds = _reynolds_similarity_speeds(reynolds_similarity_speeds)
        self.reynolds_similarity_factor_range = _reynolds_similarity_factor_range(
            reynolds_similarity_factor_range
        )
        if self.reynolds_similarity_speeds and self.reynolds_similarity_factor_range is not None:
            raise ValueError(
                "reynolds_similarity_speeds and reynolds_similarity_factor_range are mutually exclusive."
            )
        self.reynolds_similarity_reference_speed = float(reynolds_similarity_reference_speed)
        if self.reynolds_similarity_reference_speed <= 0.0:
            raise ValueError("reynolds_similarity_reference_speed must be positive.")
        self.route = int(route)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal"))
        self.surface_target_list = list(surface_target_list or [])
        self.volume_target_list = list(volume_target_list or [])
        self.coefficient_target_list = list(coefficient_target_list or [])
        if self.reynolds_similarity_speeds or self.reynolds_similarity_factor_range is not None:
            if not self.use_surface or self.use_volume or self.volume_target_list:
                raise ValueError(
                    "DrivAerNet++ Reynolds-similarity augmentation is surface-only. "
                    "Set use_surface=True, use_volume=False, and volume_target_list=[]."
                )
            if normalization != "physical":
                raise ValueError(
                    "DrivAerNet++ Reynolds-similarity augmentation requires normalization='physical' "
                    "so the pressure target remains the invariant Cp."
                )
            if not np.isclose(float(u_ref), self.reynolds_similarity_reference_speed):
                raise ValueError(
                    "u_ref must equal reynolds_similarity_reference_speed when Reynolds-similarity augmentation is enabled."
                )
            if not np.isclose(float(self.cond[1]), 0.0):
                raise ValueError("DrivAerNet++ Reynolds-similarity augmentation is defined for the ground placement (0).")
        if self.coefficient_target_list:
            raise ValueError("DrivAerNet++ does not provide coefficient_target_list labels.")
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = self.num_surface_points if num_surface_query_points is None else int(num_surface_query_points)
        self.num_query_points = int(num_query_points)
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

        # Discover the cohort from caches themselves; raw meshes are never a
        # fallback for incomplete cache cohorts or an implicitly changed profile.
        cache_directory = Path(self.cache_root)
        manifest_file = cache_directory / "manifest.json"
        if manifest_file.is_file():
            metadata = json.loads(manifest_file.read_text())
            if not metadata.get("complete"):
                raise ValueError(f"Incomplete DrivAerNet cache manifest: {manifest_file}")
            case_names = sorted(row["name"] for row in metadata["records"])
            if len(set(case_names)) != len(case_names) or len(case_names) != metadata.get("cases", len(case_names)):
                raise ValueError(f"Invalid DrivAerNet cache cohort manifest: {manifest_file}")
        elif cache_directory.is_dir():
            case_names = sorted(directory.name for directory in cache_directory.iterdir()
                                if directory.is_dir() and any((directory / item).exists()
                                    for item in ("metadata.json", "surface.npy", "volume.npy", "pools")))
        else:
            raise FileNotFoundError(f"DrivAerNet cache root does not exist: {cache_directory}")
        profiles = (("surface", self.use_surface, self.surface_cache_profile),
                    ("volume", self.use_volume, self.volume_cache_profile))
        named_profiles = any(enabled and profile is not None for _, enabled, profile in profiles)
        if not case_names and self.allow_legacy_flat_cache and not named_profiles:
            modes = ("random", "uniform") if surface_sampling == "mixed" else (surface_sampling,)
            folders = [Path(cache_path(cache_directory, f"surface_{mode}", surface_cache_points, "case")).parent
                       for mode in modes] if self.use_surface else []
            if self.use_volume:
                folders.append(Path(cache_path(cache_directory, "volume_random", volume_cache_points, "case")).parent)
            case_names = sorted({path.stem for folder in folders for path in folder.glob("*.npy")})
        requested_ids = read_ids(ids_file)
        if named_profiles:
            # Explicit ids define the requested cohort. Without ids, combine the
            # actual coverage of the selected surface/volume pools by intersection.
            directories = [directory for directory in cache_directory.iterdir() if directory.is_dir()]
            coverage = []
            for domain, enabled, profile in profiles:
                if not enabled:
                    continue
                if profile is None:
                    coverage.append(set(case_names))
                else:
                    coverage.append({directory.name for directory in directories
                                     if (directory / "pools" / profile / f"{domain}.npy").is_file()})
            case_names = sorted(set(case_names).union(*coverage)) if requested_ids is not None else sorted(set.intersection(*coverage))
        missing_ids = set() if requested_ids is None else requested_ids - set(case_names)
        if missing_ids:
            raise FileNotFoundError(f"{len(missing_ids)} requested DrivAerNet cases have no cache: "
                                    f"{sorted(missing_ids)[:5]}; split cannot be truncated")
        self.case_names = filter_names(case_names, ids_file)
        if not self.case_names:
            raise RuntimeError("No DrivAerNet cases available for the requested cache profile and ids_file.")
        missing = []
        for name in self.case_names:
            for domain, enabled, profile in profiles:
                directory = cache_directory / name
                if profile is not None:
                    directory = directory / "pools" / profile
                if not enabled or (directory / f"{domain}.npy").is_file():
                    continue
                legacy = []
                if self.allow_legacy_flat_cache and profile is None:
                    if domain == "surface":
                        modes = ("random", "uniform") if surface_sampling == "mixed" else (surface_sampling,)
                        legacy = [Path(cache_path(cache_directory, f"surface_{mode}", surface_cache_points, name)) for mode in modes]
                    else:
                        legacy = [Path(cache_path(cache_directory, "volume_random", volume_cache_points, name))]
                if not legacy or not all(path.is_file() for path in legacy):
                    missing.append(str(directory / f"{domain}.npy"))
        if missing:
            raise FileNotFoundError("DrivAerNet selected cache profiles are incomplete: "
                                    f"{len(missing)} missing domain files; first: {missing[0]}")

        self.validity = json.loads(Path(validity_file).read_text()) if validity_file else None
        common_kwargs = dict(
            cache_root=self.cache_root,
            allow_legacy_flat_cache=self.allow_legacy_flat_cache,
            normalization=normalization,
            clip_cp=clip_cp,
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
                return_query_normals=self.return_surface_geometry,
                cache_profile=self.surface_cache_profile,
                **common_kwargs,
            )
            self.surface.validity = self.validity
        else:
            self.surface = EmptySurfaceDataset(self.surface_input_list, self.surface_target_list)

        if self.use_volume:
            self.volume = DrivAerNetVolumeDataset(
                num_query_points=num_query_points,
                volume_target_list=self.volume_target_list,
                volume_cache_points=volume_cache_points,
                cache_profile=self.volume_cache_profile,
                wall_distance_target_transform=wall_distance_target_transform,
                **dict(common_kwargs, pressure_mean=pressure_mean if volume_pressure_mean is None else volume_pressure_mean,
                       pressure_std=pressure_std if volume_pressure_std is None else volume_pressure_std),
            )
        else:
            self.volume = EmptyVolumeDataset(self.volume_target_list)

        print(
            "DrivAerNet++ dataset: "
            f"{len(self.case_names)} cases, use_surface={self.use_surface}, "
            f"use_volume={self.use_volume}, repeat={self.repeat}, "
            f"reynolds_similarity_speeds={self.reynolds_similarity_speeds or None}, "
            f"reynolds_similarity_factor_range={self.reynolds_similarity_factor_range}."
        )

    def __len__(self) -> int:
        speed_count = len(self.reynolds_similarity_speeds) if self.reynolds_similarity_speeds else 1
        return len(self.case_names) * speed_count * self.repeat

    def _case_and_speed(self, idx: int) -> tuple[str, float | None]:
        if not self.reynolds_similarity_speeds:
            return self.case_names[idx % len(self.case_names)], None
        per_repeat = len(self.case_names) * len(self.reynolds_similarity_speeds)
        local_idx = idx % per_repeat
        speed_idx, case_idx = divmod(local_idx, len(self.case_names))
        return self.case_names[case_idx], self.reynolds_similarity_speeds[speed_idx]

    def _coordinate_frame(self, case_name: str) -> CanonicalCoordinateFrame:
        reference = self.surface.coordinate_reference(case_name) if self.use_surface else self.volume.coordinate_reference(case_name)
        frame = CanonicalCoordinateFrame.from_surface_xyz(reference)
        return CanonicalCoordinateFrame(
            wind_heading_rad=0.0,
            xy_center=(frame.xy_center[0] if self.center_x else 0.0, frame.xy_center[1] if self.center_y else 0.0),
            z_min=frame.z_min if self.ground_z else 0.0,
        )

    def __getitem__(self, idx: int):
        case_name, speed = self._case_and_speed(idx)
        rng = rng_for_index(self.seed, idx, self.deterministic)
        if self.reynolds_similarity_factor_range is not None:
            factor = float(rng.uniform(*self.reynolds_similarity_factor_range))
            speed = self.reynolds_similarity_reference_speed * factor

        surface_input, surface_query, surface_target = self.surface.sample(case_name, rng)
        volume_query, volume_target = self.volume.sample(case_name, rng)
        if self.coordinate_frame != "raw":
            frame = self._coordinate_frame(case_name)
            surface_input = transform_packed_surface_input(surface_input, self.surface_input_list, frame)
            surface_query = frame.transform_xyz(surface_query)
            volume_query = frame.transform_xyz(volume_query)

        cond = self.cond.copy()
        if speed is not None:
            # With L' = L * (U_ref / U') this keeps Re fixed.  Cp is already
            # the physical target under the required normalization='physical'.
            coordinate_scale = self.reynolds_similarity_reference_speed / float(speed)
            surface_input[:, :3] *= coordinate_scale
            surface_query *= coordinate_scale
            volume_query *= coordinate_scale
            # Per the surface-point-cloud augmentation contract, normals,
            # areas, and Cp labels are intentionally left unchanged.
            cond = np.asarray([float(speed) / 100.0, 0.0], dtype=np.float32)

        result = (
            torch.from_numpy(surface_input).float(),
            torch.from_numpy(surface_query).float(),
            torch.from_numpy(surface_target).float(),
            torch.from_numpy(volume_query).float(),
            torch.from_numpy(volume_target).float(),
            torch.from_numpy(cond).float(),
            torch.tensor(self.route, dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
        )
        if self.return_surface_geometry:
            normals = getattr(self.surface, "query_normals", None)
            if normals is None:
                if len(surface_query):
                    raise ValueError("return_surface_geometry requires aligned query normals in the selected cache")
                normals = np.empty((0, 3), np.float32)
            return result + ({"query_normals": torch.from_numpy(normals).float()},)
        return result


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
    "DEFAULT_REYNOLDS_SIMILARITY_REFERENCE_SPEED",
    "DEFAULT_PRESSURE_SCALE",
    "DEFAULT_VELOCITY_SCALE",
    "DRIVAERNET_ROOT",
    "PRESSURE_MEAN",
    "PRESSURE_STD",
    "VELOCITY_MEAN",
    "VELOCITY_STD",
    "DrivAerNetPlusPlusCacheBuilder",
    "DrivAerNetPlusPlusWallDistanceCacheBuilder",
    "DrivAerNetPlusPlusDataset",
    "DrivAerNetPlusPlusFieldDataset",
    "Dataset",
    "compute_cache_statistics",
    "compute_coordinate_statistics",
    "coordinate_statistics_path",
    "get_field_dataset",
    "load_or_compute_coordinate_statistics",
]
