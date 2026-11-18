"""Surface-pressure dataset for the generated high-speed-train CFD cases.

The source directory already contains compact ``[x, y, z, Cp]`` arrays.  This
reader deliberately uses those caches rather than loading the 46 GB of raw VTK
surface files in dataloader workers.  Unlike the original CRH450/Maglev data,
the generated cases contain no cached normals or surface areas; requests for
those fields receive the standard zero-normal and unit-area point-cloud
fallbacks from :mod:`dataset.surface.aerodynamic`.
"""

from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np

from dataset.utils.aerodynamic_surface import AerodynamicSurfaceCase
from dataset.utils.aerodynamic_surface_dataset import AerodynamicSurfaceOnlyDataset
from dataset.utils.coordinate_frame import CanonicalCoordinateFrame, wind_heading


GENERATED_HIGH_SPEED_TRAIN_ROOT = "/lustre/home/zdhs0017/data/ThingiCFD/high_speed_train"
GENERATED_HIGH_SPEED_TRAIN_CACHE_ROOT = "/data/home/zdhs0017/zoujunhong/data"
DEFAULT_PRESSURE_CACHE_DIR = "surface_pressure_100k_tmp_cell"
_CASE_NAME_RE = re.compile(
    r"^HighSpeedTrain_(?P<geometry>\d+)_Ux_(?P<ux>[-+]?\d*\.?\d+)_Uy_(?P<uy>[-+]?\d*\.?\d+)$"
)


def parse_generated_high_speed_train_name(stem: str) -> tuple[str, float, float]:
    """Return ``(geometry_id, Ux, Uy)`` encoded by a generated case name."""

    match = _CASE_NAME_RE.match(str(stem))
    if match is None:
        raise ValueError(f"Cannot parse generated high-speed-train case name: {stem!r}")
    return match.group("geometry"), float(match.group("ux")), float(match.group("uy"))


def generated_high_speed_train_cases(
    root_dir: str | Path = GENERATED_HIGH_SPEED_TRAIN_ROOT,
    *,
    pressure_cache_dir: str = DEFAULT_PRESSURE_CACHE_DIR,
    cache_root: str | Path | None = None,
    speed_scale: float = 100.0,
) -> list[AerodynamicSurfaceCase]:
    """Discover cached generated cases in the canonical ``[U/100, ground]`` form."""

    if float(speed_scale) <= 0.0:
        raise ValueError("speed_scale must be positive.")
    root = Path(root_dir)
    canonical_root = Path(cache_root or GENERATED_HIGH_SPEED_TRAIN_CACHE_ROOT)
    if canonical_root.name != "GeneratedHighSpeedTrain":
        canonical_root = canonical_root / "GeneratedHighSpeedTrain"
    canonical_dir = canonical_root / "cases"
    cache_dir = root / "cache" / str(pressure_cache_dir)
    use_canonical = (cache_root is not None or root == Path(GENERATED_HIGH_SPEED_TRAIN_ROOT)) and canonical_dir.is_dir()
    if use_canonical:
        manifest = canonical_root / "case_manifest.json"
        if manifest.is_file():
            paths = [canonical_dir / case["case"] / "surface.npy" for case in json.loads(manifest.read_text())["cases"]]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Frozen GeneratedHighSpeedTrain cohort has missing caches: {missing[:8]}")
        else:
            paths = sorted(canonical_dir.glob("*/surface.npy"))
    elif cache_dir.is_dir():
        paths = sorted(cache_dir.glob("*.npy"))
    else:
        raise FileNotFoundError(f"Generated high-speed-train pressure-cache directory does not exist: {cache_dir}")

    cases: list[AerodynamicSurfaceCase] = []
    for path in paths:
        case_name = path.parent.name if use_canonical else path.stem
        geometry_id, ux, uy = parse_generated_high_speed_train_name(case_name)
        speed = float(np.hypot(ux, uy))
        if speed <= 0.0:
            raise ValueError(f"Generated high-speed-train case {case_name!r} has non-positive inlet speed.")
        cases.append(
            AerodynamicSurfaceCase(
                name=case_name,
                geometry_id=geometry_id,
                cond=(speed / float(speed_scale), 0.0),
                source_paths=(str(root / "surface" / f"{case_name}.vtk"),),
            )
        )
    if not cases:
        raise RuntimeError(f"No generated high-speed-train .npy surface-pressure caches found in {cache_dir}.")
    return cases


class GeneratedHighSpeedTrainDataset(AerodynamicSurfaceOnlyDataset):
    """Generated train geometries with yaw-aligned point clouds and Cp labels."""

    dataset_name = "GeneratedHighSpeedTrain"

    def __init__(
        self,
        root_dir: str | Path = GENERATED_HIGH_SPEED_TRAIN_ROOT,
        cache_root: str | Path | None = None,
        pressure_cache_dir: str = DEFAULT_PRESSURE_CACHE_DIR,
        speed_scale: float = 100.0,
        align_freestream: bool = True,
        **kwargs,
    ) -> None:
        root = Path(root_dir)
        self.align_freestream = bool(align_freestream)
        self._case_wind: dict[str, tuple[float, float]] = {}
        cases = generated_high_speed_train_cases(
            root,
            pressure_cache_dir=pressure_cache_dir,
            cache_root=cache_root,
            speed_scale=speed_scale,
        )
        for case in cases:
            _, ux, uy = parse_generated_high_speed_train_name(case.name)
            self._case_wind[case.name] = (ux, uy)
        canonical_root = Path(cache_root or GENERATED_HIGH_SPEED_TRAIN_CACHE_ROOT)
        canonical_dataset_root = canonical_root if canonical_root.name == self.dataset_name else canonical_root / self.dataset_name
        use_canonical = (cache_root is not None or root == Path(GENERATED_HIGH_SPEED_TRAIN_ROOT)) and (canonical_dataset_root / "cases").is_dir()
        super().__init__(
            cases=cases,
            cache_root=canonical_root if use_canonical else root / "cache",
            cache_dataset_name=self.dataset_name if use_canonical else str(pressure_cache_dir),
            route=kwargs.pop("route", 1),
            **kwargs,
        )

    def _coordinate_frame(
        self,
        case: AerodynamicSurfaceCase,
        aoa_deg: float | None,
    ) -> CanonicalCoordinateFrame:
        if aoa_deg is not None:
            raise ValueError("GeneratedHighSpeedTrain does not define an angle-of-attack transform.")
        reference = self.surface.coordinate_reference(case.name)
        heading = 0.0
        if self.align_freestream:
            ux, uy = self._case_wind[case.name]
            heading = wind_heading(ux, uy)
        frame = CanonicalCoordinateFrame.from_surface_xyz(reference, heading)
        return CanonicalCoordinateFrame(
            wind_heading_rad=frame.wind_heading_rad,
            xy_center=(frame.xy_center[0] if self.center_x else 0.0, frame.xy_center[1] if self.center_y else 0.0),
            z_min=frame.z_min if self.ground_z else 0.0,
        )


Dataset = GeneratedHighSpeedTrainDataset


__all__ = [
    "DEFAULT_PRESSURE_CACHE_DIR",
    "GENERATED_HIGH_SPEED_TRAIN_ROOT",
    "Dataset",
    "GeneratedHighSpeedTrainDataset",
    "generated_high_speed_train_cases",
    "parse_generated_high_speed_train_name",
]
