"""Shared, dependency-light helpers for surface-pressure datasets.

The paper datasets use different raw formats, but their training representation is
the same: cached rows of ``xyz, Cp, normal, area``.  Keeping this module free of
VTK/PyVista imports lets cached datasets be used on training workers that only
need NumPy and PyTorch.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SURFACE_CACHE_COLUMNS = ("x", "y", "z", "p", "nx", "ny", "nz", "area")
SURFACE_FIELD_DIMS = {
    "xyz": 3,
    "p": 1,
    "normal": 3,
    "normals": 3,
    "area": 1,
    "log_area": 1,
}


@dataclass(frozen=True)
class AerodynamicSurfaceCase:
    """One CFD condition over one geometry.

    ``name`` is the stable cache key, whereas ``geometry_id`` is used for
    geometry-level train/validation/test splits.  The raw source paths are
    intentionally carried as metadata only; regular dataset reads use caches.
    """

    name: str
    geometry_id: str
    cond: tuple[float, ...]
    source_paths: tuple[str, ...] = ()
    coefficients: Mapping[str, float] = field(default_factory=dict)
    aoa_deg: float | None = None


def cached_surface_cases(
    cache_root: str | Path, dataset_name: str, *, ids_file: str | Path | None = None,
) -> list[AerodynamicSurfaceCase]:
    """Discover a frozen cache cohort without raw-file enumeration or silent skips."""
    root = Path(cache_root)
    root = root if root.name == dataset_name else root / dataset_name
    manifest = root / "case_manifest.json"
    if manifest.is_file():
        records = json.loads(manifest.read_text())["cases"]
    else:
        records = []
        for folder in sorted((root / "cases").glob("*")):
            if not folder.is_dir():
                continue
            path = folder / "metadata.json"
            if not path.is_file():
                raise FileNotFoundError(f"Canonical cache metadata is missing: {path}")
            records.append(json.loads(path.read_text()))
    selected = None
    if ids_file is not None:
        selected = {line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()}
    cases = []
    for data in records:
        name = data["case"]
        if selected is not None and name not in selected and str(data.get("geometry_id")) not in selected:
            continue
        folder = root / "cases" / name
        if not (folder / "metadata.json").is_file() or not (folder / "surface.npy").is_file():
            raise FileNotFoundError(f"Frozen {dataset_name} cohort has an incomplete cache case: {folder}. Build its cache before training or evaluation.")
        if "condition" not in data or "geometry_id" not in data:
            raise ValueError(f"Canonical cache metadata lacks condition/geometry_id: {folder}")
        cases.append(AerodynamicSurfaceCase(
            name=name, geometry_id=str(data["geometry_id"]),
            cond=tuple(float(value) for value in data["condition"]),
            source_paths=tuple(data.get("source_paths", ())),
            coefficients=data.get("coefficients", {}), aoa_deg=data.get("aoa_deg"),
        ))
    return cases


def ensure_surface_input_fields(
    fields: Iterable[str] | None,
    default: tuple[str, ...] = ("xyz", "normal", "area"),
) -> list[str]:
    values = list(default if fields is None else fields)
    if "xyz" not in values:
        values.insert(0, "xyz")
    if values[0] != "xyz":
        values = ["xyz"] + [field for field in values if field != "xyz"]
    validate_surface_fields(values)
    return values


def validate_surface_fields(fields: Iterable[str]) -> None:
    unsupported = [field for field in fields if field not in SURFACE_FIELD_DIMS]
    if unsupported:
        raise KeyError(
            "Unsupported aerodynamic surface fields: "
            f"{unsupported}. Available fields are {sorted(SURFACE_FIELD_DIMS)}."
        )


def surface_field_dim(fields: Iterable[str]) -> int:
    validate_surface_fields(fields)
    return sum(SURFACE_FIELD_DIMS[field] for field in fields)


def surface_cache_path(cache_root: str | Path, dataset_name: str, case_name: str) -> Path:
    root = Path(cache_root)
    dataset_root = root if root.name == dataset_name else root / dataset_name
    return dataset_root / "cases" / case_name / "surface.npy"


def find_surface_cache_path(cache_root: str | Path, dataset_name: str, case_name: str) -> Path:
    """Read canonical cases first, retaining an explicit legacy cache fallback."""
    canonical = surface_cache_path(cache_root, dataset_name, case_name)
    if canonical.is_file():
        return canonical
    legacy = Path(cache_root) / dataset_name / f"{case_name}.npy"
    return legacy if legacy.is_file() else canonical


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Sort identifiers naturally, e.g. sample-9 before sample-10."""

    return tuple(int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", value))


def select_cases_by_geometry(
    cases: Iterable[AerodynamicSurfaceCase],
    split: str,
    val_ratio: float,
) -> list[AerodynamicSurfaceCase]:
    """Select a deterministic geometry-disjoint split.

    The final geometries in natural order are reserved for validation/test.  A
    user-supplied ``ids_file`` remains available in the dataset class for fully
    custom or published splits.
    """

    split = str(split).lower()
    if split not in {"all", "train", "val", "test"}:
        raise ValueError("split must be 'all', 'train', 'val', or 'test'.")
    cases = list(cases)
    if split == "all":
        return cases
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be in (0, 1) when split is not 'all'.")

    geometry_ids = sorted({case.geometry_id for case in cases}, key=natural_sort_key)
    held_out_count = max(1, int(round(len(geometry_ids) * float(val_ratio))))
    held_out = set(geometry_ids[-held_out_count:])
    if split == "train":
        return [case for case in cases if case.geometry_id not in held_out]
    return [case for case in cases if case.geometry_id in held_out]


def filter_cases_by_ids(
    cases: Iterable[AerodynamicSurfaceCase], ids_file: str | Path | None
) -> list[AerodynamicSurfaceCase]:
    if ids_file is None:
        return list(cases)
    with open(ids_file, "r", encoding="utf-8") as handle:
        ids = {line.strip() for line in handle if line.strip()}
    return [case for case in cases if case.name in ids or case.geometry_id in ids]


def rotation_matrix_y(aoa_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(aoa_deg))
    return np.asarray(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=np.float32,
    )


def transform_surface_xyz(
    xyz: np.ndarray,
    *,
    aoa_deg: float | None = None,
    center_x: bool = True,
    center_y: bool = True,
    ground_z: bool = True,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).copy()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz with shape (N, 3), got {xyz.shape}.")
    if aoa_deg is not None:
        xyz = xyz @ rotation_matrix_y(float(aoa_deg)).T
    if xyz.shape[0] == 0:
        return xyz
    if center_x:
        xyz[:, 0] -= 0.5 * (xyz[:, 0].max() + xyz[:, 0].min())
    if center_y:
        xyz[:, 1] -= 0.5 * (xyz[:, 1].max() + xyz[:, 1].min())
    if ground_z:
        xyz[:, 2] -= xyz[:, 2].min()
    return xyz


def transform_surface_normals(normals: np.ndarray, aoa_deg: float | None = None) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32).copy()
    if aoa_deg is not None:
        normals = normals @ rotation_matrix_y(float(aoa_deg)).T
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def parse_tecplot_variables(line: str) -> list[str]:
    if "=" not in line:
        raise ValueError(f"Invalid Tecplot VARIABLES line: {line!r}")
    return [value.strip().strip('"').lower() for value in line.split("=", 1)[1].split(",")]


def read_full_aircraft_result_coefficients(
    path: str | Path,
    *,
    tolerance: float = 5e-3,
) -> dict[tuple[float, float], dict[str, float]]:
    """Read the NGA aggregate force table keyed by ``(Mach, AoA)``.

    Values are preserved exactly as provided by the CFD result file.  This is
    deliberately separate from pressure-integrated loads, which can later be
    computed from cached normals and areas.
    """

    path = Path(path)
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    variable_line = next((line for line in lines if line.strip().upper().startswith("VARIABLES")), None)
    if variable_line is None:
        return {}
    names = parse_tecplot_variables(variable_line)
    name_to_index = {name: index for index, name in enumerate(names)}
    if "ma" not in name_to_index or "alpha" not in name_to_index:
        return {}

    result: dict[tuple[float, float], dict[str, float]] = {}
    after_zone = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("ZONE"):
            after_zone = True
            continue
        if not after_zone or not stripped:
            continue
        values = stripped.split()
        if len(values) < len(names):
            continue
        try:
            row = [float(value) for value in values[: len(names)]]
        except ValueError:
            continue
        mach = round(row[name_to_index["ma"]], 6)
        aoa = round(row[name_to_index["alpha"]], 6)
        result[(mach, aoa)] = {name.upper(): row[index] for name, index in name_to_index.items()}
    return result


def coefficients_for_condition(
    coefficients: Mapping[tuple[float, float], Mapping[str, float]],
    mach: float,
    aoa: float,
    *,
    tolerance: float = 5e-3,
) -> dict[str, float]:
    for (result_mach, result_aoa), values in coefficients.items():
        if abs(float(result_mach) - float(mach)) <= tolerance and abs(float(result_aoa) - float(aoa)) <= tolerance:
            return {str(name).upper(): float(value) for name, value in values.items()}
    return {}
