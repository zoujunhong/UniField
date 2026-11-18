"""Cache builders for AircraftWing, FullAircraft, and ThingiCFD.

Training datasets deliberately read only compact NumPy caches.  This module is
the explicit raw-data boundary: VTK is needed for AircraftWing/ThingiCFD, while
the FullAircraft Tecplot reader is implemented with NumPy and the ASCII files
provided with the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset.utils.aerodynamic_surface import SURFACE_CACHE_COLUMNS, surface_cache_path


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def _point_geometry(points: np.ndarray, faces: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Compute area-lumped point normals and control areas from triangle faces."""

    points = np.asarray(points, dtype=np.float32)
    point_count = points.shape[0]
    if faces is None or faces.size == 0:
        return np.zeros((point_count, 3), dtype=np.float32), np.ones((point_count, 1), dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    valid = np.all((triangles >= 0) & (triangles < point_count), axis=1)
    triangles = triangles[valid]
    if triangles.size == 0:
        return np.zeros((point_count, 3), dtype=np.float32), np.ones((point_count, 1), dtype=np.float32)
    edge_a = points[triangles[:, 1]] - points[triangles[:, 0]]
    edge_b = points[triangles[:, 2]] - points[triangles[:, 0]]
    cross = np.cross(edge_a, edge_b)
    double_area = np.linalg.norm(cross, axis=1)
    nondegenerate = double_area > 1e-12
    triangles = triangles[nondegenerate]
    cross = cross[nondegenerate]
    area = 0.5 * double_area[nondegenerate]
    if triangles.size == 0:
        return np.zeros((point_count, 3), dtype=np.float32), np.ones((point_count, 1), dtype=np.float32)
    normal_sum = np.zeros((point_count, 3), dtype=np.float64)
    point_area = np.zeros((point_count,), dtype=np.float64)
    np.add.at(normal_sum, triangles.reshape(-1), np.repeat(cross, 3, axis=0))
    np.add.at(point_area, triangles.reshape(-1), np.repeat(area / 3.0, 3))
    positive = point_area > 0.0
    fill_area = float(point_area[positive].mean()) if np.any(positive) else 1.0
    point_area = np.where(positive, point_area, fill_area)
    return _normalize_vectors(normal_sum), point_area.reshape(-1, 1).astype(np.float32)


def _atomic_save(path: str | Path, rows: np.ndarray, overwrite: bool = False) -> bool:
    path = Path(path)
    if path.is_file() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, np.asarray(rows, dtype=np.float32))
    temporary_path.replace(path)
    return True


def write_surface_cache(
    cache_root: str | Path,
    dataset_name: str,
    case_name: str,
    rows: np.ndarray,
    *,
    overwrite: bool = False,
    case_metadata: dict | None = None,
) -> bool:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != len(SURFACE_CACHE_COLUMNS):
        raise ValueError(
            f"Expected a non-empty (N, {len(SURFACE_CACHE_COLUMNS)}) surface cache for "
            f"{SURFACE_CACHE_COLUMNS}, got {rows.shape}."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"Refusing to cache non-finite rows for {dataset_name}/{case_name}.")
    target = surface_cache_path(cache_root, dataset_name, case_name)
    written = _atomic_save(target, rows, overwrite=overwrite)
    if written:
        metadata_path = target.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update(case_metadata or {})
        metadata.update(schema="field_case_v1", dataset=dataset_name, case=case_name,
                        surface={"file": "surface.npy", "columns": list(SURFACE_CACHE_COLUMNS),
                                 "shape": list(rows.shape), "dtype": str(rows.dtype), "pressure_kind": "Cp"})
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(metadata_path)
    return written


def _triangulated_point_geometry(surface) -> tuple[np.ndarray, np.ndarray]:
    triangulated = surface.triangulate(inplace=False)
    faces = np.asarray(triangulated.faces, dtype=np.int64)
    if faces.size == 0:
        return _point_geometry(np.asarray(triangulated.points, dtype=np.float32), None)
    faces = faces.reshape(-1, 4)[:, 1:]
    return _point_geometry(np.asarray(triangulated.points, dtype=np.float32), faces)


def read_vtk_surface_pressure(
    vtk_path: str | Path,
    *,
    pressure_field: str,
    dynamic_pressure: float | None = None,
) -> np.ndarray:
    """Return ``xyz, Cp, normal, area`` from a VTK surface mesh.

    ``dynamic_pressure`` converts OpenFOAM pressure to ``Cp``.  Values are
    expected to already be ``Cp`` when it is omitted (the AircraftWing files).
    """

    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Building VTK surface caches requires PyVista. Install the project CFD environment first."
        ) from error

    vtk_path = Path(vtk_path)
    mesh = pv.read(vtk_path)
    surface = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
    if pressure_field in surface.point_data:
        points = np.asarray(surface.points, dtype=np.float32)
        pressure = np.asarray(surface.point_data[pressure_field], dtype=np.float32).reshape(-1, 1)
        normals, area = _triangulated_point_geometry(surface)
    elif pressure_field in surface.cell_data:
        points = np.asarray(surface.cell_centers().points, dtype=np.float32)
        pressure = np.asarray(surface.cell_data[pressure_field], dtype=np.float32).reshape(-1, 1)
        try:
            normal_mesh = surface.compute_normals(
                point_normals=False,
                cell_normals=True,
                auto_orient_normals=True,
                consistent_normals=True,
                inplace=False,
            )
        except Exception:
            normal_mesh = surface.compute_normals(
                point_normals=False,
                cell_normals=True,
                auto_orient_normals=False,
                consistent_normals=False,
                inplace=False,
            )
        normals = _normalize_vectors(np.asarray(normal_mesh.cell_data["Normals"], dtype=np.float32))
        area_mesh = surface.compute_cell_sizes(length=False, area=True, volume=False)
        area = np.maximum(np.asarray(area_mesh.cell_data["Area"], dtype=np.float32).reshape(-1, 1), 1e-12)
    else:
        available = sorted(set(surface.point_data.keys()) | set(surface.cell_data.keys()))
        raise KeyError(f"Pressure field {pressure_field!r} is absent from {vtk_path}; available arrays: {available}.")
    if dynamic_pressure is not None:
        if float(dynamic_pressure) <= 0.0:
            raise ValueError(f"dynamic_pressure must be positive, got {dynamic_pressure} for {vtk_path}.")
        pressure = pressure / float(dynamic_pressure)
    rows = np.concatenate([points, pressure, normals, area], axis=1).astype(np.float32, copy=False)
    finite = np.all(np.isfinite(rows), axis=1)
    rows = rows[finite]
    if rows.shape[0] == 0:
        raise ValueError(f"No finite surface-pressure rows found in {vtk_path}.")
    return rows


_ZONE_COUNT_RE = re.compile(r"\b([NE])\s*=\s*(\d+)", flags=re.IGNORECASE)


def _tecplot_number(value: str) -> float:
    # NGA exports include MSVC's -nan(ind) spelling for failed CFD values.
    # Preserve it as NaN so finite rows elsewhere in the case remain usable.
    if "nan" in value.lower() and re.fullmatch(r"[-+]?nan(?:\([^)]*\))?", value, flags=re.IGNORECASE):
        return float("nan")
    return float(value)


def _parse_tecplot_part(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one FEPOINT Tecplot surface part as points/Cp and triangle faces."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    variable_index = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("VARIABLES")), None
    )
    zone_index = next((index for index, line in enumerate(lines) if line.strip().upper().startswith("ZONE")), None)
    if variable_index is None or zone_index is None:
        raise ValueError(f"Invalid Tecplot surface file {path}: missing VARIABLES or ZONE header.")
    if "=" not in lines[variable_index]:
        raise ValueError(f"Invalid VARIABLES header in {path}.")
    names = [value.strip().strip('"').lower() for value in lines[variable_index].split("=", 1)[1].split(",")]
    required = {"x", "y", "z", "cp"}
    missing = required.difference(names)
    if missing:
        raise KeyError(f"Tecplot surface file {path} is missing {sorted(missing)} from {names}.")
    counts = {kind.upper(): int(value) for kind, value in _ZONE_COUNT_RE.findall(lines[zone_index])}
    if "N" not in counts or "E" not in counts:
        raise ValueError(f"Cannot parse N/E counts from Tecplot ZONE header in {path}: {lines[zone_index]!r}")
    point_count, element_count = counts["N"], counts["E"]
    point_lines = lines[zone_index + 1 : zone_index + 1 + point_count]
    if len(point_lines) != point_count:
        raise ValueError(f"Tecplot surface file {path} ended before its {point_count} point rows.")
    points_and_cp = np.empty((point_count, 4), dtype=np.float32)
    indices = [names.index(name) for name in ("x", "y", "z", "cp")]
    for row_index, line in enumerate(point_lines):
        values = line.split()
        if len(values) < len(names):
            raise ValueError(f"Invalid point row {row_index} in {path}: {line!r}")
        try:
            points_and_cp[row_index] = [_tecplot_number(values[index]) for index in indices]
        except ValueError as error:
            raise ValueError(f"Non-numeric point row {row_index} in {path}: {line!r}") from error

    element_start = zone_index + 1 + point_count
    element_lines = lines[element_start : element_start + element_count]
    faces: list[list[int]] = []
    for line in element_lines:
        values = line.split()
        try:
            vertices = [int(value) - 1 for value in values]
        except ValueError:
            continue
        if len(vertices) == 3:
            faces.append(vertices)
        elif len(vertices) >= 4:
            faces.append([vertices[0], vertices[1], vertices[2]])
            faces.append([vertices[0], vertices[2], vertices[3]])
    return points_and_cp, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def read_tecplot_surface_pressure(paths: Iterable[str | Path], *, coordinate_scale: float = 1.0) -> np.ndarray:
    """Merge NGA's eleven Tecplot parts into the standard surface cache rows."""

    all_points: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for path in paths:
        points_and_cp, faces = _parse_tecplot_part(path)
        all_points.append(points_and_cp)
        if faces.size:
            all_faces.append(faces + offset)
        offset += points_and_cp.shape[0]
    if not all_points:
        raise ValueError("No Tecplot surface parts were supplied.")
    points_and_cp = np.concatenate(all_points, axis=0)
    points_and_cp[:, :3] *= float(coordinate_scale)
    faces = np.concatenate(all_faces, axis=0) if all_faces else None
    valid = np.all(np.isfinite(points_and_cp), axis=1)
    normals, area = _point_geometry(points_and_cp[:, :3], faces)
    rows = np.concatenate([points_and_cp[:, :3], points_and_cp[:, 3:4], normals, area], axis=1)
    rows = rows[valid]
    if rows.shape[0] == 0:
        raise ValueError("No finite Tecplot surface-pressure rows were found.")
    return rows.astype(np.float32, copy=False)


def _build_selected(builder, case_names: list[str] | None, limit: int | None, overwrite: bool) -> tuple[int, int]:
    return builder.build(case_names=case_names, limit=limit, overwrite=overwrite)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build cached surface-pressure arrays for the paper datasets.")
    parser.add_argument("--dataset", required=True, choices=("AircraftWing", "FullAircraft", "ThingiCFD"))
    parser.add_argument("--root-dir", default=None, help="Raw dataset root; defaults to the repository convention.")
    parser.add_argument("--cache-root", default=None, help="Cache root; defaults to ~/zoujunhong/data.")
    parser.add_argument("--case", action="append", default=None, help="Case cache key to build; can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Build at most this many selected cases.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cache files.")
    args = parser.parse_args(argv)
    if args.dataset == "AircraftWing":
        from dataset.AircraftWing import AircraftWingCacheBuilder as Builder
    elif args.dataset == "FullAircraft":
        from dataset.FullAircraft import FullAircraftCacheBuilder as Builder
    else:
        from dataset.ThingiCFD import ThingiCFDCacheBuilder as Builder
    builder = Builder(root_dir=args.root_dir, cache_root=args.cache_root)
    built, skipped = _build_selected(builder, args.case, args.limit, args.overwrite)
    print(f"{args.dataset} cache build complete: built={built}, existing_or_skipped={skipped}.")


if __name__ == "__main__":  # pragma: no cover - command line entry point
    main()


__all__ = [
    "SURFACE_CACHE_COLUMNS",
    "read_tecplot_surface_pressure",
    "read_vtk_surface_pressure",
    "write_surface_cache",
]
