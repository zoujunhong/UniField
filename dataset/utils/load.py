from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyvista as pv


FIELD_DIMS = {
    "xyz": 3,
    "p": 1,
    "U": 3,
    "k": 1,
    "omega": 1,
    "nut": 1,
    "normal": 3,
    "normals": 3,
    "area": 1,
    "log_area": 1,
}


def ensure_xyz_prefix(fields: list[str] | tuple[str, ...] | None, default: tuple[str, ...]) -> list[str]:
    values = list(default if fields is None else fields)
    if "xyz" not in values:
        values.insert(0, "xyz")
    if values[0] != "xyz":
        values = ["xyz"] + [field for field in values if field != "xyz"]
    return values


def field_dim(fields: list[str] | tuple[str, ...]) -> int:
    dim = 0
    for field in fields:
        key = "normal" if field == "normals" else field
        if key not in FIELD_DIMS:
            raise KeyError(f"Unsupported field '{field}'.")
        dim += FIELD_DIMS[key]
    return dim


def list_vtk_files(root_dir: str | Path) -> dict[str, str]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"VTK directory not found: {root}")
    files = sorted(path for path in root.iterdir() if path.suffix == ".vtk")
    return {path.stem: str(path) for path in files}


def matched_vtk_names(left_dir: str | Path, right_dir: str | Path) -> list[str]:
    left = list_vtk_files(left_dir)
    right = list_vtk_files(right_dir)
    names = sorted(set(left).intersection(right))
    if not names:
        raise RuntimeError(f"No matched VTK files found in {left_dir} and {right_dir}.")
    return names


def normalize_np_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


def surface_polydata(mesh: pv.DataSet) -> pv.PolyData:
    if isinstance(mesh, pv.PolyData):
        return mesh
    return mesh.extract_surface()


def compute_point_area(surface: pv.PolyData) -> np.ndarray:
    if surface.n_points == 0:
        return np.empty((0, 1), dtype=np.float32)
    if surface.n_cells == 0 or surface.faces.size == 0:
        return np.ones((surface.n_points, 1), dtype=np.float32)

    tri_surface = surface.triangulate(inplace=False)
    area_mesh = tri_surface.compute_cell_sizes(length=False, area=True, volume=False)
    cell_area = np.asarray(area_mesh.cell_data["Area"], dtype=np.float64)
    faces = np.asarray(tri_surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    point_area = np.zeros(tri_surface.n_points, dtype=np.float64)
    np.add.at(point_area, faces.reshape(-1), np.repeat(cell_area / 3.0, 3))
    positive = point_area > 0
    fill = float(point_area[positive].mean()) if np.any(positive) else 1.0
    point_area = np.where(positive, point_area, fill)
    return point_area.reshape(-1, 1).astype(np.float32)


def compute_cell_area(surface: pv.PolyData) -> np.ndarray:
    area_mesh = surface.compute_cell_sizes(length=False, area=True, volume=False)
    area = np.asarray(area_mesh.cell_data["Area"], dtype=np.float32).reshape(-1, 1)
    return np.maximum(area, 1e-12)


def compute_point_normals(surface: pv.PolyData) -> np.ndarray:
    try:
        normal_mesh = surface.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            inplace=False,
        )
    except Exception:
        normal_mesh = surface.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=False,
            consistent_normals=False,
            inplace=False,
        )
    return normalize_np_vectors(np.asarray(normal_mesh.point_data["Normals"], dtype=np.float32))


def compute_cell_normals(surface: pv.PolyData) -> np.ndarray:
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
    return normalize_np_vectors(np.asarray(normal_mesh.cell_data["Normals"], dtype=np.float32))


def take_data_array(mesh: pv.DataSet, field: str, location: str) -> np.ndarray:
    if location == "cell":
        data = mesh.cell_data
    elif location == "point":
        data = mesh.point_data
    else:
        raise ValueError("location must be either 'cell' or 'point'.")
    if field not in data:
        raise KeyError(f"Field '{field}' not found in {location}_data.")
    return np.asarray(data[field], dtype=np.float32)


def read_drivaer_surface_pressure(vtk_file_path: str | Path) -> np.ndarray:
    mesh = pv.read(vtk_file_path)
    surface = surface_polydata(mesh)
    if "p" in mesh.point_data:
        points = np.asarray(surface.points, dtype=np.float32)
        pressure = np.asarray(surface.point_data["p"], dtype=np.float32).reshape(-1, 1)
        normals = compute_point_normals(surface)
        area = compute_point_area(surface)
    elif "p" in mesh.cell_data:
        points = np.asarray(surface.cell_centers().points, dtype=np.float32)
        pressure = np.asarray(surface.cell_data["p"], dtype=np.float32).reshape(-1, 1)
        normals = compute_cell_normals(surface)
        area = compute_cell_area(surface)
    else:
        raise KeyError(f"No pressure field 'p' found in {vtk_file_path}.")
    return np.concatenate([points, pressure, normals, np.maximum(area, 1e-12)], axis=1).astype(
        np.float32,
        copy=False,
    )


def read_drivaer_volume_velocity(vtk_file_path: str | Path, location: str = "point") -> np.ndarray:
    mesh = pv.read(vtk_file_path)
    if location == "point":
        if "U" not in mesh.point_data:
            raise KeyError(f"No point-data velocity field 'U' found in {vtk_file_path}.")
        points = np.asarray(mesh.points, dtype=np.float32)
        velocity = np.asarray(mesh.point_data["U"], dtype=np.float32)
    elif location == "cell":
        if "U" not in mesh.cell_data:
            raise KeyError(f"No cell-data velocity field 'U' found in {vtk_file_path}.")
        points = np.asarray(mesh.cell_centers().points, dtype=np.float32)
        velocity = np.asarray(mesh.cell_data["U"], dtype=np.float32)
    else:
        raise ValueError("location must be 'point' or 'cell'.")
    return np.concatenate([points, velocity], axis=1).astype(np.float32, copy=False)


def parse_high_speed_case_cond(path_or_name: str | Path) -> tuple[np.ndarray, float]:
    name = Path(path_or_name).name
    match = re.search(r"_Ux_([-+]?\d*\.?\d+)_Uy_([-+]?\d*\.?\d+)(?:$|[._])", name)
    if match is None:
        raise ValueError(f"Cannot parse Ux/Uy from case name: {name}")
    ux = float(match.group(1))
    uy = float(match.group(2))
    speed = float((ux * ux + uy * uy) ** 0.5)
    if speed <= 0.0:
        raise ValueError(f"Non-positive speed parsed from case name: {name}")
    return np.array([ux / 100.0, uy / 20.0], dtype=np.float32), speed


def high_speed_free_stream(path_or_name: str | Path) -> np.ndarray:
    name = Path(path_or_name).name
    match = re.search(r"_Ux_([-+]?\d*\.?\d+)_Uy_([-+]?\d*\.?\d+)(?:$|[._])", name)
    if match is None:
        raise ValueError(f"Cannot parse Ux/Uy from case name: {name}")
    return np.array([float(match.group(1)), float(match.group(2)), 0.0], dtype=np.float32)


def high_speed_type_dir(root_dir: str | Path, train_type: str) -> Path:
    root = Path(root_dir)
    candidates = [train_type, train_type.lower(), train_type.upper()]
    if train_type.lower() == "maglev":
        candidates.extend(["maglev", "Maglev"])
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Cannot find train_type={train_type!r} under {root}.")


def list_high_speed_cases(root_dir: str | Path, train_type: str) -> list[Path]:
    type_dir = high_speed_type_dir(root_dir, train_type)
    cases = sorted(path for path in type_dir.iterdir() if path.is_dir())
    if not cases:
        raise FileNotFoundError(f"No case directories found in {type_dir}.")
    return cases


def read_high_speed_field(vtk_file_path: str | Path, include_geometry: bool, location: str = "cell") -> np.ndarray:
    mesh = pv.read(vtk_file_path)
    if location == "cell":
        points = np.asarray(mesh.cell_centers().points, dtype=np.float32)
        data_location = "cell"
    elif location == "point":
        points = np.asarray(mesh.points, dtype=np.float32)
        data_location = "point"
    else:
        raise ValueError("location must be 'cell' or 'point'.")

    pressure = take_data_array(mesh, "p", data_location).reshape(-1, 1)
    velocity = take_data_array(mesh, "U", data_location).reshape(-1, 3)
    k = take_data_array(mesh, "k", data_location).reshape(-1, 1)
    omega = take_data_array(mesh, "omega", data_location).reshape(-1, 1)
    nut = take_data_array(mesh, "nut", data_location).reshape(-1, 1)
    arrays = [points, pressure, velocity, k, omega, nut]

    if include_geometry:
        surface = surface_polydata(mesh)
        if location == "cell":
            normals = compute_cell_normals(surface)
            area = compute_cell_area(surface)
        else:
            normals = compute_point_normals(surface)
            area = compute_point_area(surface)
        if normals.shape[0] != points.shape[0]:
            normals = np.zeros((points.shape[0], 3), dtype=np.float32)
            area = np.ones((points.shape[0], 1), dtype=np.float32)
        arrays.extend([normals, np.maximum(area, 1e-12)])

    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
