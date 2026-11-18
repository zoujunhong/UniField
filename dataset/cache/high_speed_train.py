from __future__ import annotations

import os
import json
import re
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.utils.load import high_speed_type_dir, list_high_speed_cases, read_high_speed_field
from dataset.utils.sampling import sample_rows_np, voxel_uniform_sample_rows


HIGH_SPEED_TRAIN_ROOT = "/lustre/home/zdhs0017/data/HighSpeedTrain"
HIGH_SPEED_TRAIN_CACHE_ROOT = "/data/home/zdhs0017/zoujunhong/data/HighSpeedTrain"
DEFAULT_SURFACE_CACHE_POINTS = 100_000
DEFAULT_VOLUME_CACHE_POINTS = 300_000
DEFAULT_CACHE_DTYPE = "fp32"
HIGH_SPEED_TRAIN_GEOMETRY_COUNTS = {
    "CRH450": 20,
    "Maglev": 30,
}
HIGH_SPEED_SURFACE_CACHE_COLUMNS = (
    "x",
    "y",
    "z",
    "p",
    "Ux",
    "Uy",
    "Uz",
    "k",
    "omega",
    "nut",
    "nx",
    "ny",
    "nz",
    "area",
)
HIGH_SPEED_VOLUME_CACHE_COLUMNS = (
    "x",
    "y",
    "z",
    "p",
    "Ux",
    "Uy",
    "Uz",
    "k",
    "omega",
    "nut",
    "surface_distance",
)
HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL = 10


def normalized_train_type(train_type: str) -> str:
    return "Maglev" if train_type.lower() == "maglev" else train_type


def parse_high_speed_geo_id(path_or_name: str | Path) -> int | None:
    match = re.search(r"_geo_(\d+)_", Path(path_or_name).name)
    if match is None:
        return None
    return int(match.group(1))


def high_speed_case_sort_key(path_or_name: str | Path) -> tuple[int, int, str]:
    name = Path(path_or_name).name
    geo_id = parse_high_speed_geo_id(name)
    match = re.search(r"_geo_\d+_(\d+)_", name)
    run_id = int(match.group(1)) if match is not None else -1
    return (-1 if geo_id is None else geo_id, run_id, name)


def normalized_cache_dtype(cache_dtype: str) -> str:
    cache_dtype = str(cache_dtype).lower()
    if cache_dtype not in {"fp16", "fp32"}:
        raise ValueError("cache_dtype must be 'fp16' or 'fp32'.")
    return cache_dtype


def cache_numpy_dtype(cache_dtype: str) -> np.dtype:
    cache_dtype = normalized_cache_dtype(cache_dtype)
    return np.dtype(np.float16 if cache_dtype == "fp16" else np.float32)


def cache_dir(
    cache_root: str | Path,
    train_type: str,
    kind: str,
    points: int,
    cache_dtype: str = DEFAULT_CACHE_DTYPE,
    case_name: str | None = None,
) -> str:
    """Return the case collection, or one case's auxiliary profile directory."""
    cache_dtype = normalized_cache_dtype(cache_dtype)
    root = Path(cache_root) / "cases"
    if case_name is None:
        return str(root)
    return str(root / case_name / "auxiliary" / f"{kind}_{int(points)}_{cache_dtype}")


def auxiliary_cache_path(cache_root: str | Path, case_name: str, profile: str, suffix: str) -> str:
    return str(Path(cache_root) / "cases" / case_name / "auxiliary" / profile / f"data{suffix}")


def shared_cache_dir(cache_root: str | Path, train_type: str, profile: str) -> str:
    return str(Path(cache_root) / "shared" / normalized_train_type(train_type) / profile)


def register_auxiliary_cache(path: str | Path, *, arrays=None, shared_decoder: str | Path | None = None) -> None:
    """Index a case-local auxiliary cache without changing the primary field metadata."""
    path = Path(path)
    if path.parent.parent.name != "auxiliary" or path.parents[3].name != "cases":
        raise ValueError(f"Expected cases/<case>/auxiliary/<profile>/file: {path}")
    case_dir = path.parents[2]
    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
        "schema": "field_case_v1", "dataset": "HighSpeedTrain", "case": case_dir.name,
        "train_type": normalized_train_type(case_dir.name.split("_geo_", 1)[0]),
        "geometry_id": parse_high_speed_geo_id(case_dir.name),
        "cache_only": True,
    }
    record = {"file": str(path.relative_to(case_dir))}
    if arrays is not None:
        record["arrays"] = {
            name: {"shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
            for name, value in arrays.items()
        }
    if shared_decoder is not None:
        record["shared_decoder"] = str(Path(shared_decoder).relative_to(case_dir.parent.parent))
    metadata.setdefault("auxiliary", {})[path.parent.name] = record
    temporary = metadata_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)


def cache_path(
    cache_root: str | Path,
    train_type: str,
    kind: str,
    points: int,
    case_name: str,
    cache_dtype: str = DEFAULT_CACHE_DTYPE,
) -> str:
    root = Path(cache_root)
    default_points = {"surface_random": DEFAULT_SURFACE_CACHE_POINTS, "volume_random": DEFAULT_VOLUME_CACHE_POINTS}
    if default_points.get(kind) == int(points) and normalized_cache_dtype(cache_dtype) == "fp32":
        return str(root / "cases" / case_name / ("surface.npy" if kind.startswith("surface") else "volume.npy"))
    return os.path.join(cache_dir(cache_root, train_type, kind, points, cache_dtype, case_name), "data.npy")


def identity_collate(batch):
    return batch


def valid_npy_cache(path: str | Path, min_cols: int = 1, dtype: str | np.dtype | None = None) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < int(min_cols):
            return False
        if dtype is not None and arr.dtype != np.dtype(dtype):
            return False
    except Exception:
        return False
    return True


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        os.replace(tmp_path, path)
        if path.name in {"surface.npy", "volume.npy"} and path.parent.parent.name == "cases":
            from dataset.utils.load import parse_high_speed_case_cond, high_speed_free_stream
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            case = path.parent.name
            cond, _ = parse_high_speed_case_cond(case)
            kind = path.stem
            columns = HIGH_SPEED_SURFACE_CACHE_COLUMNS if kind == "surface" else HIGH_SPEED_VOLUME_CACHE_COLUMNS
            metadata.update(schema="field_case_v1", dataset="HighSpeedTrain", case=case,
                            train_type=normalized_train_type(case.split("_geo_", 1)[0]),
                            geometry_id=parse_high_speed_geo_id(case), condition=cond.tolist(),
                            free_stream=high_speed_free_stream(case).tolist())
            metadata.pop("cache_only", None)
            metadata[kind] = {"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                              "columns": list(columns[:arr.shape[1]]), "pressure_kind": "kinematic_p"}
            temporary = metadata_path.with_suffix(f".json.tmp.{os.getpid()}")
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(metadata_path)
        elif path.parent.parent.name == "auxiliary":
            register_auxiliary_cache(path, arrays={"data": arr})
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _surface_cache_path_for_distance(
    cache_root: str | Path,
    train_type: str,
    surface_cache_points: int,
    case_name: str,
    cache_dtype: str,
) -> str:
    for kind in ("surface_uniform", "surface_random"):
        path = cache_path(cache_root, train_type, kind, surface_cache_points, case_name, cache_dtype)
        if valid_npy_cache(path, min_cols=3, dtype=cache_numpy_dtype(cache_dtype)):
            return path
    raise FileNotFoundError(
        "Missing HighSpeedTrain surface cache for distance computation: "
        f"{normalized_train_type(train_type)}/{case_name}, points={int(surface_cache_points)}, dtype={cache_dtype}."
    )


def _full_surface_points_for_distance(
    root_dir: str | Path,
    train_type: str,
    case_name: str,
) -> tuple[str, np.ndarray]:
    import pyvista as pv

    surface_path = high_speed_type_dir(root_dir, train_type) / case_name / "surface.vtk"
    if not surface_path.is_file() or surface_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing HighSpeedTrain surface VTK for distance computation: {surface_path}")

    mesh = pv.read(surface_path)
    if mesh.n_cells > 0:
        points = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    else:
        points = np.asarray(mesh.points, dtype=np.float32)
    return str(surface_path), points


def append_surface_distance_to_volume_cache(
    cache_root: str | Path,
    train_type: str,
    case_name: str,
    root_dir: str = HIGH_SPEED_TRAIN_ROOT,
    surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
    volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
    cache_dtype: str = DEFAULT_CACHE_DTYPE,
    clip: float | None = None,
    overwrite: bool = False,
    surface_distance_source: str = "surface_vtk",
) -> dict[str, str | int | float | bool]:
    """Append or refresh volume-cache column 10 with distance to the case surface."""
    from scipy.spatial import cKDTree

    if surface_distance_source not in {"surface_vtk", "surface_cache"}:
        raise ValueError("surface_distance_source must be 'surface_vtk' or 'surface_cache'.")

    cache_dtype = normalized_cache_dtype(cache_dtype)
    cache_np_dtype = cache_numpy_dtype(cache_dtype)
    volume_path = cache_path(cache_root, train_type, "volume_random", volume_cache_points, case_name, cache_dtype)
    if not valid_npy_cache(volume_path, min_cols=10, dtype=cache_np_dtype):
        raise FileNotFoundError(f"Missing or invalid HighSpeedTrain volume cache: {volume_path}")

    volume = np.load(volume_path).astype(np.float32, copy=False)
    if volume.shape[1] > HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL and not overwrite:
        return {
            "case_name": case_name,
            "status": "exists",
            "path": volume_path,
            "rows": int(volume.shape[0]),
            "cols": int(volume.shape[1]),
        }

    if surface_distance_source == "surface_vtk":
        surface_path, surface_points = _full_surface_points_for_distance(root_dir, train_type, case_name)
    else:
        surface_path = _surface_cache_path_for_distance(cache_root, train_type, surface_cache_points, case_name, cache_dtype)
        surface = np.load(surface_path, mmap_mode="r")
        surface_points = np.asarray(surface[:, :3], dtype=np.float32)

    tree = cKDTree(surface_points)
    distance = tree.query(np.asarray(volume[:, :3], dtype=np.float32), workers=-1)[0].astype(np.float32)
    if clip is not None:
        distance = np.minimum(distance, float(clip)).astype(np.float32, copy=False)
    distance = distance.reshape(-1, 1)

    if volume.shape[1] > HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL:
        updated = volume.copy()
        updated[:, HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL : HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL + 1] = distance
    else:
        updated = np.concatenate([volume, distance], axis=1)
    atomic_save_npy(volume_path, updated.astype(cache_np_dtype, copy=False))
    return {
        "case_name": case_name,
        "status": "updated",
        "path": volume_path,
        "surface_path": surface_path,
        "surface_distance_source": surface_distance_source,
        "rows": int(updated.shape[0]),
        "cols": int(updated.shape[1]),
        "distance_min": float(distance.min()) if distance.size else 0.0,
        "distance_max": float(distance.max()) if distance.size else 0.0,
        "distance_mean": float(distance.mean()) if distance.size else 0.0,
    }


def complete_high_speed_raw_case(case_dir: str | Path, build_surface: bool, build_volume: bool) -> bool:
    case_dir = Path(case_dir)
    if build_surface:
        surface_path = case_dir / "surface.vtk"
        if not surface_path.is_file() or surface_path.stat().st_size <= 0:
            return False
    if build_volume:
        volume_path = case_dir / "full_field.vtk"
        if not volume_path.is_file() or volume_path.stat().st_size <= 0:
            return False
    return True


class HighSpeedTrainCacheBuilder(Dataset):
    """
    Build dtype-selected caches for high_speed_train cases.

    Surface rows:
        [x, y, z, p, Ux, Uy, Uz, k, omega, nut, nx, ny, nz, area]
    Volume rows:
        [x, y, z, p, Ux, Uy, Uz, k, omega, nut, surface_distance]
    """

    def __init__(
        self,
        root_dir: str = HIGH_SPEED_TRAIN_ROOT,
        train_type: str = "CRH450",
        cache_root: str | None = None,
        surface_cache_points: int = DEFAULT_SURFACE_CACHE_POINTS,
        volume_cache_points: int = DEFAULT_VOLUME_CACHE_POINTS,
        surface_sampling: str = "random",
        cache_dtype: str = DEFAULT_CACHE_DTYPE,
        overwrite: bool = False,
        build_surface: bool = True,
        build_volume: bool = True,
        build_volume_surface_distance: bool = True,
        surface_distance_clip: float | None = None,
        surface_distance_source: str = "surface_vtk",
        skip_incomplete: bool = True,
    ) -> None:
        if surface_sampling not in {"random", "uniform"}:
            raise ValueError("surface_sampling must be 'random' or 'uniform'.")
        if surface_distance_source not in {"surface_vtk", "surface_cache"}:
            raise ValueError("surface_distance_source must be 'surface_vtk' or 'surface_cache'.")
        self.root_dir = root_dir
        self.train_type = normalized_train_type(train_type)
        self.cache_root = cache_root or HIGH_SPEED_TRAIN_CACHE_ROOT
        self.surface_cache_points = int(surface_cache_points)
        self.volume_cache_points = int(volume_cache_points)
        self.surface_sampling = surface_sampling
        self.cache_dtype = normalized_cache_dtype(cache_dtype)
        self.cache_np_dtype = cache_numpy_dtype(self.cache_dtype)
        self.overwrite = bool(overwrite)
        self.build_surface = bool(build_surface)
        self.build_volume = bool(build_volume)
        self.build_volume_surface_distance = bool(build_volume_surface_distance)
        self.surface_distance_clip = None if surface_distance_clip is None else float(surface_distance_clip)
        self.surface_distance_source = surface_distance_source
        self.skip_incomplete = bool(skip_incomplete)
        if not self.build_surface and not self.build_volume:
            raise ValueError("At least one of build_surface/build_volume must be True.")

        case_dirs = sorted(list_high_speed_cases(self.root_dir, self.train_type), key=high_speed_case_sort_key)
        if self.skip_incomplete:
            need_raw_surface = self.build_surface or (
                self.build_volume and self.build_volume_surface_distance and self.surface_distance_source == "surface_vtk"
            )
            complete_case_dirs = [
                path
                for path in case_dirs
                if complete_high_speed_raw_case(path, need_raw_surface, self.build_volume)
            ]
            skipped = len(case_dirs) - len(complete_case_dirs)
            if skipped > 0:
                print(
                    f"HighSpeedTrain cache builder: skipped {skipped} incomplete raw cases "
                    f"for {self.train_type}.",
                    flush=True,
                )
            case_dirs = complete_case_dirs
        self.case_dirs = case_dirs
        for kind, points in [
            (f"surface_{self.surface_sampling}", self.surface_cache_points),
            ("volume_random", self.volume_cache_points),
        ]:
            if kind.startswith("surface") and not self.build_surface:
                continue
            if kind.startswith("volume") and not self.build_volume:
                continue
            os.makedirs(cache_dir(self.cache_root, self.train_type, kind, points, self.cache_dtype), exist_ok=True)
        print(f"HighSpeedTrain cache builder: {self.train_type}, {len(self.case_dirs)} cases, dtype={self.cache_dtype}.")

    def __len__(self) -> int:
        return len(self.case_dirs)

    def _read_case_field(self, case_dir: Path, file_name: str, include_geometry: bool) -> np.ndarray | None:
        try:
            return read_high_speed_field(case_dir / file_name, include_geometry=include_geometry, location="cell")
        except Exception as exc:
            if not self.skip_incomplete:
                raise
            print(
                f"HighSpeedTrain cache builder: skipped incomplete case {case_dir.name}: {exc}",
                flush=True,
            )
            return None

    def __getitem__(self, idx: int) -> dict[str, str]:
        case_dir = self.case_dirs[idx]
        rng = np.random.default_rng(idx)
        written: dict[str, str] = {}

        if self.build_surface:
            kind = f"surface_{self.surface_sampling}"
            out_path = cache_path(self.cache_root, self.train_type, kind, self.surface_cache_points, case_dir.name, self.cache_dtype)
            if self.overwrite or not os.path.exists(out_path):
                rows = self._read_case_field(case_dir, "surface.vtk", include_geometry=True)
                if rows is not None:
                    if self.surface_sampling == "uniform":
                        rows = voxel_uniform_sample_rows(rows, self.surface_cache_points, rng)
                    else:
                        rows = sample_rows_np(rows, self.surface_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(self.cache_np_dtype))
            elif not valid_npy_cache(out_path, min_cols=14, dtype=self.cache_np_dtype):
                rows = self._read_case_field(case_dir, "surface.vtk", include_geometry=True)
                if rows is not None:
                    if self.surface_sampling == "uniform":
                        rows = voxel_uniform_sample_rows(rows, self.surface_cache_points, rng)
                    else:
                        rows = sample_rows_np(rows, self.surface_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(self.cache_np_dtype))
            if valid_npy_cache(out_path, min_cols=14, dtype=self.cache_np_dtype):
                written[kind] = out_path

        if self.build_volume:
            out_path = cache_path(self.cache_root, self.train_type, "volume_random", self.volume_cache_points, case_dir.name, self.cache_dtype)
            if self.overwrite or not os.path.exists(out_path):
                rows = self._read_case_field(case_dir, "full_field.vtk", include_geometry=False)
                if rows is not None:
                    rows = sample_rows_np(rows, self.volume_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(self.cache_np_dtype))
            elif not valid_npy_cache(out_path, min_cols=10, dtype=self.cache_np_dtype):
                rows = self._read_case_field(case_dir, "full_field.vtk", include_geometry=False)
                if rows is not None:
                    rows = sample_rows_np(rows, self.volume_cache_points, rng)
                    atomic_save_npy(out_path, rows.astype(self.cache_np_dtype))
            if self.build_volume_surface_distance and valid_npy_cache(out_path, min_cols=10, dtype=self.cache_np_dtype):
                try:
                    append_surface_distance_to_volume_cache(
                        self.cache_root,
                        self.train_type,
                        case_dir.name,
                        root_dir=self.root_dir,
                        surface_cache_points=self.surface_cache_points,
                        volume_cache_points=self.volume_cache_points,
                        cache_dtype=self.cache_dtype,
                        clip=self.surface_distance_clip,
                        overwrite=self.overwrite,
                        surface_distance_source=self.surface_distance_source,
                    )
                except FileNotFoundError as exc:
                    if not self.skip_incomplete:
                        raise
                    print(
                        f"HighSpeedTrain cache builder: skipped surface distance for {case_dir.name}: {exc}",
                        flush=True,
                    )
            min_cols = HIGH_SPEED_VOLUME_SURFACE_DISTANCE_COL + 1 if self.build_volume_surface_distance else 10
            if valid_npy_cache(out_path, min_cols=min_cols, dtype=self.cache_np_dtype):
                written["volume_random"] = out_path
        return written

    def build_cache(
        self,
        start_index: int = 0,
        limit: int | None = None,
        log_every: int = 1,
        num_workers: int = 0,
    ) -> None:
        end = len(self) if limit is None else min(len(self), int(start_index) + int(limit))
        indices = list(range(int(start_index), end))
        if num_workers <= 0:
            for i in indices:
                self[i]
                if log_every > 0 and ((i - start_index + 1) % log_every == 0 or i + 1 == end):
                    print(f"[{i + 1}/{len(self)}] processed {self.case_dirs[i].name}", flush=True)
            return

        subset = Subset(self, indices)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=identity_collate,
        )
        for local_i, _ in enumerate(loader, start=1):
            global_i = int(start_index) + local_i - 1
            if log_every > 0 and (local_i % log_every == 0 or global_i + 1 == end):
                print(f"[{global_i + 1}/{len(self)}] processed {self.case_dirs[global_i].name}", flush=True)
