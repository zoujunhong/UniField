"""GeometryGen low-subsonic surface-Cp dataset and cache builder.

The CFD campaign stores geometry after applying its incidence rotation while
the OpenFOAM free stream remains ``(U, 0, 0)``.  This reader caches the inverse
representation used for learning: canonical (zero-incidence) geometry and the
equivalent three-component body-frame velocity ``U_inf_2``.

Training caches contain two deliberately different point sets:

* ``surface_input``: a fixed 8192-point farthest-point sample (xyz + normal),
* ``query_pool``: native CFD face centres (xyz + Cp), sampled randomly at read
  time to form 8192 pressure queries.

Open3D is required only while building caches.  Regular training workers need
only NumPy and PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from dataset.validation import pressure_valid_rows
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "GEOMETRYGEN_SURFACE_ROOT",
        "/lustre/home/zdhs0017/data/GeometryGen/CanonicalSurface/cases",
    )
)
DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "GEOMETRYGEN_SURFACE_CACHE_ROOT",
        Path.home() / "zoujunhong/data/GeometryGen/surface",
    )
)
INDEX_NAME = "index.json"
CACHE_SCHEMA_VERSION = 2


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _stable_seed(name: str, seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(seed)) % (2**63 - 1)


def _rotation(metadata: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(metadata["openfoam"]["geometry"]["transform"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 geometry transform, got {transform.shape}.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6):
        raise ValueError("Geometry transform does not contain an orthonormal rotation.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-6):
        raise ValueError("Geometry transform rotation is not proper.")
    return rotation


def _canonical_surface(
    source_path: Path,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical vertices/faces/centres/normals/area/Cp.

    For row vectors, multiplying by the CFD rotation ``R`` applies ``R^T``
    to the corresponding column vector.  This exactly undoes the body's CFD
    rotation without introducing a second, sample-dependent recentering.

    The archived physical pressure is converted through kinematic gauge
    pressure so the target follows the requested convention
    ``Cp = p / (0.5 * ||U_inf_2||^2)`` where ``p=(P-P_inf)/rho``.
    """

    with np.load(source_path, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
        centers = np.asarray(archive["cell_centers"], dtype=np.float64)
        normals = np.asarray(archive["normals"], dtype=np.float64)
        area = np.asarray(archive["face_area"], dtype=np.float64).reshape(-1)
        pressure_pa = np.asarray(archive["pressure_pa"], dtype=np.float64).reshape(-1)
        archived_cp = np.asarray(archive["cp"], dtype=np.float64).reshape(-1)

    cell_count = faces.shape[0]
    expected = {
        "faces": (cell_count, 3),
        "cell_centers": (cell_count, 3),
        "normals": (cell_count, 3),
        "face_area": (cell_count,),
        "pressure_pa": (cell_count,),
        "cp": (cell_count,),
    }
    actual = {
        "faces": faces.shape,
        "cell_centers": centers.shape,
        "normals": normals.shape,
        "face_area": area.shape,
        "pressure_pa": pressure_pa.shape,
        "cp": archived_cp.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"Invalid {name} shape in {source_path}: {actual[name]} != {shape}.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point array in {source_path}: {points.shape}.")

    freestream = metadata["openfoam"]["freestream"]
    rho = float(freestream["density"])
    speed = float(freestream["speed"])
    p_inf = float(freestream["pressure"])
    if rho <= 0.0 or speed <= 0.0:
        raise ValueError(f"Invalid freestream rho/speed in {source_path}.")
    cp = ((pressure_pa - p_inf) / rho) / (0.5 * speed**2)
    # pressure_pa and archived Cp are both float32 products.  At the lowest
    # Mach numbers their subtraction amplifies rounding, so use a validation
    # tolerance that is still far below the learning error scale.
    if not np.allclose(cp, archived_cp, rtol=1e-4, atol=1e-4):
        difference = float(np.max(np.abs(cp - archived_cp)))
        raise ValueError(f"Recomputed Cp disagrees with archive in {source_path}: max |dCp|={difference:g}.")

    valid = (
        np.all(np.isfinite(centers), axis=1)
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(area)
        & np.isfinite(cp)
        & (area > 0.0)
        & np.all((faces >= 0) & (faces < points.shape[0]), axis=1)
    )
    if not np.any(valid):
        raise ValueError(f"No valid surface cells in {source_path}.")

    rotation = _rotation(metadata)
    canonical_points = points @ rotation
    canonical_centers = centers[valid] @ rotation
    canonical_normals = normals[valid] @ rotation
    canonical_normals /= np.maximum(np.linalg.norm(canonical_normals, axis=1, keepdims=True), 1e-12)
    return (
        canonical_points,
        faces[valid],
        canonical_centers,
        canonical_normals,
        area[valid],
        cp[valid],
    )


def _dense_surface_candidates(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    area: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(area, dtype=np.float64)
    probability /= probability.sum()
    selected = rng.choice(faces.shape[0], size=int(count), replace=True, p=probability)
    triangles = points[faces[selected]]
    root_u = np.sqrt(rng.random(int(count)))
    v = rng.random(int(count))
    xyz = (
        (1.0 - root_u)[:, None] * triangles[:, 0]
        + (root_u * (1.0 - v))[:, None] * triangles[:, 1]
        + (root_u * v)[:, None] * triangles[:, 2]
    )
    return xyz.astype(np.float32), normals[selected].astype(np.float32)


def _farthest_point_indices(xyz: np.ndarray, count: int) -> np.ndarray:
    """Run exact Open3D FPS and map the selected coordinates to source rows."""

    try:
        import open3d as o3d
        from scipy.spatial import cKDTree
    except ImportError as error:  # pragma: no cover - cache-build dependency
        raise ImportError(
            "Building GeometryGenSurface caches requires Open3D and SciPy. "
            "Use the zoujunhong environment on a compute node."
        ) from error

    xyz = np.asarray(xyz, dtype=np.float64)
    count = int(count)
    if xyz.shape[0] < count:
        raise ValueError(f"FPS needs at least {count} candidates, got {xyz.shape[0]}.")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    selected_cloud = cloud.farthest_point_down_sample(count)
    selected_xyz = np.asarray(selected_cloud.points)
    distance, indices = cKDTree(xyz).query(selected_xyz, k=1, workers=-1)
    tolerance = max(1e-8, 2e-7 * float(np.ptp(xyz, axis=0).max()))
    if float(np.max(distance)) > tolerance:
        raise RuntimeError(f"Could not map Open3D FPS output to candidates: max distance={distance.max():g}.")
    indices = np.asarray(indices, dtype=np.int64)
    if np.unique(indices).size != count:
        raise RuntimeError("Open3D FPS returned duplicate source points.")
    return indices


def _velocity_body_frame(metadata: dict[str, Any]) -> np.ndarray:
    velocity_cfd = np.asarray(metadata["openfoam"]["directions"]["velocity"], dtype=np.float64)
    if velocity_cfd.shape != (3,):
        raise ValueError(f"Expected a three-component CFD velocity, got {velocity_cfd.shape}.")
    velocity_body = velocity_cfd @ _rotation(metadata)
    speed = float(metadata["openfoam"]["freestream"]["speed"])
    if not math.isclose(float(np.linalg.norm(velocity_body)), speed, rel_tol=2e-6, abs_tol=2e-6):
        raise ValueError("Inverse-rotated velocity does not preserve freestream speed.")
    return velocity_body.astype(np.float32)


def _case_directories(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"GeometryGen campaign case directory is missing: {source_root}")
    return [
        path
        for path in sorted(source_root.glob("case_*"))
        if (path / "metadata.json").is_file() and (path / "surface_fields.npz").is_file()
    ]


class GeometryGenSurfaceCacheBuilder:
    def __init__(
        self,
        source_root: str | Path = DEFAULT_SOURCE_ROOT,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
    ) -> None:
        self.source_root = Path(source_root)
        self.cache_root = Path(cache_root)

    def build(
        self,
        *,
        input_points: int = 8192,
        fps_candidates: int = 32768,
        query_pool_points: int = 32768,
        seed: int = 200099,
        overwrite: bool = False,
        limit: int | None = None,
        continue_on_error: bool = False,
        progress_interval: int = 25,
    ) -> tuple[int, int, int]:
        if input_points <= 0:
            raise ValueError("input_points must be positive.")
        if fps_candidates < input_points:
            raise ValueError("fps_candidates must be >= input_points.")
        if query_pool_points <= 0:
            raise ValueError("query_pool_points must be positive.")
        directories = _case_directories(self.source_root)
        if limit is not None:
            directories = directories[: int(limit)]
        built = skipped = failed = 0
        records: list[dict[str, Any]] = []
        for position, case_dir in enumerate(directories, start=1):
            case_name = case_dir.name
            target = self.cache_root / "cases" / case_name / "surface.npy"
            try:
                metadata = _read_json(case_dir / "metadata.json")
                if overwrite or not target.is_file():
                    points, faces, centers, normals, area, cp = _canonical_surface(
                        case_dir / "surface_fields.npz", metadata
                    )
                    case_seed = _stable_seed(case_name, seed)
                    rng = np.random.default_rng(case_seed)
                    candidate_xyz, candidate_normals = _dense_surface_candidates(
                        points, faces, normals, area, fps_candidates, rng
                    )
                    fps_indices = _farthest_point_indices(candidate_xyz, input_points)
                    surface_input = np.concatenate(
                        [candidate_xyz[fps_indices], candidate_normals[fps_indices]], axis=1
                    ).astype(np.float32, copy=False)
                    if centers.shape[0] > query_pool_points:
                        query_rng = np.random.default_rng((case_seed + 1) % (2**63 - 1))
                        query_indices = query_rng.choice(
                            centers.shape[0], size=query_pool_points, replace=False
                        )
                        centers = centers[query_indices]
                        cp = cp[query_indices]
                    query_pool = np.concatenate(
                        [centers.astype(np.float32), cp[:, None].astype(np.float32)], axis=1
                    )
                    cond = _velocity_body_frame(metadata)
                    from dataset.GeometryGenSurfaceVolume import atomic_npy
                    atomic_npy(target, query_pool)
                    atomic_npy(target.parent / "input.npy", surface_input)
                    built += 1
                else:
                    surface_input = np.load(target.parent / "input.npy", allow_pickle=False)
                    query_pool = np.load(target, allow_pickle=False)
                    cond = _velocity_body_frame(metadata)
                    if surface_input.shape != (input_points, 6):
                        raise ValueError(f"Existing cache has wrong surface_input shape: {surface_input.shape}.")
                    if query_pool.ndim != 2 or query_pool.shape[1] != 4 or query_pool.shape[0] == 0:
                        raise ValueError(f"Existing cache has wrong query_pool shape: {query_pool.shape}.")
                    if query_pool.shape[0] > query_pool_points:
                        raise ValueError(
                            f"Existing query pool has {query_pool.shape[0]} points, above the "
                            f"configured cap {query_pool_points}; rebuild with --overwrite."
                        )
                    if cond.shape != (3,):
                        raise ValueError(f"Existing cache has wrong cond shape: {cond.shape}.")
                    skipped += 1

                case = metadata["case"]
                records.append(
                    {
                        "case_id": case_name,
                        "geometry_id": f"geometry_{int(metadata['geometry_index']):05d}",
                        "cache": str(Path("cases") / case_name / "surface.npy"),
                        "input_cache": str(Path("cases") / case_name / "input.npy"),
                        "condition": [float(value) for value in cond],
                        "mach": float(case["mach"]),
                        "alpha_deg": float(case["alpha_deg"]),
                        "beta_deg": float(case["beta_deg"]),
                        "velocity_body_mps": [float(value) for value in cond],
                        "query_pool_points": int(query_pool.shape[0]),
                    }
                )
                _atomic_json(target.parent / "metadata.json", dict(records[-1], columns=["x", "y", "z", "Cp"], source=str(case_dir)))
            except Exception as error:
                failed += 1
                if not continue_on_error:
                    raise
                print(f"GeometryGenSurface cache skipped {case_name}: {error}", flush=True)
            if progress_interval and (position % int(progress_interval) == 0 or position == len(directories)):
                print(
                    f"GeometryGenSurface cache: {position}/{len(directories)} "
                    f"(built={built}, reused={skipped}, failed={failed})",
                    flush=True,
                )

        index = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_root": str(self.source_root.resolve()),
            "case_count": len(records),
            "input_points": int(input_points),
            "fps_candidates": int(fps_candidates),
            "query_pool_points": int(query_pool_points),
            "input_fields": ["xyz", "normal"],
            "target": "Cp",
            "cp_definition": "((P-P_inf)/rho)/(0.5*||U_inf_2||^2)",
            "coordinate_frame": "canonical geometry; inverse of archived CFD body rotation",
            "condition": "U_inf_2 body-frame velocity [Ux, Uy, Uz] in m/s",
            "cases": records,
        }
        _atomic_json(self.cache_root / INDEX_NAME, index)
        return built, skipped, failed


class GeometryGenSurfaceDataset(TorchDataset):
    """Surface-only query dataset backed exclusively by compact cache files."""

    dataset_name = "GeometryGenSurface"

    def __init__(
        self,
        *,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
        split: str = "train",
        val_ratio: float = 0.1,
        num_surface_points: int = 8192,
        num_surface_query_points: int = 8192,
        num_query_points: int = 0,
        surface_input_list: tuple[str, ...] | list[str] = ("xyz", "normal"),
        surface_target_list: tuple[str, ...] | list[str] = ("p",),
        volume_target_list: tuple[str, ...] | list[str] = (),
        use_surface: bool = True,
        use_volume: bool = False,
        repeat: int = 1,
        deterministic: bool = False,
        seed: int = 200099,
        route: int = 0,
        limit: int | None = None,
        excluded_case_ids: tuple[str, ...] | list[str] = (),
        exclude_ids_file: str | Path | None = None,
        **_: Any,
    ) -> None:
        self.cache_root = Path(cache_root)
        index_path = self.cache_root / INDEX_NAME
        if not index_path.is_file():
            raise FileNotFoundError(
                f"GeometryGenSurface cache index is missing: {index_path}. "
                "Run `python -m dataset.GeometryGenSurface` first."
            )
        index = _read_json(index_path)
        if int(index.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported GeometryGenSurface cache schema in {index_path}.")
        if not use_surface or use_volume:
            raise ValueError("GeometryGenSurface is a surface-only dataset.")
        if list(surface_input_list) != ["xyz", "normal"]:
            raise ValueError("GeometryGenSurface requires surface_input_list=['xyz', 'normal'].")
        if list(surface_target_list) != ["p"]:
            raise ValueError("GeometryGenSurface requires surface_target_list=['p']; this field stores Cp.")
        if list(volume_target_list):
            raise ValueError("GeometryGenSurface has no volume targets.")
        if int(num_query_points) != 0:
            raise ValueError("GeometryGenSurface has no volume queries; num_query_points must be 0.")
        if int(num_surface_points) != int(index["input_points"]):
            raise ValueError(
                f"Requested {num_surface_points} encoder points, cache contains {index['input_points']}."
            )
        if int(num_surface_query_points) <= 0:
            raise ValueError("num_surface_query_points must be positive.")
        if int(repeat) <= 0:
            raise ValueError("repeat must be positive.")

        records = sorted(index.get("cases", []), key=lambda item: item["case_id"])
        excluded = {str(case_id) for case_id in excluded_case_ids}
        if exclude_ids_file is not None:
            exclude_path = Path(exclude_ids_file)
            if not exclude_path.is_absolute():
                exclude_path = PROJECT_ROOT / exclude_path
            if not exclude_path.is_file():
                raise FileNotFoundError(f"GeometryGenSurface exclusion list is missing: {exclude_path}")
            excluded.update(
                line.strip()
                for line in exclude_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        known_ids = {record["case_id"] for record in records}
        unknown_exclusions = sorted(excluded.difference(known_ids))
        if unknown_exclusions:
            raise ValueError(f"Unknown GeometryGenSurface exclusions: {unknown_exclusions}.")
        records = [record for record in records if record["case_id"] not in excluded]
        split = str(split).lower()
        if split not in {"all", "train", "val", "test"}:
            raise ValueError("split must be one of all/train/val/test.")
        if split != "all":
            if not 0.0 < float(val_ratio) < 1.0:
                raise ValueError("val_ratio must lie in (0, 1).")
            held_out = max(1, int(round(len(records) * float(val_ratio))))
            records = records[:-held_out] if split == "train" else records[-held_out:]
        if limit is not None:
            records = records[: int(limit)]
        if not records:
            raise RuntimeError(f"GeometryGenSurface split {split!r} is empty.")
        missing = [record["case_id"] for record in records if not (self.cache_root / record["cache"]).is_file()]
        if missing:
            raise FileNotFoundError(f"GeometryGenSurface is missing {len(missing)} cache files; first={missing[0]}.")

        self.records = records
        self.case_names = [record["case_id"] for record in records]
        self.num_surface_points = int(num_surface_points)
        self.num_surface_query_points = int(num_surface_query_points)
        self.num_query_points = 0
        self.surface_input_list = ["xyz", "normal"]
        self.surface_target_list = ["p"]
        self.volume_target_list: list[str] = []
        self.coefficient_target_list: list[str] = []
        self.repeat = int(repeat)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.route = int(route)
        self.excluded_case_ids = tuple(sorted(excluded))
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
        print(
            f"GeometryGenSurface: {len(self.records)} cases, split={split}, "
            f"excluded={len(self.excluded_case_ids)}, input={self.num_surface_points} FPS, "
            f"query={self.num_surface_query_points} random.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int):
        record = self.records[index % len(self.records)]
        path = self.cache_root / record["cache"]
        if path.suffix == ".npy":
            surface_input = np.load(self.cache_root / record["input_cache"], allow_pickle=False)
            query_pool = np.load(path, mmap_mode="r", allow_pickle=False)
            cond = np.asarray(record["condition"], dtype=np.float32)
        else:
            # Explicit older cache roots remain readable during migration.
            with np.load(path, allow_pickle=False) as cache:
                surface_input = np.asarray(cache["surface_input"], dtype=np.float32)
                query_pool = np.asarray(cache["query_pool"], dtype=np.float32)
                cond = np.asarray(cache["cond"], dtype=np.float32)
        rng = np.random.default_rng(self.seed + int(index)) if self.deterministic else np.random.default_rng(int(np.random.randint(0, 2**32, dtype=np.uint32)))
        query_pool = pressure_valid_rows(self, query_pool, query_pool[:, 3])
        query_indices = rng.choice(
            query_pool.shape[0],
            size=self.num_surface_query_points,
            replace=query_pool.shape[0] < self.num_surface_query_points,
        )
        query = query_pool[query_indices]
        return (
            torch.from_numpy(surface_input.copy()),
            torch.from_numpy(query[:, :3].copy()),
            torch.from_numpy(query[:, 3:4].copy()),
            torch.empty((0, 3), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            torch.from_numpy(cond.copy()),
            torch.tensor(self.route, dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )


Dataset = GeometryGenSurfaceDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GeometryGen surface-Cp FPS/query caches.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--input-points", type=int, default=8192)
    parser.add_argument("--fps-candidates", type=int, default=32768)
    parser.add_argument("--query-pool-points", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=200099)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    built, reused, failed = GeometryGenSurfaceCacheBuilder(
        args.source_root,
        args.cache_root,
    ).build(
        input_points=args.input_points,
        fps_candidates=args.fps_candidates,
        query_pool_points=args.query_pool_points,
        seed=args.seed,
        overwrite=args.overwrite,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
        progress_interval=args.progress_interval,
    )
    print(
        f"GeometryGenSurface cache complete: built={built}, reused={reused}, failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "Dataset",
    "GeometryGenSurfaceCacheBuilder",
    "GeometryGenSurfaceDataset",
]
