"""Rocket surface-pressure dataset produced by the AutoCFD campaign.

The repository stores compact manifests and NumPy training caches.  The large
OpenFOAM-derived source tree remains configurable through ``ROCKET_CFD_ROOT``.
Pressure targets named ``p`` in the common UniField interface are dimensionless
surface pressure coefficients, ``Cp``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset.cache.aerodynamic_surface import write_surface_cache
from dataset.utils.aerodynamic_surface import AerodynamicSurfaceCase, surface_cache_path
from dataset.utils.aerodynamic_surface_dataset import AerodynamicSurfaceOnlyDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROCKET_CFD_DATA_DIR = Path("/lustre/home/zdhs0017/data/RocketCFD")
ROCKET_CFD_ROOT = Path(os.environ.get("ROCKET_CFD_ROOT", ROCKET_CFD_DATA_DIR / "source"))
ROCKET_CFD_MANIFEST = Path(
    os.environ.get("ROCKET_CFD_MANIFEST", ROCKET_CFD_DATA_DIR / "manifests" / "cases.csv")
)
ROCKET_CFD_FORCE_MANIFEST = Path(
    os.environ.get(
        "ROCKET_CFD_FORCE_MANIFEST",
        ROCKET_CFD_DATA_DIR / "manifests" / "force_supervision.csv",
    )
)
ROCKET_CFD_CACHE_ROOT = Path(
    os.environ.get("ROCKET_CFD_CACHE_ROOT", "/data/home/zdhs0017/zoujunhong/data")
)

ROCKET_CFD_SUBSETS = (
    "train",
    "aoa_interpolation",
    "aoa_extrapolation",
    "geometry_generalization",
    "ood_geometry",
)
ROCKET_CFD_EXCLUDED_GEOMETRIES = frozenset({"g020", "g050"})
ROCKET_CFD_QUALITY_POLICIES = {
    "usable": frozenset({"A_converged", "B_quasi_steady"}),
    "converged_only": frozenset({"A_converged"}),
    "all_completed": None,
}


def _quality_classes(policy: str) -> frozenset[str] | None:
    try:
        return ROCKET_CFD_QUALITY_POLICIES[str(policy)]
    except KeyError as error:
        raise ValueError(
            f"quality_policy must be one of {sorted(ROCKET_CFD_QUALITY_POLICIES)}, got {policy!r}."
        ) from error


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"RocketCFD manifest is missing: {path}. Run `python -m dataset.cache.rocket_manifest` first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "case_id",
        "geometry_id",
        "dataset_split",
        "mach",
        "aoa_deg",
        "convergence_class",
        "case_relpath",
    }
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"RocketCFD manifest {path} is missing columns {sorted(missing)}.")
    return rows


def _read_force_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "Cd", "Cl", "Aref", "force_supervision_eligible"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"RocketCFD force manifest {path} is missing columns {sorted(missing)}.")
    return {row["case_id"]: row for row in rows}


def rocket_cfd_cases(
    root_dir: str | Path = ROCKET_CFD_ROOT,
    *,
    manifest_path: str | Path = ROCKET_CFD_MANIFEST,
    force_manifest_path: str | Path = ROCKET_CFD_FORCE_MANIFEST,
    subset: str = "train",
    quality_policy: str = "all_completed",
    excluded_geometry_ids: Iterable[str] = ROCKET_CFD_EXCLUDED_GEOMETRIES,
    condition_representation: str = "canonical",
    force_consistency_only: bool = False,
) -> list[AerodynamicSurfaceCase]:
    """Discover one explicit, non-overlapping RocketCFD split.

    ``canonical`` rotates each body by its AoA in the dataset wrapper, while
    the two external conditions are ``[3.4 * Mach, flight_placement=1]``.  This
    matches the existing UniField cross-domain conditioning convention.
    """

    if subset not in {*ROCKET_CFD_SUBSETS, "all", "val", "test"}:
        raise ValueError(f"subset must be one of {ROCKET_CFD_SUBSETS} or 'all', got {subset!r}.")
    if condition_representation not in {"canonical", "mach_aoa"}:
        raise ValueError("condition_representation must be 'canonical' or 'mach_aoa'.")
    selected_subsets = ({"aoa_interpolation"} if subset == "val" else
                        {"aoa_extrapolation", "geometry_generalization", "ood_geometry"} if subset == "test" else {subset})
    allowed_quality = _quality_classes(quality_policy)
    excluded = {str(value) for value in excluded_geometry_ids}
    root = Path(root_dir)
    force_rows = _read_force_manifest(force_manifest_path)
    if force_consistency_only and not force_rows:
        raise FileNotFoundError(
            f"RocketCFD force-supervision manifest is missing: {force_manifest_path}. "
            "Run `python -m dataset.cache.rocket_force_manifest` first."
        )

    cases: list[AerodynamicSurfaceCase] = []
    for row in _read_manifest(manifest_path):
        if row["geometry_id"] in excluded:
            continue
        if subset != "all" and row["dataset_split"] not in selected_subsets:
            continue
        if allowed_quality is not None and row["convergence_class"] not in allowed_quality:
            continue
        force_row = force_rows.get(row["case_id"])
        if force_consistency_only and (
            force_row is None or force_row["force_supervision_eligible"].strip().lower() != "true"
        ):
            continue
        mach = float(row["mach"])
        aoa = float(row["aoa_deg"])
        cond = (3.4 * mach, 1.0) if condition_representation == "canonical" else (mach, aoa)
        coefficients = {}
        if force_row is not None:
            coefficients = {
                "CD": float(force_row["Cd"]),
                "CL": float(force_row["Cl"]),
                "AREF": float(force_row["Aref"]),
            }
        cases.append(
            AerodynamicSurfaceCase(
                name=row["case_id"],
                geometry_id=row["geometry_id"],
                cond=cond,
                source_paths=(str(root / row["case_relpath"] / "surface_fields.npz"),),
                coefficients=coefficients,
                aoa_deg=aoa,
            )
        )
    return cases


def _stable_case_seed(case_name: str, seed: int) -> int:
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(seed)) % (2**63 - 1)


def _surface_rows(path: str | Path, *, point_count: int | None, seed: int) -> np.ndarray:
    """Convert one campaign NPZ into ``xyz, Cp, normal, area`` rows.

    A cache retains all valid native face centres when their count is at most
    ``point_count``.  Only larger surfaces are capped using deterministic,
    area-proportional barycentric draws.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RocketCFD source surface is missing: {path}")
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int64)
        centers = np.asarray(data["cell_centers"], dtype=np.float32)
        cp = np.asarray(data["cp"], dtype=np.float32).reshape(-1)
        normals = np.asarray(data["normals"], dtype=np.float32)
        area = np.asarray(data["face_area"], dtype=np.float32).reshape(-1)
        valid = np.asarray(data.get("valid_mask", np.ones(cp.shape, dtype=bool)), dtype=bool)

    row_finite = (
        valid
        & np.all(np.isfinite(centers), axis=1)
        & np.isfinite(cp)
        & np.all(np.isfinite(normals), axis=1)
        & np.isfinite(area)
        & (area > 0.0)
    )
    if faces.shape[0] != cp.shape[0]:
        raise ValueError(f"Face/Cp size mismatch in {path}: faces={faces.shape}, Cp={cp.shape}.")
    faces = faces[row_finite]
    centers = centers[row_finite]
    cp = cp[row_finite]
    normals = normals[row_finite]
    area = area[row_finite]
    if cp.size == 0:
        raise ValueError(f"No finite valid surface cells in {path}.")
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norm, 1e-12)

    if point_count is None or cp.size <= int(point_count):
        return np.concatenate(
            [centers, cp[:, None], normals, area[:, None]], axis=1
        ).astype(np.float32, copy=False)

    point_count = int(point_count)
    if point_count <= 0:
        raise ValueError("point_count must be positive or None.")
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise ValueError(f"Triangle indices are out of range in {path}.")
    rng = np.random.default_rng(seed)
    probability = area.astype(np.float64)
    probability /= probability.sum()
    selected = rng.choice(cp.shape[0], size=point_count, replace=True, p=probability)
    triangles = points[faces[selected]]
    root_u = np.sqrt(rng.random(point_count, dtype=np.float32))
    v = rng.random(point_count, dtype=np.float32)
    sampled_xyz = (
        (1.0 - root_u)[:, None] * triangles[:, 0]
        + (root_u * (1.0 - v))[:, None] * triangles[:, 1]
        + (root_u * v)[:, None] * triangles[:, 2]
    )
    # Equal quadrature weights are appropriate after area-proportional draws.
    sampled_area = np.full((point_count, 1), float(area.sum()) / point_count, dtype=np.float32)
    return np.concatenate(
        [sampled_xyz, cp[selected, None], normals[selected], sampled_area], axis=1
    ).astype(np.float32, copy=False)


class RocketCFDCacheBuilder:
    """Build deterministic, fixed-size surface caches from campaign NPZ files."""

    dataset_name = "RocketCFD"

    def __init__(
        self,
        root_dir: str | Path = ROCKET_CFD_ROOT,
        manifest_path: str | Path = ROCKET_CFD_MANIFEST,
        cache_root: str | Path = ROCKET_CFD_CACHE_ROOT,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        self.cache_root = Path(cache_root)

    def build(
        self,
        *,
        subset: str = "all",
        quality_policy: str = "all_completed",
        point_count: int | None = 32768,
        seed: int = 200099,
        overwrite: bool = False,
        progress_interval: int = 100,
        continue_on_error: bool = False,
    ) -> tuple[int, int]:
        cases = rocket_cfd_cases(
            self.root_dir,
            manifest_path=self.manifest_path,
            subset=subset,
            quality_policy=quality_policy,
        )
        built = skipped = 0
        for index, case in enumerate(cases, start=1):
            target = surface_cache_path(self.cache_root, self.dataset_name, case.name)
            if target.is_file() and not overwrite:
                skipped += 1
                continue
            try:
                rows = _surface_rows(
                    case.source_paths[0],
                    point_count=point_count,
                    seed=_stable_case_seed(case.name, seed),
                )
                if write_surface_cache(
                    self.cache_root, self.dataset_name, case.name, rows, overwrite=overwrite,
                    case_metadata={"geometry_id": case.geometry_id, "condition": list(case.cond),
                                   "source_paths": list(case.source_paths), "coefficients": dict(case.coefficients),
                                   "aoa_deg": case.aoa_deg},
                ):
                    built += 1
                else:
                    skipped += 1
            except Exception as error:
                if not continue_on_error:
                    raise
                skipped += 1
                print(f"RocketCFD cache skipped {case.name}: {error}", flush=True)
            if progress_interval and (index % int(progress_interval) == 0 or index == len(cases)):
                print(
                    f"RocketCFD cache progress: {index}/{len(cases)} "
                    f"(built={built}, skipped={skipped})",
                    flush=True,
                )
        return built, skipped


class RocketCFDDataset(AerodynamicSurfaceOnlyDataset):
    """Five-way RocketCFD surface-Cp dataset with explicit published splits."""

    dataset_name = "RocketCFD"
    supported_coefficient_fields = ("CD", "CL", "AREF")

    def __init__(
        self,
        root_dir: str | Path = ROCKET_CFD_ROOT,
        manifest_path: str | Path = ROCKET_CFD_MANIFEST,
        force_manifest_path: str | Path = ROCKET_CFD_FORCE_MANIFEST,
        cache_root: str | Path = ROCKET_CFD_CACHE_ROOT,
        subset: str = "train",
        quality_policy: str = "all_completed",
        condition_representation: str = "canonical",
        excluded_geometry_ids: Iterable[str] = ROCKET_CFD_EXCLUDED_GEOMETRIES,
        force_consistency_only: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("rotate_geometry_by_aoa", condition_representation == "canonical")
        kwargs.setdefault("split", "all")
        kwargs.setdefault("ground_z", False)
        super().__init__(
            cases=rocket_cfd_cases(
                root_dir,
                manifest_path=manifest_path,
                force_manifest_path=force_manifest_path,
                subset=subset,
                quality_policy=quality_policy,
                excluded_geometry_ids=excluded_geometry_ids,
                condition_representation=condition_representation,
                force_consistency_only=force_consistency_only,
            ),
            cache_root=cache_root,
            cache_dataset_name=self.dataset_name,
            route=kwargs.pop("route", 0),
            **kwargs,
        )


Dataset = RocketCFDDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RocketCFD up-to-32768-point UniField caches.")
    parser.add_argument("--root-dir", default=str(ROCKET_CFD_ROOT))
    parser.add_argument("--manifest", default=str(ROCKET_CFD_MANIFEST))
    parser.add_argument("--cache-root", default=str(ROCKET_CFD_CACHE_ROOT))
    parser.add_argument("--subset", choices=(*ROCKET_CFD_SUBSETS, "all"), default="all")
    parser.add_argument("--quality-policy", choices=tuple(ROCKET_CFD_QUALITY_POLICIES), default="all_completed")
    parser.add_argument("--point-count", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=200099)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    built, skipped = RocketCFDCacheBuilder(args.root_dir, args.manifest, args.cache_root).build(
        subset=args.subset,
        quality_policy=args.quality_policy,
        point_count=args.point_count,
        seed=args.seed,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    print(f"RocketCFD cache complete: built={built}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "Dataset",
    "ROCKET_CFD_CACHE_ROOT",
    "ROCKET_CFD_EXCLUDED_GEOMETRIES",
    "ROCKET_CFD_FORCE_MANIFEST",
    "ROCKET_CFD_MANIFEST",
    "ROCKET_CFD_QUALITY_POLICIES",
    "ROCKET_CFD_ROOT",
    "ROCKET_CFD_SUBSETS",
    "RocketCFDCacheBuilder",
    "RocketCFDDataset",
    "rocket_cfd_cases",
]
