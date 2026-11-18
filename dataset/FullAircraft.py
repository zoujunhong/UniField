"""FullAircraft (NGA) surface-pressure and integrated-load dataset."""

from __future__ import annotations

import re
from pathlib import Path

from dataset.cache.aerodynamic_surface import read_tecplot_surface_pressure, write_surface_cache
from dataset.utils.aerodynamic_surface import (
    AerodynamicSurfaceCase,
    cached_surface_cases,
    coefficients_for_condition,
    read_full_aircraft_result_coefficients,
    surface_cache_path,
)
from dataset.utils.aerodynamic_surface_dataset import AerodynamicSurfaceOnlyDataset


FULL_AIRCRAFT_ROOT = "/lustre/home/zdhs0017/data/Flight/nga"
FULL_AIRCRAFT_COEFFICIENT_FIELDS = ("CX", "CY", "CZ", "MX", "MY", "MZ", "CD", "CL", "K", "XCP")
_FIRST_PART_RE = re.compile(
    r"^WallSurfGridFlowParameters_"
    r"(?P<mach>[-+]?\d*\.?\d+)_[-+]?\d*\.?\d+_"
    r"(?P<aoa>[-+]?\d*\.?\d+)_1_11\.plt$"
)


def _case_name(sample_name: str, mach: float, aoa: float) -> str:
    return f"{sample_name}__M{mach:.6f}__A{aoa:+.6f}"


def _part_paths(sample_dir: Path, mach: float, aoa: float) -> tuple[str, ...]:
    return tuple(
        str(sample_dir / f"WallSurfGridFlowParameters_{mach:.6f}_0.000000_{aoa:.6f}_{part_id}_11.plt")
        for part_id in range(1, 12)
    )


def full_aircraft_cases(
    root_dir: str | Path = FULL_AIRCRAFT_ROOT,
    *,
    condition_representation: str = "canonical",
) -> list[AerodynamicSurfaceCase]:
    """Discover valid NGA geometry/Mach/AoA cases and optional CFD load labels."""

    if condition_representation not in {"canonical", "mach_aoa", "legacy_freestream"}:
        raise ValueError("condition_representation must be 'canonical', 'mach_aoa', or 'legacy_freestream'.")
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"FullAircraft root does not exist: {root}")
    cases = []
    for sample_dir in sorted((path for path in root.glob("sample-*") if path.is_dir()), key=lambda path: path.name):
        coefficient_table = read_full_aircraft_result_coefficients(sample_dir / f"{sample_dir.name}-result.plt")
        for first_part in sorted(sample_dir.glob("WallSurfGridFlowParameters_*_1_11.plt")):
            match = _FIRST_PART_RE.match(first_part.name)
            if match is None:
                continue
            mach = float(match.group("mach"))
            aoa = float(match.group("aoa"))
            paths = _part_paths(sample_dir, mach, aoa)
            if not all(Path(path).is_file() for path in paths):
                continue
            if condition_representation == "canonical":
                cond = (3.4 * mach, 1.0)
            elif condition_representation == "mach_aoa":
                cond = (mach, aoa)
            else:
                cond = (3.4 * mach, 0.0)
            cases.append(
                AerodynamicSurfaceCase(
                    name=_case_name(sample_dir.name, mach, aoa),
                    geometry_id=sample_dir.name,
                    cond=cond,
                    source_paths=paths,
                    coefficients=coefficients_for_condition(coefficient_table, mach, aoa),
                    aoa_deg=aoa,
                )
            )
    if not cases:
        raise RuntimeError(f"No complete FullAircraft cases found under {root}.")
    return cases


class FullAircraftCacheBuilder:
    """Build pointwise surface caches from the eleven Tecplot parts per NGA case."""

    dataset_name = "FullAircraft"

    def __init__(
        self,
        root_dir: str | Path | None = None,
        cache_root: str | Path | None = None,
        coordinate_scale: float = 1e-3,
    ) -> None:
        self.root_dir = Path(root_dir or FULL_AIRCRAFT_ROOT)
        self.cache_root = Path(cache_root) if cache_root is not None else Path("/data/home/zdhs0017/zoujunhong/data")
        self.coordinate_scale = float(coordinate_scale)

    def build(
        self,
        case_names: list[str] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
        *,
        progress_interval: int = 0,
        continue_on_error: bool = False,
    ) -> tuple[int, int]:
        cases = full_aircraft_cases(self.root_dir)
        if case_names is not None:
            selected = {str(name) for name in case_names}
            cases = [case for case in cases if case.name in selected]
            missing = selected.difference(case.name for case in cases)
            if missing:
                raise KeyError(f"Unknown FullAircraft cache case(s): {sorted(missing)}")
        if limit is not None:
            cases = cases[: int(limit)]
        built = skipped = 0
        progress_interval = max(0, int(progress_interval))
        for index, case in enumerate(cases, start=1):
            target_path = surface_cache_path(self.cache_root, self.dataset_name, case.name)
            if target_path.is_file() and not overwrite:
                skipped += 1
                continue
            try:
                rows = read_tecplot_surface_pressure(case.source_paths, coordinate_scale=self.coordinate_scale)
                if write_surface_cache(self.cache_root, self.dataset_name, case.name, rows, overwrite=overwrite,
                                       case_metadata={"geometry_id": case.geometry_id, "condition": list(case.cond),
                                                      "source_paths": list(case.source_paths), "coefficients": dict(case.coefficients),
                                                      "aoa_deg": case.aoa_deg}):
                    built += 1
                else:
                    skipped += 1
            except Exception as error:
                if not continue_on_error:
                    raise
                skipped += 1
                print(f"FullAircraft cache skipped {case.name}: {error}", flush=True)
            if progress_interval and (index % progress_interval == 0 or index == len(cases)):
                print(f"FullAircraft cache progress: {index}/{len(cases)} (built={built}, skipped={skipped})", flush=True)
        return built, skipped


class FullAircraftDataset(AerodynamicSurfaceOnlyDataset):
    """NGA full-aircraft Cp dataset with optional aggregate CFD coefficients."""

    dataset_name = "FullAircraft"
    supported_coefficient_fields = FULL_AIRCRAFT_COEFFICIENT_FIELDS

    def __init__(
        self,
        root_dir: str | Path = FULL_AIRCRAFT_ROOT,
        cache_root: str | Path | None = None,
        condition_representation: str = "canonical",
        **kwargs,
    ) -> None:
        root = Path(root_dir)
        # The original NGADataset_v implementation used sample-001--080 for
        # training and sample-081--100 for held-out evaluation.  Retain that
        # 80/20 default while making the split geometry-disjoint and explicit.
        kwargs.setdefault("val_ratio", 0.2)
        kwargs.setdefault("rotate_geometry_by_aoa", condition_representation == "canonical")
        records = cached_surface_cases(cache_root or "/data/home/zdhs0017/zoujunhong/data", self.dataset_name, ids_file=kwargs.get("ids_file")) if root == Path(FULL_AIRCRAFT_ROOT) and condition_representation == "canonical" else []
        super().__init__(
            cases=records or full_aircraft_cases(root, condition_representation=condition_representation),
            cache_root=cache_root or "/data/home/zdhs0017/zoujunhong/data",
            route=kwargs.pop("route", 1),
            **kwargs,
        )


Dataset = FullAircraftDataset


__all__ = [
    "FULL_AIRCRAFT_COEFFICIENT_FIELDS",
    "FULL_AIRCRAFT_ROOT",
    "FullAircraftCacheBuilder",
    "FullAircraftDataset",
    "Dataset",
    "full_aircraft_cases",
]
