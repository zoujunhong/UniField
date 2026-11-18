"""AircraftWing surface-pressure dataset migrated from ``field/dataset/Flight*.py``."""

from __future__ import annotations

from pathlib import Path

from dataset.cache.aerodynamic_surface import read_vtk_surface_pressure, write_surface_cache
from dataset.utils.aerodynamic_surface import AerodynamicSurfaceCase, cached_surface_cases, surface_cache_path
from dataset.utils.aerodynamic_surface_dataset import AerodynamicSurfaceOnlyDataset


AIRCRAFT_WING_ROOT = "/lustre/home/zdhs0017/data/Flight/3D_Wings_full/aircraft"
_WING_CONDITION_MARKER = "_m"


def _parse_wing_name(stem: str) -> tuple[str, float, float]:
    """Parse names such as ``Jiyi_12_0_m06a-15`` into geometry/Mach/AoA."""

    if _WING_CONDITION_MARKER not in stem:
        raise ValueError(f"AircraftWing file name has no Mach/AoA marker: {stem!r}")
    geometry_id, condition = stem.rsplit(_WING_CONDITION_MARKER, 1)
    if "a" not in condition:
        raise ValueError(f"AircraftWing file name has no AoA marker: {stem!r}")
    mach_text, aoa_text = condition.split("a", 1)
    try:
        return geometry_id, float(mach_text) / 10.0, float(aoa_text)
    except ValueError as error:
        raise ValueError(f"Cannot parse AircraftWing file name: {stem!r}") from error


def aircraft_wing_cases(
    root_dir: str | Path = AIRCRAFT_WING_ROOT,
    *,
    condition_representation: str = "canonical",
) -> list[AerodynamicSurfaceCase]:
    """Discover all wing geometry/condition cases without reading VTK payloads."""

    if condition_representation not in {"canonical", "mach_aoa", "legacy_freestream"}:
        raise ValueError("condition_representation must be 'canonical', 'mach_aoa', or 'legacy_freestream'.")
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"AircraftWing root does not exist: {root}")
    cases = []
    for path in sorted(root.glob("*.vtk")):
        geometry_id, mach, aoa = _parse_wing_name(path.stem)
        if condition_representation == "canonical":
            cond = (3.4 * mach, 1.0)
        elif condition_representation == "mach_aoa":
            cond = (mach, aoa)
        else:
            cond = (3.4 * mach, 0.0)
        cases.append(
            AerodynamicSurfaceCase(
                name=path.stem,
                geometry_id=geometry_id,
                cond=cond,
                source_paths=(str(path),),
                aoa_deg=aoa,
            )
        )
    if not cases:
        raise RuntimeError(f"No .vtk cases found under AircraftWing root {root}.")
    return cases


class AircraftWingCacheBuilder:
    """Build ``xyz, Cp, normal, area`` caches from the original wing VTK files."""

    dataset_name = "AircraftWing"

    def __init__(self, root_dir: str | Path | None = None, cache_root: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir or AIRCRAFT_WING_ROOT)
        self.cache_root = Path(cache_root) if cache_root is not None else Path("/data/home/zdhs0017/zoujunhong/data")

    def build(
        self,
        case_names: list[str] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
        *,
        progress_interval: int = 0,
        continue_on_error: bool = False,
    ) -> tuple[int, int]:
        cases = aircraft_wing_cases(self.root_dir)
        if case_names is not None:
            selected = {str(name) for name in case_names}
            cases = [case for case in cases if case.name in selected]
            missing = selected.difference(case.name for case in cases)
            if missing:
                raise KeyError(f"Unknown AircraftWing cache case(s): {sorted(missing)}")
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
                rows = read_vtk_surface_pressure(case.source_paths[0], pressure_field="Cp")
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
                print(f"AircraftWing cache skipped {case.name}: {error}", flush=True)
            if progress_interval and (index % progress_interval == 0 or index == len(cases)):
                print(f"AircraftWing cache progress: {index}/{len(cases)} (built={built}, skipped={skipped})", flush=True)
        return built, skipped


class AircraftWingDataset(AerodynamicSurfaceOnlyDataset):
    """Surface-only Cp dataset with geometry-disjoint splitting by wing design."""

    dataset_name = "AircraftWing"

    def __init__(
        self,
        root_dir: str | Path = AIRCRAFT_WING_ROOT,
        cache_root: str | Path | None = None,
        condition_representation: str = "canonical",
        **kwargs,
    ) -> None:
        root = Path(root_dir)
        kwargs.setdefault("rotate_geometry_by_aoa", condition_representation == "canonical")
        records = cached_surface_cases(cache_root or "/data/home/zdhs0017/zoujunhong/data", self.dataset_name, ids_file=kwargs.get("ids_file")) if root == Path(AIRCRAFT_WING_ROOT) and condition_representation == "canonical" else []
        super().__init__(
            cases=records or aircraft_wing_cases(root, condition_representation=condition_representation),
            cache_root=cache_root or "/data/home/zdhs0017/zoujunhong/data",
            route=kwargs.pop("route", 1),
            **kwargs,
        )


Dataset = AircraftWingDataset


__all__ = [
    "AIRCRAFT_WING_ROOT",
    "AircraftWingCacheBuilder",
    "AircraftWingDataset",
    "Dataset",
    "aircraft_wing_cases",
]
