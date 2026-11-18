"""ThingiCFD surface-pressure dataset migrated from ``field/OpenFOAMGeneration.py``."""

from __future__ import annotations

import re
from pathlib import Path

from dataset.cache.aerodynamic_surface import read_vtk_surface_pressure, write_surface_cache
from dataset.utils.aerodynamic_surface import AerodynamicSurfaceCase, cached_surface_cases, surface_cache_path
from dataset.utils.aerodynamic_surface_dataset import AerodynamicSurfaceOnlyDataset


THINGI_CFD_ROOT = "/lustre/home/zdhs0017/data/ThingiCFD/thingi"
_THINGI_NAME_RE = re.compile(
    r"^Thingi_(?P<geometry>\d+)_U_(?P<speed>[-+]?\d*\.?\d+)_(?P<placement>ground|center)$"
)


def _parse_thingi_name(stem: str) -> tuple[str, float, float]:
    match = _THINGI_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse ThingiCFD file name: {stem!r}")
    speed = float(match.group("speed"))
    ground_flag = 0.0 if match.group("placement") == "ground" else 1.0
    return match.group("geometry"), speed, ground_flag


def thingi_cfd_cases(
    root_dir: str | Path = THINGI_CFD_ROOT,
    *,
    speed_scale: float = 100.0,
) -> list[AerodynamicSurfaceCase]:
    """Discover ThingiCFD cases in the shared ``(U_inf / 100, placement)`` form."""

    if float(speed_scale) <= 0.0:
        raise ValueError("speed_scale must be positive.")
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ThingiCFD root does not exist: {root}")
    cases = []
    for path in sorted(root.glob("*.vtk")):
        geometry_id, speed, ground_flag = _parse_thingi_name(path.stem)
        cases.append(
            AerodynamicSurfaceCase(
                name=path.stem,
                geometry_id=geometry_id,
                cond=(speed / float(speed_scale), ground_flag),
                source_paths=(str(path),),
            )
        )
    if not cases:
        raise RuntimeError(f"No .vtk cases found under ThingiCFD root {root}.")
    return cases


class ThingiCFDCacheBuilder:
    """Build Cp caches from raw OpenFOAM VTK pressure fields."""

    dataset_name = "ThingiCFD"

    def __init__(
        self,
        root_dir: str | Path | None = None,
        cache_root: str | Path | None = None,
        density: float = 1.0,
    ) -> None:
        if float(density) <= 0.0:
            raise ValueError("density must be positive.")
        self.root_dir = Path(root_dir or THINGI_CFD_ROOT)
        self.cache_root = Path(cache_root) if cache_root is not None else Path("/data/home/zdhs0017/zoujunhong/data")
        self.density = float(density)

    def build(
        self,
        case_names: list[str] | None = None,
        limit: int | None = None,
        overwrite: bool = False,
        *,
        progress_interval: int = 0,
        continue_on_error: bool = False,
    ) -> tuple[int, int]:
        cases = thingi_cfd_cases(self.root_dir)
        if case_names is not None:
            selected = {str(name) for name in case_names}
            cases = [case for case in cases if case.name in selected]
            missing = selected.difference(case.name for case in cases)
            if missing:
                raise KeyError(f"Unknown ThingiCFD cache case(s): {sorted(missing)}")
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
                _, speed, _ = _parse_thingi_name(Path(case.source_paths[0]).stem)
                rows = read_vtk_surface_pressure(
                    case.source_paths[0],
                    pressure_field="p",
                    dynamic_pressure=0.5 * self.density * speed * speed,
                )
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
                print(f"ThingiCFD cache skipped {case.name}: {error}", flush=True)
            if progress_interval and (index % progress_interval == 0 or index == len(cases)):
                print(f"ThingiCFD cache progress: {index}/{len(cases)} (built={built}, skipped={skipped})", flush=True)
        return built, skipped


class ThingiCFDDataset(AerodynamicSurfaceOnlyDataset):
    """Surface-only ThingiCFD Cp dataset with geometry-disjoint held-out shapes."""

    dataset_name = "ThingiCFD"

    def __init__(
        self,
        root_dir: str | Path = THINGI_CFD_ROOT,
        cache_root: str | Path | None = None,
        speed_scale: float = 100.0,
        **kwargs,
    ) -> None:
        root = Path(root_dir)
        records = cached_surface_cases(cache_root or "/data/home/zdhs0017/zoujunhong/data", self.dataset_name, ids_file=kwargs.get("ids_file")) if root == Path(THINGI_CFD_ROOT) and float(speed_scale) == 100.0 else []
        super().__init__(
            cases=records or thingi_cfd_cases(root, speed_scale=speed_scale),
            cache_root=cache_root or "/data/home/zdhs0017/zoujunhong/data",
            route=kwargs.pop("route", 2),
            **kwargs,
        )


Dataset = ThingiCFDDataset


__all__ = [
    "THINGI_CFD_ROOT",
    "ThingiCFDCacheBuilder",
    "ThingiCFDDataset",
    "Dataset",
    "thingi_cfd_cases",
]
