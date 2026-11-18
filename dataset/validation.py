"""Complete validation splits and pressure-aware query sampling.

The threshold is applied to physical Cp before normalization and to the entire
query row, so velocity and every other supervised field share the same mask.
"""
from pathlib import Path
import numpy as np


def prepare_validation_kwargs(name, kwargs):
    kwargs = dict(kwargs)
    kwargs.update(deterministic=True, repeat=1)
    for key in ("limit", "max_cases", "validation_cases", "max_samples"):
        kwargs.pop(key, None)
    if not kwargs.get("use_volume", bool(kwargs.get("volume_target_list"))):
        kwargs["num_query_points"] = 0
    if not kwargs.get("surface_target_list", ["p"]) and not kwargs.get("unsupervised_surface_context", False):
        kwargs["num_surface_query_points"] = 0
    if name == "DrivAerNet":
        kwargs.pop("split", None)
        kwargs["ids_file"] = str(Path(__file__).parent / "splits/DrivAerNet/val_design_ids.txt")
        kwargs["reynolds_similarity_speeds"] = None
        kwargs["reynolds_similarity_factor_range"] = None
    if name == "RocketCFD":
        kwargs["split"] = "all"  # Published subset is already the complete held-out cohort.
    # All validation members must exist; never silently shrink the cohort.
    if name in {"AircraftWing", "FullAircraft", "GeneratedHighSpeedTrain", "ThingiCFD", "RocketCFD", "CRH450", "Maglev", "CRH450TripNet", "MaglevTripNet", "CRH450Gaussian3D", "MaglevGaussian3D"}:
        kwargs["skip_incomplete"] = False
    kwargs.pop("unsupervised_surface_context", None)
    return kwargs


def configure_validation(dataset):
    dataset.validation_cp_min = -10.0
    for key in ("surface", "volume"):
        reader = getattr(dataset, key, None)
        if reader is not None:
            reader.validation_cp_min = -10.0
            if hasattr(reader, "surface_sampling"):
                reader.surface_sampling = "random"
    return dataset


def pressure_valid_rows(reader, rows, cp):
    minimum = getattr(reader, "validation_cp_min", None)
    if minimum is None:
        return rows
    cp = np.asarray(cp, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(cp)):
        raise ValueError("Nonfinite validation pressure; rebuild the cache explicitly")
    result = rows[cp >= minimum]
    if not len(result):
        raise ValueError("No validation query points remain after Cp >= -10 filtering")
    return result
