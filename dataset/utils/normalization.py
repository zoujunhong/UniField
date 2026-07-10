from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DEFAULT_U_REF = 30.0
DEFAULT_PRESSURE_SCALE = 450.0
DEFAULT_VELOCITY_SCALE = 30.0

PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25
VELOCITY_MEAN = (21.725932072168813, -0.23861807530826643, 0.6522818340397506)
VELOCITY_STD = (12.373392759706936, 3.8865940380244175, 3.9738297581062536)

HIGH_SPEED_STANDARD_STATS = {
    "combined_train": {
        "p": {"mean": [-75.515029090242], "std": [263.619239069873]},
        "U": {
            "mean": [65.45423110624606, 6.47849221371492, 0.47900760752373295],
            "std": [18.601505700956604, 8.636807120478109, 5.498748091003668],
        },
        "k": {"mean": [6.967757472319873], "std": [26.0782976773323]},
        "omega": {"mean": [749.0660075504542], "std": [1952.4277519119837]},
        "nut": {"mean": [0.15504855036235843], "std": [0.14018536402961934]},
    },
    "CRH450_train": {
        "p": {"mean": [-32.34975199114525], "std": [218.69346787655905]},
        "U": {
            "mean": [68.48530119847301, 9.40344046768596, 0.1319855732071856],
            "std": [18.054944760267468, 7.769734096453999, 4.028086278027959],
        },
        "k": {"mean": [3.6750794839260092], "std": [6.859804880877931]},
        "omega": {"mean": [191.66724128690316], "std": [955.6215105680669]},
        "nut": {"mean": [0.19486157855665068], "std": [0.11338932748545408]},
    },
    "Maglev_train": {
        "p": {"mean": [-103.97213764568858], "std": [285.90548265539244]},
        "U": {
            "mean": [63.455970068369356, 4.550192990437629, 0.7077850972613741],
            "std": [18.686207428672057, 8.637958560538713, 6.272257829353861],
        },
        "k": {"mean": [9.138485916255022], "std": [32.94757634046616]},
        "omega": {"mean": [1116.5363121207968], "std": [2320.161125191411]},
        "nut": {"mean": [0.12880144261638253], "std": [0.14965821462717585]},
    },
}


def resolve_high_speed_standard_stats(standard_stats=None) -> dict:
    if standard_stats is None:
        return HIGH_SPEED_STANDARD_STATS["combined_train"]
    if isinstance(standard_stats, str):
        if standard_stats not in HIGH_SPEED_STANDARD_STATS:
            raise KeyError(
                f"Unknown HighSpeedTrain standard_stats preset: {standard_stats}. "
                f"Available presets: {sorted(HIGH_SPEED_STANDARD_STATS)}"
            )
        return HIGH_SPEED_STANDARD_STATS[standard_stats]
    if isinstance(standard_stats, dict):
        if "fields" in standard_stats:
            return standard_stats["fields"]
        return standard_stats
    raise TypeError("standard_stats must be None, a preset name, or a field stats dict.")


def high_speed_standard_field_stats(name: str, standard_stats=None) -> tuple[np.ndarray, np.ndarray]:
    stats = resolve_high_speed_standard_stats(standard_stats)
    if name not in stats:
        raise KeyError(f"HighSpeedTrain standard stats do not contain field {name!r}.")
    mean = np.asarray(stats[name]["mean"], dtype=np.float32).reshape(1, -1)
    std = np.maximum(np.asarray(stats[name]["std"], dtype=np.float32).reshape(1, -1), 1e-12)
    return mean, std


def normalize_drivaer_field(
    name: str,
    values: np.ndarray,
    normalization: str,
    u_ref: float = DEFAULT_U_REF,
    pressure_mean: float = PRESSURE_MEAN,
    pressure_std: float = PRESSURE_STD,
    velocity_mean: Iterable[float] = VELOCITY_MEAN,
    velocity_std: Iterable[float] = VELOCITY_STD,
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if normalization == "none":
        return values
    if normalization not in {"physical", "standard"}:
        raise ValueError("normalization must be 'physical', 'standard', or 'none'.")

    if name == "p":
        if normalization == "physical":
            return values / max(0.5 * float(u_ref) * float(u_ref), 1e-12)
        return (values - float(pressure_mean)) / max(float(pressure_std), 1e-12)

    if name == "U":
        if normalization == "physical":
            velocity_ref = np.array([float(u_ref), 0.0, 0.0], dtype=np.float32).reshape(1, 3)
            return (values - velocity_ref) / max(abs(float(u_ref)), 1e-12)
        mean = np.asarray(list(velocity_mean), dtype=np.float32).reshape(1, 3)
        std = np.maximum(np.asarray(list(velocity_std), dtype=np.float32).reshape(1, 3), 1e-12)
        return (values - mean) / std

    return values


def normalize_high_speed_field(
    name: str,
    values: np.ndarray,
    normalization: str,
    speed: float,
    clamp_cp: tuple[float, float] | None = None,
    standard_stats=None,
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if normalization == "none":
        return values
    if normalization not in {"physical", "standard"}:
        raise ValueError("HighSpeedTrain normalization must be 'physical', 'standard', or 'none'.")

    if normalization == "standard":
        mean, std = high_speed_standard_field_stats(name, standard_stats)
        return (values - mean) / std

    speed_scale = max(float(speed), 1e-12)
    dynamic_pressure = max(0.5 * speed_scale * speed_scale, 1e-12)
    if name == "p":
        values = values / dynamic_pressure
        if clamp_cp is not None:
            values = np.clip(values, clamp_cp[0], clamp_cp[1])
        return values
    if name == "U":
        return values / speed_scale
    if name == "k":
        return values / max(0.01 * speed_scale * speed_scale, 1e-12)
    if name == "omega":
        return values / max(100.0 * speed_scale, 1e-12)
    if name == "nut":
        return values / max(0.01 * speed_scale, 1e-12)
    return values
