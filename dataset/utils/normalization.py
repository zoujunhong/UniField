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
    free_stream: np.ndarray,
    speed: float,
    clamp_cp: tuple[float, float] | None = (-5.0, 1.05),
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if normalization == "none":
        return values
    if normalization != "physical":
        raise ValueError("HighSpeedTrain currently supports normalization='physical' or 'none'.")

    speed_scale = max(float(speed), 1e-12)
    dynamic_pressure = max(0.5 * speed_scale * speed_scale, 1e-12)
    if name == "p":
        values = values / dynamic_pressure
        if clamp_cp is not None:
            values = np.clip(values, clamp_cp[0], clamp_cp[1])
        return values
    if name == "U":
        return (values - free_stream.reshape(1, 3).astype(np.float32)) / speed_scale
    if name == "k":
        return values / max(speed_scale * speed_scale, 1e-12)
    if name in {"omega", "nut"}:
        return values
    return values
