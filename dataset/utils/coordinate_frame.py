"""Canonical coordinate frames shared by the cross-domain aerodynamic datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FIELD_WIDTHS = {
    "xyz": 3,
    "U": 3,
    "normal": 3,
    "normals": 3,
    "p": 1,
    "k": 1,
    "omega": 1,
    "nut": 1,
    "area": 1,
    "log_area": 1,
    "surface_distance": 1,
    "wall_distance": 1,
    "sdf": 1,
}


@dataclass(frozen=True)
class CanonicalCoordinateFrame:
    """Rotate a raw horizontal wind to +x, then centre/ground the geometry.

    ``wind_heading_rad`` is the raw wind heading measured from +x.  The active
    rotation is therefore ``R_z(-wind_heading_rad)``.  Positions receive the
    additional bounding-box translation, whereas vectors only rotate.
    """

    wind_heading_rad: float
    xy_center: tuple[float, float]
    z_min: float

    @classmethod
    def from_surface_xyz(
        cls,
        xyz: np.ndarray,
        wind_heading_rad: float = 0.0,
    ) -> "CanonicalCoordinateFrame":
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
            raise ValueError(f"Expected a non-empty (N, >=3) surface coordinate array, got {xyz.shape}.")
        rotated = cls._rotate_xyz(xyz[:, :3], wind_heading_rad)
        return cls(
            wind_heading_rad=float(wind_heading_rad),
            xy_center=(
                0.5 * float(rotated[:, 0].min() + rotated[:, 0].max()),
                0.5 * float(rotated[:, 1].min() + rotated[:, 1].max()),
            ),
            z_min=float(rotated[:, 2].min()),
        )

    @staticmethod
    def _rotate_xyz(values: np.ndarray, heading_rad: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        rotated = values.copy()
        cos_theta = float(np.cos(float(heading_rad)))
        sin_theta = float(np.sin(float(heading_rad)))
        # R_z(-theta) [x, y]^T = [cos*x + sin*y, -sin*x + cos*y]^T.
        rotated[..., 0] = cos_theta * values[..., 0] + sin_theta * values[..., 1]
        rotated[..., 1] = -sin_theta * values[..., 0] + cos_theta * values[..., 1]
        return rotated

    def transform_xyz(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError(f"Expected coordinates with shape (N, >=3), got {values.shape}.")
        transformed = self._rotate_xyz(values, self.wind_heading_rad)
        transformed[:, 0] -= self.xy_center[0]
        transformed[:, 1] -= self.xy_center[1]
        transformed[:, 2] -= self.z_min
        return transformed

    def rotate_vectors(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Expected vectors with shape (N, 3), got {values.shape}.")
        return self._rotate_xyz(values, self.wind_heading_rad)


def wind_heading(ux: float, uy: float) -> float:
    speed = float(np.hypot(float(ux), float(uy)))
    if speed <= 0.0:
        raise ValueError("A canonical wind frame needs a non-zero horizontal free-stream speed.")
    return float(np.arctan2(float(uy), float(ux)))


def transform_packed_surface_input(
    values: np.ndarray,
    fields: Iterable[str],
    frame: CanonicalCoordinateFrame,
) -> np.ndarray:
    """Transform packed encoder inputs containing ``xyz`` and optional normals."""

    values = np.asarray(values, dtype=np.float32).copy()
    offset = 0
    for field in fields:
        if field not in FIELD_WIDTHS:
            raise KeyError(f"Unknown packed field {field!r}.")
        width = FIELD_WIDTHS[field]
        if field == "xyz":
            values[:, offset : offset + 3] = frame.transform_xyz(values[:, offset : offset + 3])
        elif field in {"normal", "normals"}:
            values[:, offset : offset + 3] = frame.rotate_vectors(values[:, offset : offset + 3])
        offset += width
    return values


def rotate_packed_vector_targets(
    values: np.ndarray,
    fields: Iterable[str],
    frame: CanonicalCoordinateFrame,
    vector_fields: Iterable[str] = ("U",),
) -> np.ndarray:
    """Rotate selected three-component target fields, preserving all scalars."""

    values = np.asarray(values, dtype=np.float32).copy()
    vector_fields = set(vector_fields)
    offset = 0
    for field in fields:
        if field not in FIELD_WIDTHS:
            raise KeyError(f"Unknown packed field {field!r}.")
        width = FIELD_WIDTHS[field]
        if field in vector_fields:
            if width != 3:
                raise ValueError(f"Vector field {field!r} must have three channels, got {width}.")
            values[:, offset : offset + 3] = frame.rotate_vectors(values[:, offset : offset + 3])
        offset += width
    return values


__all__ = [
    "CanonicalCoordinateFrame",
    "FIELD_WIDTHS",
    "rotate_packed_vector_targets",
    "transform_packed_surface_input",
    "wind_heading",
]
