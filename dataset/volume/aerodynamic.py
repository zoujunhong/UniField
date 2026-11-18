"""Empty volume-side component for surface-only aerodynamic pressure datasets."""

from __future__ import annotations

import numpy as np


class EmptyAerodynamicVolumeDataset:
    """Surface-only paper datasets have no volume CFD labels in their original form."""

    def __init__(self, volume_target_list: list[str] | tuple[str, ...] | None = None) -> None:
        fields = list(volume_target_list or [])
        if fields:
            raise ValueError(f"Surface-only aerodynamic datasets do not provide volume targets: {fields}.")

    def sample(self, case_name: str, rng: np.random.Generator):
        del case_name, rng
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)


__all__ = ["EmptyAerodynamicVolumeDataset"]
