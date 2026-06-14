from __future__ import annotations

import numpy as np

from dataset.utils.load import ensure_xyz_prefix, field_dim


class EmptySurfaceDataset:
    def __init__(
        self,
        surface_input_list: list[str] | tuple[str, ...] | None = None,
        surface_target_list: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.surface_input_list = ensure_xyz_prefix(surface_input_list, ("xyz", "normal", "area"))
        self.surface_target_list = list(surface_target_list or [])

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del case_name, rng, kwargs
        input_dim = field_dim(self.surface_input_list)
        target_dim = field_dim(self.surface_target_list) if self.surface_target_list else 0
        return (
            np.zeros((0, input_dim), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, target_dim), dtype=np.float32),
        )
