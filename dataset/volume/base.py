from __future__ import annotations

import numpy as np

from dataset.utils.load import field_dim


class EmptyVolumeDataset:
    def __init__(self, volume_target_list: list[str] | tuple[str, ...] | None = None) -> None:
        self.volume_target_list = list(volume_target_list or [])

    def sample(self, case_name: str, rng: np.random.Generator, **kwargs):
        del case_name, rng, kwargs
        target_dim = field_dim(self.volume_target_list) if self.volume_target_list else 0
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, target_dim), dtype=np.float32),
        )
