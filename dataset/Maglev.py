from __future__ import annotations

from dataset.cache.high_speed_train import HIGH_SPEED_TRAIN_ROOT, HighSpeedTrainCacheBuilder
from dataset.utils.high_speed_train_dataset import HighSpeedTrainDatasetBase


class MaglevDataset(HighSpeedTrainDatasetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("Maglev", *args, **kwargs)


Dataset = MaglevDataset


__all__ = ["HIGH_SPEED_TRAIN_ROOT", "HighSpeedTrainCacheBuilder", "MaglevDataset", "Dataset"]
