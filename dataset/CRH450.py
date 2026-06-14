from __future__ import annotations

from dataset.cache.high_speed_train import HIGH_SPEED_TRAIN_ROOT, HighSpeedTrainCacheBuilder
from dataset.utils.high_speed_train_dataset import HighSpeedTrainDatasetBase


class CRH450Dataset(HighSpeedTrainDatasetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("CRH450", *args, **kwargs)


Dataset = CRH450Dataset


__all__ = ["HIGH_SPEED_TRAIN_ROOT", "HighSpeedTrainCacheBuilder", "CRH450Dataset", "Dataset"]
