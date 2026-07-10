from __future__ import annotations

from dataset.cache.high_speed_train import HIGH_SPEED_TRAIN_ROOT
from dataset.cache.tripnet_high_speed import HighSpeedTrainTriplaneCacheBuilder
from dataset.utils.high_speed_train_tripnet_dataset import HighSpeedTrainTripNetDatasetBase


class CRH450TripNetDataset(HighSpeedTrainTripNetDatasetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("CRH450", *args, **kwargs)


Dataset = CRH450TripNetDataset


__all__ = ["HIGH_SPEED_TRAIN_ROOT", "HighSpeedTrainTriplaneCacheBuilder", "CRH450TripNetDataset", "Dataset"]
