from __future__ import annotations

from dataset.cache.gaussian3d_high_speed import HighSpeedTrainGaussian3DCacheBuilder
from dataset.cache.high_speed_train import HIGH_SPEED_TRAIN_ROOT
from dataset.utils.high_speed_train_gaussian3d_dataset import HighSpeedTrainGaussian3DDatasetBase


class MaglevGaussian3DDataset(HighSpeedTrainGaussian3DDatasetBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("Maglev", *args, **kwargs)


Dataset = MaglevGaussian3DDataset


__all__ = ["HIGH_SPEED_TRAIN_ROOT", "HighSpeedTrainGaussian3DCacheBuilder", "MaglevGaussian3DDataset", "Dataset"]
