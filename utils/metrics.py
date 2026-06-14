from __future__ import annotations

from dataclasses import dataclass

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight: int | float) -> None:
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight: int | float = 1) -> None:
        if not self.initialized:
            self.initialize(val, weight)
            return
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


@dataclass
class MetricAccumulator:
    count: int = 0
    abs_err: float = 0.0
    sq_err: float = 0.0
    abs_target: float = 0.0
    sq_target: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().double()
        target = target.detach().double()
        diff = pred - target
        self.count += diff.numel()
        self.abs_err += diff.abs().sum().item()
        self.sq_err += diff.square().sum().item()
        self.abs_target += target.abs().sum().item()
        self.sq_target += target.square().sum().item()

    def result(self) -> dict[str, float]:
        eps = 1e-12
        return {
            "mse": self.sq_err / max(self.count, 1),
            "rmse": (self.sq_err / max(self.count, 1)) ** 0.5,
            "mae": self.abs_err / max(self.count, 1),
            "relative_l1": self.abs_err / max(self.abs_target, eps),
            "relative_l2": (self.sq_err ** 0.5) / max(self.sq_target ** 0.5, eps),
        }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"mse={metrics['mse']:.6g}, "
        f"rmse={metrics['rmse']:.6g}, "
        f"mae={metrics['mae']:.6g}, "
        f"rel_l1={metrics['relative_l1'] * 100:.3f}%, "
        f"rel_l2={metrics['relative_l2'] * 100:.3f}%"
    )
