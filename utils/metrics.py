from __future__ import annotations

from dataclasses import dataclass, field

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
    sample_maxae: list[float] = field(default_factory=list)
    global_maxae: float = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().double()
        target = target.detach().double()
        diff = pred - target
        self.count += diff.numel()
        self.abs_err += diff.abs().sum().item()
        self.sq_err += diff.square().sum().item()
        self.abs_target += target.abs().sum().item()
        self.sq_target += target.square().sum().item()
        if diff.ndim == 0:
            sample_maxima = diff.abs().reshape(1)
        elif diff.ndim == 1:
            sample_maxima = diff.abs().reshape(1, -1).amax(dim=1)
        else:
            sample_maxima = diff.abs().reshape(diff.shape[0], -1).amax(dim=1)
        self.sample_maxae.extend(float(value) for value in sample_maxima.cpu())
        if sample_maxima.numel():
            self.global_maxae = max(self.global_maxae, float(sample_maxima.max().item()))

    def result(self) -> dict[str, float]:
        eps = 1e-12
        result = {
            "mse": self.sq_err / max(self.count, 1),
            "rmse": (self.sq_err / max(self.count, 1)) ** 0.5,
            "mae": self.abs_err / max(self.count, 1),
            "relative_l1": self.abs_err / max(self.abs_target, eps),
            "relative_l2": (self.sq_err ** 0.5) / max(self.sq_target ** 0.5, eps),
        }
        if self.sample_maxae:
            maxima = torch.tensor(self.sample_maxae, dtype=torch.float64)
            result.update(
                {
                    "mean_sample_maxae": float(maxima.mean().item()),
                    "median_sample_maxae": float(maxima.median().item()),
                    "p95_sample_maxae": float(torch.quantile(maxima, 0.95).item()),
                    "global_maxae": self.global_maxae,
                }
            )
        return result


def format_metrics(metrics: dict[str, float]) -> str:
    text = (
        f"mse={metrics['mse']:.6g}, "
        f"rmse={metrics['rmse']:.6g}, "
        f"mae={metrics['mae']:.6g}, "
        f"rel_l1={metrics['relative_l1'] * 100:.3f}%, "
        f"rel_l2={metrics['relative_l2'] * 100:.3f}%"
    )
    if "mean_sample_maxae" in metrics:
        text += (
            f", mean_maxae={metrics['mean_sample_maxae']:.6g}"
            f", median_maxae={metrics['median_sample_maxae']:.6g}"
            f", p95_maxae={metrics['p95_sample_maxae']:.6g}"
            f", global_maxae={metrics['global_maxae']:.6g}"
        )
    return text
