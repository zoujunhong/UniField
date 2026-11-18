from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    """Minimal decoupled-weight-decay Lion optimizer used by the SMART baseline."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid Lion betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=float(lr), betas=tuple(float(beta) for beta in betas), weight_decay=float(weight_decay))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg.lerp(gradient, 1.0 - beta1)
                parameter.add_(update.sign(), alpha=-lr)
                exp_avg.lerp_(gradient, 1.0 - beta2)
        return loss


def get_params_groups(
    model: torch.nn.Module,
    lr: float = 0.001,
    wd: float = 0.01,
    special_keyword: list[str] | tuple[str, ...] | None = None,
    special_lr: list[float] | tuple[float, ...] | None = None,
) -> list[dict]:
    special_keyword = list(special_keyword or [])
    special_lr = list(special_lr or [])
    if len(special_keyword) != len(special_lr):
        raise ValueError("special_keyword and special_lr must have the same length.")

    matched_params = set()
    param_groups: list[dict] = []

    for keyword, sk_lr in zip(special_keyword, special_lr):
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad or param in matched_params or keyword not in name:
                continue
            matched_params.add(param)
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        if regularized:
            param_groups.append({"params": regularized, "base_lr": sk_lr, "weight_decay": wd})
        if not_regularized:
            param_groups.append({"params": not_regularized, "base_lr": sk_lr, "weight_decay": 0.0})

    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param in matched_params:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    if regularized:
        param_groups.append({"params": regularized, "base_lr": lr, "weight_decay": wd})
    if not_regularized:
        param_groups.append({"params": not_regularized, "base_lr": lr, "weight_decay": 0.0})
    return param_groups


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    name = str(config.get("name", "AdamW"))
    kwargs = dict(config.get("kwargs", {}))
    lr = float(kwargs.pop("lr", config.get("lr", 1e-4)))
    weight_decay = float(kwargs.pop("weight_decay", config.get("weight_decay", 0.04)))
    param_groups = get_params_groups(
        model,
        lr=lr,
        wd=weight_decay,
        special_keyword=config.get("special_keyword", []),
        special_lr=config.get("special_lr", []),
    )
    if name == "AdamW":
        return torch.optim.AdamW(param_groups, lr=lr, **kwargs)
    if name == "Adam":
        return torch.optim.Adam(param_groups, lr=lr, **kwargs)
    if name == "Ranger":
        from utils.ranger2020 import Ranger
        return Ranger(param_groups, lr=lr, **kwargs)
    if name == "Lion":
        kwargs.pop("eps", None)
        return Lion(param_groups, lr=lr, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


__all__ = ["Lion", "build_optimizer", "get_params_groups"]
