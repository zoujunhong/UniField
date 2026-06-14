from __future__ import annotations

import torch


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
    name = config.get("name", "AdamW")
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
    raise ValueError(f"Unsupported optimizer: {name}")
