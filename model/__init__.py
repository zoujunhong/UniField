from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn

from model.query_unifield import QueryUniField


def _resolve_class(package: str, class_name: str):
    module = importlib.import_module(package)
    if not hasattr(module, class_name):
        raise AttributeError(f"{package} must export {class_name}.")
    return getattr(module, class_name)


def _encoder_memory_dim(encoder_kwargs: dict[str, Any]) -> int:
    if "memory_dim" in encoder_kwargs:
        return int(encoder_kwargs["memory_dim"])
    embed_dims = encoder_kwargs.get("embed_dims")
    if embed_dims:
        return int(embed_dims[-1])
    channels = encoder_kwargs.get("channels")
    if channels:
        return int(channels[0])
    raise KeyError("encoder_kwargs must contain embed_dims, channels, or memory_dim so decoder memory_dim can be inferred.")


def build_model(config: dict[str, Any]) -> nn.Module:
    native_model_name = config.get("native_model_name", config.get("model_class"))
    if native_model_name:
        native_model_kwargs = dict(config.get("native_model_kwargs", config.get("model_kwargs", {})))
        native_package = "model.DSAPT.field_model" if native_model_name == "DSAPTFieldModel" else "model.Baseline"
        native_cls = _resolve_class(native_package, str(native_model_name))
        return native_cls(**native_model_kwargs)

    if "encoder_name" not in config:
        raise KeyError(
            "model config must use either native_model_name/native_model_kwargs or the composable schema: "
            "encoder_name/encoder_kwargs with an optional decoder_name/decoder_kwargs."
        )

    encoder_name = config["encoder_name"]
    decoder_name = config.get("decoder_name")
    encoder_kwargs = dict(config.get("encoder_kwargs", {}))
    decoder_kwargs = dict(config.get("decoder_kwargs", {}))
    coefficient_head_kwargs = config.get("coefficient_head_kwargs")
    if coefficient_head_kwargs is not None:
        coefficient_head_kwargs = dict(coefficient_head_kwargs)

    encoder_cls = _resolve_class("model.Encoder", encoder_name)
    memory_dim = _encoder_memory_dim(encoder_kwargs)
    if coefficient_head_kwargs is not None:
        coefficient_head_kwargs.setdefault("in_dim", memory_dim)
    encoder = encoder_cls(**encoder_kwargs)
    if decoder_name is None:
        if coefficient_head_kwargs is None:
            raise KeyError("An encoder-only model requires coefficient_head_kwargs.")
        decoder = None
    else:
        decoder_cls = _resolve_class("model.Decoder", decoder_name)
        decoder_kwargs.setdefault("memory_dim", memory_dim)
        decoder = decoder_cls(**decoder_kwargs)
    return QueryUniField(
        encoder,
        decoder,
        default_query_chunk_size=int(config.get("default_query_chunk_size", decoder_kwargs.get("default_query_chunk_size", 50_000))),
        coefficient_head_kwargs=coefficient_head_kwargs,
        coordinate_normalization=config.get("coordinate_normalization"),
        query_reuses_surface_points=bool(config.get("query_reuses_surface_points", False)),
    )


__all__ = ["build_model", "QueryUniField"]
