from __future__ import annotations

import importlib
from typing import Any

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


def build_model(config: dict[str, Any]) -> QueryUniField:
    if "encoder_name" not in config or "decoder_name" not in config:
        raise KeyError(
            "model config must use the composable schema: "
            "encoder_name/encoder_kwargs/decoder_name/decoder_kwargs."
        )

    encoder_name = config["encoder_name"]
    decoder_name = config["decoder_name"]
    encoder_kwargs = dict(config.get("encoder_kwargs", {}))
    decoder_kwargs = dict(config.get("decoder_kwargs", {}))

    encoder_cls = _resolve_class("model.Encoder", encoder_name)
    decoder_cls = _resolve_class("model.Decoder", decoder_name)

    decoder_kwargs.setdefault("memory_dim", _encoder_memory_dim(encoder_kwargs))
    encoder = encoder_cls(**encoder_kwargs)
    decoder = decoder_cls(**decoder_kwargs)
    return QueryUniField(
        encoder,
        decoder,
        default_query_chunk_size=int(config.get("default_query_chunk_size", decoder_kwargs.get("default_query_chunk_size", 50_000))),
    )


__all__ = ["build_model", "QueryUniField"]
