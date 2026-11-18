"""Retained baseline implementations, imported only when selected."""
from importlib import import_module

_MODELS = {
    'ABUPTFieldModel': '.abupt',
    'SMARTFieldModel': '.smart',
    'Transolver3FieldModel': '.transolver3',
}


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(name)
    return getattr(import_module(_MODELS[name], __name__), name)


__all__ = list(_MODELS)
