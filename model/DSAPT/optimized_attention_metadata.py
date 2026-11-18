"""Batch-one attention indexing without device-to-host scalar reads.

Apply ``patch(encoder)`` after the existing local/global tensor attention
functions have been installed.  Only eager Point metadata preparation changes:
the bound ``_tensor_attention`` functions, RoPE, SDPA and all floating-point
operations remain untouched.

Caches belong to one Point and one serialization tensor pair.  They are never
global or keyed by point count alone; another stage, vehicle, or serialization
gets its own cache.  The current Point pipeline creates a new Point on pooling
and does not mutate serialization tensors in place.
"""

import types

import torch


_CACHE_KEY = "_batch1_attention_metadata_cache"


class _MetadataCache:
    def __init__(self, point, count):
        self.serialized_order = point.serialized_order
        self.serialized_inverse = point.serialized_inverse
        self.count = count
        self.padding = {}
        self.orders = {}

    def matches(self, point, count):
        return (
            self.count == count
            and self.serialized_order is point.serialized_order
            and self.serialized_inverse is point.serialized_inverse
        )


def batch1_padding(count, patch_size, device):
    """Return the official pad/unpad/cu_seqlens using shape-derived integers.

    An incomplete last patch borrows the tail of the preceding full patch.
    For example N=5, K=4 gives pad=[0,1,2,3,4,1,2,3], not repeated point 4.
    Clouds with N<=K remain unpadded, exactly as in the official implementation.
    """
    if count <= 0 or patch_size <= 0:
        raise ValueError("Attention requires positive point count and patch size")
    padded_count = (
        ((count + patch_size - 1) // patch_size) * patch_size
        if count > patch_size
        else count
    )
    pad = torch.arange(padded_count, dtype=torch.long, device=device)
    unpad = torch.arange(count, dtype=torch.long, device=device)
    remainder = count % patch_size
    if count > patch_size and remainder:
        # Original source interval:
        # [padded_count - 2*K + remainder : padded_count - K].
        # Source and destination do not overlap.
        pad[count:padded_count] = pad[count - patch_size : count - remainder]
    cu_seqlens = torch.nn.functional.pad(
        torch.arange(0, padded_count, patch_size, dtype=torch.int32, device=device),
        (0, 1),
        value=padded_count,
    )
    return pad, unpad, cu_seqlens


def attention_metadata(point, patch_size, order_index):
    """Cache padded order/inverse for a stage and serialization order.

    Batch-one offset is required to describe all feature rows, as in the source
    model/data contract.  Count comes from ``feat.shape[0]``; no offset tensor
    value, GPU scalar, ``item()``, ``tolist()``, or tensor truth value is read.
    """
    if point.offset.numel() != 1:
        raise ValueError("Validated for one vehicle per GPU only")
    count = point.feat.shape[0]
    cache = point.get(_CACHE_KEY)
    if not isinstance(cache, _MetadataCache) or not cache.matches(point, count):
        cache = _MetadataCache(point, count)
        point[_CACHE_KEY] = cache
    key = (patch_size, order_index)
    if key not in cache.orders:
        if patch_size not in cache.padding:
            cache.padding[patch_size] = batch1_padding(
                count, patch_size, point.offset.device
            )
        pad, unpad, _ = cache.padding[patch_size]
        cache.orders[key] = (
            point.serialized_order[order_index][pad],
            unpad[point.serialized_inverse[order_index]],
            min(patch_size, count),
        )
    # Retain the public fields populated by the official helper.  Use the
    # current patch size even when multiple sizes share one Point.
    point["pad"], point["unpad"], point["cu_seqlens_key"] = cache.padding[patch_size]
    return cache.orders[key]


def _local_forward(self, point):
    order, inverse, length = attention_metadata(
        point, self.patch_size, self.order_index
    )
    point.feat = self._tensor_attention(
        point.feat, point.coord, order, inverse, length
    )
    return point


def _global_forward(self, point):
    order, inverse, length = attention_metadata(
        point, self.patch_size, self.order_index
    )
    point.feat = self._tensor_attention(
        point.feat,
        point.coord,
        order,
        inverse,
        length,
        point.context["feat"],
        point.context["coord"],
    )
    return point


def patch(encoder):
    """Patch installed Tiny local/global attention forward metadata only.

    No parameters, buffers, compile configuration or tensor attention functions
    are changed.  This function can be called repeatedly and returns the number
    of attention modules covered.
    """
    from model.third_party.utonia_dsapt.model import SerializedAttention

    modules = [m for m in encoder.modules() if isinstance(m, SerializedAttention)]
    for module in modules:
        if not hasattr(module, "_tensor_attention"):
            raise ValueError("Install existing local/global tensor attention first")
        if not module.enable_flash or module.enable_rpe:
            raise ValueError("This patch requires the existing Flash SDPA attention")
    for module in modules:
        forward = _global_forward if hasattr(module, "global_projection") else _local_forward
        module.forward = types.MethodType(forward, module)
    return len(modules)
