"""Remove singleton reductions from the Tiny fine branch's stride-one pools.

Usage after constructing the existing DualEncoder::

    from . import optimized_pool
    optimized_pool.patch(encoder)

Only ``fine.backbone.enc.enc2/enc3/enc4.down`` are changed. The default path
keeps the original serialization (including its CPU RNG draw) and sparsify
calls. ``patch(encoder, reuse_serialization=True)`` additionally reuses the
canonical serialization computed at enc2 for enc3 and enc4, while making the
same fresh CPU randperm at each pooling call.

This is deliberately restricted to the existing (2, 1, 1, 1) fine branch.
The first GridPooling emits unique grid-coordinate rows in the canonical
``torch.unique(sorted=True, dim=0)`` order (lexicographic for integer rows).
The intervening Blocks change features only. The shortcut requires another
identical unique call to preserve that row order; this must be checked on the
installed CUDA backend. Consequently, each later pool's cluster, sorting
indices and cluster heads are arange(N), and counts are all one. Projection,
normalization, activation, parent links and RNG order remain
unchanged. Geometry tensors are treated as immutable and may share storage
between parent and child, but sparse tensors and feature tensors are distinct.

The shortcut is restricted to the current max feature reducer and geometry
without gradients. For an ordinary finite singleton, max has one selected
input and an identity Jacobian; the geometry mean is a one-element sum divided
by one. The shortcut does not promise the native reducer's special handling
of NaNs, infinities, dtype-boundary sentinels, subnormals or signed-zero bit
patterns. Those native kernels are binary-only in the inspected environment,
so exact numeric behavior is a validation requirement, not an assumed test
result. No test is run by this module, and it changes no parameters, buffers,
or checkpoint keys.
"""

from dataclasses import dataclass
import types

import torch
from model.third_party.utonia_dsapt.model import GridPooling
from model.third_party.utonia_dsapt.structure import Point


_CACHE_KEY = "_exact_stride_one_serialization"


@dataclass(frozen=True)
class _SerializationCache:
    names: tuple
    depth: int
    code: torch.Tensor
    order: torch.Tensor
    inverse: torch.Tensor


def _serialize_from_cache(point, parent, order, shuffle_orders):
    """Cache canonical rows, never an earlier stage's shuffled row order."""
    names = tuple(order)
    cache = parent.get(_CACHE_KEY)
    if cache is None:
        # This is the same encode/argsort/scatter as upstream, with only the
        # randperm postponed until immediately below. No extra RNG is consumed.
        point.serialization(order=order, shuffle_orders=False)
        cache = _SerializationCache(
            names,
            point.serialized_depth,
            point.serialized_code,
            point.serialized_order,
            point.serialized_inverse,
        )
    elif cache.names != names:
        raise RuntimeError("Serialization orders changed within identity pooling")

    point["order"] = order
    point["serialized_depth"] = cache.depth
    code, serialized_order, inverse = cache.code, cache.order, cache.inverse
    if shuffle_orders:
        # Keep the upstream default device and generator. In the current
        # training process this is a CPU randperm, including in eval mode.
        perm = torch.randperm(code.shape[0])
        code = code[perm]
        serialized_order = serialized_order[perm]
        inverse = inverse[perm]
    point["serialized_code"] = code
    point["serialized_order"] = serialized_order
    point["serialized_inverse"] = inverse
    point[_CACHE_KEY] = cache


def _forward(self, point):
    if self.stride != 1:
        raise RuntimeError("Singleton pooling patch requires stride=1")
    if point.offset.numel() != 1:
        raise ValueError("Singleton pooling patch is restricted to batch=1")
    if "grid_coord" not in point:
        raise ValueError("Expected grid coordinates from the preceding GridPooling")
    if self.reduce != "max":
        raise ValueError("Singleton pooling patch supports the current max reducer only")
    for key in ("coord", "origin_coord", "color"):
        if key in point and point[key].requires_grad:
            raise ValueError(f"Singleton pooling patch requires fixed {key} geometry")

    n = point.feat.shape[0]
    # Keep the original Linear call before normalization and activation. With
    # one ordinary finite value per segment, feature max reduces to identity.
    data = dict(
        feat=self.proj(point.feat),
        coord=point.coord,
        grid_coord=point.grid_coord,
        batch=point.batch,
    )
    for key in ("origin_coord", "condition", "context", "name", "split", "color"):
        if key in point:
            data[key] = point[key]
    if "grid_size" in point:
        data["grid_size"] = point.grid_size * self.stride
    if self.traceable:
        # The upstream cluster and CSR pointers are int64 tensors. Preserve
        # the complete parent hierarchy even though this encoder has no decoder.
        data["pooling_inverse"] = torch.arange(n, dtype=torch.long, device=point.feat.device)
        data["pooling_parent"] = point
        data["idx_ptr"] = torch.arange(n + 1, dtype=torch.long, device=point.feat.device)

    output = Point(data)
    if getattr(self, "norm", None) is not None:
        output = self.norm(output)
    if getattr(self, "act", None) is not None:
        output = self.act(output)
    if self._exact_pool_reuse_serialization:
        _serialize_from_cache(output, point, point.order, self.shuffle_orders)
    else:
        output.serialization(order=point.order, shuffle_orders=self.shuffle_orders)
    output.sparsify()
    return output


def _fine_pools(encoder):
    if not hasattr(encoder, "fine") or not hasattr(encoder.fine, "backbone"):
        raise TypeError("Pass the DualEncoder instance, not an arbitrary backbone")
    backbone = encoder.fine.backbone
    stages = list(backbone.enc.children())
    if len(stages) != 5 or not backbone.enc_mode:
        raise ValueError("Expected the five-stage encoder-only Tiny fine branch")
    pools = [getattr(stage, "down", None) for stage in stages[1:]]
    if not all(isinstance(pool, GridPooling) for pool in pools):
        raise TypeError("Expected the original GridPooling modules at stages 1..4")
    if tuple(pool.stride for pool in pools) != (2, 1, 1, 1):
        raise ValueError("Expected fine pooling strides (2, 1, 1, 1)")
    if any(pool.reduce != "max" for pool in pools):
        raise ValueError("Expected the original max feature pooling reducers")
    return pools[1:]


def patch(encoder, *, reuse_serialization=False):
    """Patch exactly three fine pools in place; return their module names.

    Calling this again is safe and updates the optional serialization mode.
    Geometry caches live only on the current forward's Point hierarchy, so no
    cached tensor or autograd graph is retained between samples or steps.
    """
    pools = _fine_pools(encoder)
    for pool in pools:
        if not hasattr(pool, "_exact_pool_original_forward"):
            pool._exact_pool_original_forward = pool.forward
        pool._exact_pool_reuse_serialization = bool(reuse_serialization)
        pool.forward = types.MethodType(_forward, pool)
    return tuple(f"fine.backbone.enc.enc{stage}.down" for stage in (2, 3, 4))


def unpatch(encoder):
    """Restore the original three bound forward methods without changing weights."""
    for pool in _fine_pools(encoder):
        if hasattr(pool, "_exact_pool_original_forward"):
            pool.forward = pool._exact_pool_original_forward
            del pool._exact_pool_original_forward
            del pool._exact_pool_reuse_serialization
