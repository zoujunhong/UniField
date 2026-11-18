"""Reuse exact batch-one integer geometry metadata within one encoder call.

Install after constructing the existing DualEncoder (before or after the other
local optimization patches)::

    from . import optimized_geometry_metadata
    handle = optimized_geometry_metadata.patch(encoder)
    ...
    handle.uninstall()

The process-wide Point method wrappers only act while this particular encoder
is running. Module hooks establish that scope and the currently executing
GridPooling's stride; they do not replace any pool/branch/block forward method.
Normal module invocation is required: use encoder(batch), not encoder.forward.
One active installation is supported per process, with synchronous execution of
this encoder. Uninstall only outside a forward call.

For each forward, the two branches receive the same initial grid tensor. Its
three integer maxima are read to the host once and cached by tensor identity
for this call only. At a pool of stride s, each maximum is exactly parent_max
// s: unique removes duplicates without removing any represented coordinate.
The exact upstream depth remains (max(max_xyz) + 1).bit_length(), and sparsify
uses max_xyz + pad. No approximation or fixed enclosing cube is substituted.
This preserves all original unique/sort/reduce operations, serialization order
calculations, and serialization randperm calls. It creates a fresh sparse
tensor for each Point and does not share sparse indice dictionaries.

Input contract: one nonempty vehicle, valid zero-based batch indices (all zero),
integer nonnegative grid coordinates in the existing serialization range, and
positive integer pooling strides. Blocks and hooks must not mutate grid_coord
or reorder coordinates in place. batch/offset must describe every feature row.
These are the current Tiny DualEncoder data invariants; inspecting tensor
values to revalidate them inside the hot path would reintroduce synchronizations.
No change is made to coord/origin_coord/color/features or their autograd paths.
Coordinate gradients are retained by the original pooling and attention code.

Caches reside in a ContextVar execution scope and in private immutable Point
metadata, never in parameters/buffers. Per-forward root references are released
on return, including exceptional return. Point parent hierarchies may retain
their tiny CPU metadata for as long as the original output hierarchy lives.
No tensors or graphs are retained between calls by the patch handle.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
import spconv.pytorch as spconv
from model.third_party.utonia_dsapt.model import GridPooling
from model.third_party.utonia_dsapt.structure import Point


_KEY = "_exact_batch1_geometry_metadata"
_ACTIVE = ContextVar("exact_batch1_geometry_execution", default=None)
_INSTALLED = None


@dataclass(frozen=True)
class _Geometry:
    max_xyz: tuple
    execution_token: object

    @property
    def depth(self):
        return (max(self.max_xyz) + 1).bit_length()

    def downsample(self, stride):
        return _Geometry(tuple(value // stride for value in self.max_xyz), self.execution_token)


@dataclass
class _Root:
    grid: torch.Tensor
    geometry: _Geometry
    batch: torch.Tensor = None


@dataclass
class _PoolScope:
    module: object
    geometry: object
    stride: int


@dataclass
class _Execution:
    owner: object
    previous: object
    token: object = field(default_factory=object)
    roots: dict = field(default_factory=dict)
    pools: list = field(default_factory=list)


def _execution():
    current = _ACTIVE.get()
    if current is not None and current.owner is _INSTALLED:
        return current
    return None


def _initial_geometry(execution, grid):
    entry = execution.roots.get(id(grid))
    if entry is not None and entry.grid is grid:
        execution.owner.stats["initial_geometry_cache_hits"] += 1
        return entry
    # Exactly the upstream maxima, combined in one host transfer. Integer max
    # does not interact with coordinate or feature autograd.
    maxima = tuple(int(value) for value in grid.amax(dim=0).tolist())
    if len(maxima) != 3:
        raise ValueError("Expected three grid-coordinate columns")
    entry = _Root(grid, _Geometry(maxima, execution.token))
    execution.roots[id(grid)] = entry
    execution.owner.stats["initial_max_host_reads"] += 1
    return entry


def _point_init(self, *args, **kwargs):
    execution = _execution()
    original = _INSTALLED.original_init if _INSTALLED is not None else None
    if original is None:
        raise RuntimeError("Geometry Point initializer outlived its installation")
    if execution is None or kwargs or len(args) != 1 or not isinstance(args[0], dict):
        return original(self, *args, **kwargs)

    source = args[0]
    # Addict recursively constructs Points for arbitrary nested dictionaries.
    # Only actual geometric Points qualify; coarse context dictionaries do not.
    if not {"feat", "coord", "grid_coord"}.issubset(source):
        return original(self, *args, **kwargs)
    if "offset" not in source and "batch" not in source:
        return original(self, *args, **kwargs)
    if "offset" in source and source["offset"].numel() != 1:
        raise ValueError("Geometry metadata optimization requires batch=1")

    data = dict(source)
    count = data["feat"].shape[0]
    if count <= 0:
        raise ValueError("Geometry metadata optimization requires nonempty Points")
    geometry = data.get(_KEY)
    root = None
    if not isinstance(geometry, _Geometry) or geometry.execution_token is not execution.token:
        scope = execution.pools[-1] if execution.pools else None
        if (scope is not None and isinstance(scope.geometry, _Geometry)
                and scope.geometry.execution_token is execution.token):
            geometry = scope.geometry.downsample(scope.stride)
            execution.owner.stats["pool_metadata_derivations"] += 1
        else:
            root = _initial_geometry(execution, data["grid_coord"])
            geometry = root.geometry
        data[_KEY] = geometry

    if "batch" not in data:
        if root is not None:
            if root.batch is None:
                root.batch = torch.zeros(count, dtype=torch.long, device=data["feat"].device)
            data["batch"] = root.batch
        else:
            data["batch"] = torch.zeros(count, dtype=torch.long, device=data["feat"].device)
        execution.owner.stats["explicit_batch_constructions"] += 1
    if "offset" not in data:
        data["offset"] = torch.full(
            (1,), count, dtype=torch.long, device=data["feat"].device
        )
        execution.owner.stats["explicit_offset_constructions"] += 1
    return original(self, data)


def _serialization(self, order="z", depth=None, shuffle_orders=False):
    execution = _execution()
    geometry = self.get(_KEY)
    if (execution is not None and depth is None and isinstance(geometry, _Geometry)
            and geometry.execution_token is execution.token):
        depth = geometry.depth
        execution.owner.stats["serialization_depth_reuses"] += 1
    return _INSTALLED.original_serialization(
        self, order=order, depth=depth, shuffle_orders=shuffle_orders
    )


def _sparsify(self, pad=96):
    execution = _execution()
    geometry = self.get(_KEY)
    if (execution is None or not isinstance(geometry, _Geometry)
            or geometry.execution_token is not execution.token):
        return _INSTALLED.original_sparsify(self, pad=pad)
    if self.offset.numel() != 1:
        raise ValueError("Geometry sparsify optimization requires batch=1")

    # Preserve an explicitly supplied shape just as the original method does.
    # Standard GridPooling creates a new Point without that field.
    shape = self["sparse_shape"] if "sparse_shape" in self else [
        value + pad for value in geometry.max_xyz
    ]
    sparse = spconv.SparseConvTensor(
        features=self.feat,
        indices=torch.cat(
            [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
        ).contiguous(),
        spatial_shape=shape,
        batch_size=1,
    )
    self["sparse_shape"] = shape
    self["sparse_conv_feat"] = sparse
    execution.owner.stats["sparsify_calls"] += 1


class GeometryPatch:
    """Own the removable hooks and original Point methods; no model state."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.original_init = Point.__init__
        self.original_serialization = Point.serialization
        self.original_sparsify = Point.sparsify
        self.hooks = []
        self.installed = True
        self.stats = dict(
            initial_max_host_reads=0,
            initial_geometry_cache_hits=0,
            pool_metadata_derivations=0,
            explicit_batch_constructions=0,
            explicit_offset_constructions=0,
            serialization_depth_reuses=0,
            sparsify_calls=0,
        )

    def _enter(self, module, args):
        _ACTIVE.set(_Execution(self, _ACTIVE.get()))

    def _leave(self, module, args, output):
        execution = _ACTIVE.get()
        if execution is not None and execution.owner is self:
            _ACTIVE.set(execution.previous)

    def _enter_pool(self, module, args):
        execution = _execution()
        if execution is None:
            return
        parent = args[0]
        if parent.offset.numel() != 1:
            raise ValueError("Geometry pool metadata optimization requires batch=1")
        execution.pools.append(_PoolScope(module, parent.get(_KEY), module.stride))

    def _leave_pool(self, module, args, output):
        execution = _execution()
        if execution is None or not execution.pools:
            return
        if execution.pools[-1].module is module:
            execution.pools.pop()

    def uninstall(self):
        global _INSTALLED
        if not self.installed:
            return
        if _execution() is not None:
            raise RuntimeError("Uninstall geometry metadata outside an encoder call")
        if (Point.__init__ is not _point_init
                or Point.serialization is not _serialization
                or Point.sparsify is not _sparsify):
            raise RuntimeError("A later Point method patch must be removed first")
        for hook in self.hooks:
            hook.remove()
        Point.__init__ = self.original_init
        Point.serialization = self.original_serialization
        Point.sparsify = self.original_sparsify
        self.hooks.clear()
        self.installed = False
        _INSTALLED = None


def patch(encoder):
    """Install one scoped batch-one metadata optimization and return its handle."""
    global _INSTALLED
    if _INSTALLED is not None:
        if _INSTALLED.encoder is encoder:
            return _INSTALLED
        raise RuntimeError("Only one geometry metadata installation per process is supported")
    if not hasattr(encoder, "coarse") or not hasattr(encoder, "fine"):
        raise TypeError("Pass the DualEncoder instance")
    pools = [module for module in encoder.modules() if isinstance(module, GridPooling)]
    if not pools:
        raise ValueError("Expected original GridPooling modules")
    if any(not isinstance(pool.stride, int) or pool.stride <= 0 for pool in pools):
        raise ValueError("Exact metadata propagation requires positive integer strides")

    handle = GeometryPatch(encoder)
    _INSTALLED = handle
    Point.__init__ = _point_init
    Point.serialization = _serialization
    Point.sparsify = _sparsify
    handle.hooks.append(encoder.register_forward_pre_hook(handle._enter))
    handle.hooks.append(encoder.register_forward_hook(handle._leave, always_call=True))
    for pool in pools:
        handle.hooks.append(pool.register_forward_pre_hook(handle._enter_pool))
        handle.hooks.append(pool.register_forward_hook(handle._leave_pool, always_call=True))
    return handle


def unpatch(handle):
    handle.uninstall()
