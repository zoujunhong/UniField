"""Remove redundant PointSequential/SparseConvTensor dispatch in PTv3 blocks.

This is an execution-only, instance-local patch. It does not replace child
modules, change parameters/buffers, change attention, or compile/fuse arithmetic.
The ten ``replace_feature`` calls in an official Block become one, at its exit.
The spconv output and all tensor operations retain their original order.

Usage (after the existing attention/MLP execution patches and ``encoder.eval``)::

    from .optimized_block import patch
    handle = patch(encoder)
    # ... run the already-created encoder ...
    handle.uninstall()

The current Tiny dual encoder keeps its backbones in eval even while training
LoRA. Gradients are fully supported: eval here only fixes stochastic behavior.
This patch intentionally requires eval Block/descendants and rejects unknown
Point containers, normalization classes, in-place LayerScale, custom Block
forwards, and hooks below a Block. Block/stage hooks still execute normally.

Sparse features are stale inside a Block after CPE. Therefore attention and
tensor children must not access/modify ``point.sparse_conv_feat`` through aliases
or closures. The official and current local/global attention wrappers satisfy
this requirement. An additional wrapper may be explicitly certified by passing
its unbound function in ``attention_forwards``. It must use only dense features
and geometry/serialization/context there, must preserve sparse metadata, and
must return the same Point. Arbitrary Point snapshots/intermediate SparseTensor
identity observations, global module hooks, tracing and compilation of the
entire Block are outside this patch's contract.

Keep module structure, forwards, hooks and eval mode fixed while installed;
uninstall before changing them. Existing compiled tensor attention/MLP forwards
are left intact. Use state_dict for checkpoints and uninstall before deepcopy,
pickling the full Module, or replacing children: closures cache child references.
This file performs no model construction or GPU work on import.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable


# Temp scripts must resolve repository modules independently of their cwd.


def _function(value: Any) -> Any:
    return getattr(value, "__func__", value)


def _has_hooks(module: Any) -> bool:
    return any(
        bool(getattr(module, key, None))
        for key in (
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
            "_backward_pre_hooks",
        )
    )


@dataclass
class _Record:
    name: str
    block: Any
    had_instance_forward: bool
    original_instance_forward: Any
    installed_forward: Any


class BlockPatch:
    """Own an installed patch; restoring forwards leaves state_dict unchanged."""

    def __init__(self, records: list[_Record]):
        self._records = records
        self.active = True

    @property
    def block_count(self) -> int:
        return len(self._records)

    @property
    def block_names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    def uninstall(self) -> None:
        if not self.active:
            return
        # Refuse to overwrite a separate patch installed after this one.
        for record in self._records:
            if record.block.__dict__.get("forward") is not record.installed_forward:
                raise RuntimeError(
                    f"Block {record.name!r} forward changed while patch was active"
                )
        for record in reversed(self._records):
            if record.had_instance_forward:
                record.block.forward = record.original_instance_forward
            else:
                delattr(record.block, "forward")
        self.active = False

    def __enter__(self) -> "BlockPatch":
        if not self.active:
            raise RuntimeError("An uninstalled BlockPatch cannot be reused")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.uninstall()


def _make_forward(children: tuple[Any, ...]) -> Callable:
    conv, linear, cpe_norm, norm1, norm2, ls1, ls2, mlp, drop_path, attn = children

    def forward(self: Any, point: Any) -> Any:
        if self.training:
            raise RuntimeError("optimized_block requires eval-mode Blocks")

        shortcut = point.feat
        point.sparse_conv_feat = conv(point.sparse_conv_feat)
        point.feat = point.sparse_conv_feat.features
        point.feat = linear(point.feat)
        point.feat = cpe_norm(point.feat)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point.feat = norm1(point.feat)
        point = attn(point)
        point.feat = ls1(point.feat)
        point.feat = drop_path(point.feat)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point.feat = norm1(point.feat)

        shortcut = point.feat
        if self.pre_norm:
            point.feat = norm2(point.feat)
        point.feat = mlp(point.feat)
        point.feat = ls2(point.feat)
        point.feat = drop_path(point.feat)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point.feat = norm2(point.feat)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

    return forward


def patch(
    encoder: Any,
    *,
    attention_forwards: Iterable[Callable] = (),
) -> BlockPatch:
    """Patch official eval Blocks after validating the whole encoder atomically.

    ``attention_forwards`` explicitly certifies extra attention wrappers under
    the dense-only contract described in this module's docstring. The default
    allows official attention, loaded local_compile.forward, and the current
    network.global_forward. The underlying tensor attention may already be
    compiled. No dtype, RNG, autocast, DropPath probability, LayerScale value,
    model configuration, module registration, parameter or buffer is modified.

    Only the ``forward`` attribute of validated Block instances is assigned.
    No class/global monkeypatch is installed, and state_dict keys are unchanged.
    """
    import torch.nn as nn
    import torch.nn.modules.module as module_impl
    import spconv.pytorch as spconv
    from timm.layers import DropPath
    from model.third_party.utonia_dsapt.model import Block, LayerScale, MLP, SerializedAttention
    from model.third_party.utonia_dsapt.module import PointSequential

    global_hook_keys = (
        "_global_forward_hooks", "_global_forward_pre_hooks",
        "_global_backward_hooks", "_global_backward_pre_hooks",
    )
    if any(bool(getattr(module_impl, key, None)) for key in global_hook_keys):
        raise ValueError("Global module hooks would observe skipped containers")

    allowed_attention = {_function(SerializedAttention.forward)}
    for module_name, name in (
        ("model.DSAPT.local_compile", "forward"), ("model.DSAPT.network", "global_forward"),
    ):
        module = sys.modules.get(module_name)
        candidate = getattr(module, name, None) if module is not None else None
        if candidate is not None:
            allowed_attention.add(_function(candidate))
    allowed_attention.update(_function(fn) for fn in attention_forwards)

    pending: list[tuple[str, Any, tuple[Any, ...]]] = []
    for name, block in encoder.named_modules():
        if not isinstance(block, Block):
            continue
        if type(block) is not Block or _function(block.forward) is not Block.forward:
            raise ValueError(f"{name}: only the unmodified official Block is supported")
        for child_name, child in block.named_modules():
            if child.training:
                raise ValueError(f"{name}.{child_name}: eval mode is required")
            if child_name and _has_hooks(child):
                raise ValueError(f"{name}.{child_name}: descendant module hooks are unsupported")

        def parts(key: str, length: int) -> tuple[Any, ...]:
            container = getattr(block, key)
            if type(container) is not PointSequential or len(container) != length:
                raise ValueError(f"{name}.{key}: expected PointSequential of length {length}")
            if _function(container.forward) is not PointSequential.forward:
                raise ValueError(f"{name}.{key}: custom PointSequential.forward is unsupported")
            return tuple(container._modules.values())

        conv, linear, cpe_norm = parts("cpe", 3)
        norm1, = parts("norm1", 1)
        norm2, = parts("norm2", 1)
        ls1, = parts("ls1", 1)
        ls2, = parts("ls2", 1)
        mlp, = parts("mlp", 1)
        drop_path, = parts("drop_path", 1)
        if type(conv) is not spconv.SubMConv3d or type(linear) is not nn.Linear:
            raise ValueError(f"{name}: unsupported CPE structure")
        if any(type(norm) is not nn.LayerNorm for norm in (cpe_norm, norm1, norm2)):
            raise ValueError(f"{name}: only dense LayerNorm is supported")
        if type(mlp) is not MLP:
            raise ValueError(f"{name}: expected the original tensor MLP")
        for key, scale in (("ls1", ls1), ("ls2", ls2)):
            if type(scale) not in (nn.Identity, LayerScale):
                raise ValueError(f"{name}.{key}: unsupported LayerScale")
            if isinstance(scale, LayerScale) and scale.inplace:
                raise ValueError(f"{name}.{key}: in-place LayerScale is outside the strict contract")
        if type(drop_path) not in (nn.Identity, DropPath):
            raise ValueError(f"{name}: unsupported DropPath")
        if type(block.attn) is not SerializedAttention:
            raise ValueError(f"{name}: expected SerializedAttention")
        if _function(block.attn.forward) not in allowed_attention:
            raise ValueError(
                f"{name}: attention forward is unknown; certify dense-only wrappers "
                "with attention_forwards=(the_unbound_forward,)"
            )
        pending.append((name, block, (
            conv, linear, cpe_norm, norm1, norm2, ls1, ls2, mlp,
            drop_path, block.attn,
        )))

    if not pending:
        raise ValueError("No official PTv3 Blocks were found")
    records = []
    for name, block, children in pending:
        installed = MethodType(_make_forward(children), block)
        records.append(_Record(
            name, block, "forward" in block.__dict__,
            block.__dict__.get("forward"), installed,
        ))
        block.forward = installed
    return BlockPatch(records)


def unpatch(handle: BlockPatch) -> None:
    """Convenience alias for ``handle.uninstall()``."""
    handle.uninstall()
