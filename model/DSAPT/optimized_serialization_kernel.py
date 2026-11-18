"""Fuse Utonia's integer serialization encoding, preserving its sorting/RNG.

This is an isolated experiment patch, not a change to the official source or a
running training process.  ``patch(encoder)`` replaces only the ``encode`` name
resolved by the original ``utonia.structure.Point.serialization`` function.
Its depth selection, argsort, inverse scatter, randperm and shuffle stay eager
and unchanged.  ``unpatch()`` restores the original encode binding.

The Hilbert transformation follows the official bit-array implementation in
``official/utonia/serialization/hilbert.py`` (Skilling's transformation), but
stores each coordinate's bits in an int64 tensor.  There are no floating-point
operations, gradients, random calls, tensor-valued Python conditions or scalar
device reads in the encoding kernel.  torch.compile specializes on the Python
depth/order; only tensor dimension 0 is explicitly marked dynamic (N>1).
The N=1 boundary keeps its required singleton specialization. Compilation is lazy.

The worker already raises Dynamo's specialization cache limits.  At most 64
depth/order kernels are needed for the supported depths 1..16 and four orders;
workers using this module separately should allow this specialization count.
GPU integer equivalence, model equivalence and performance were measured in this
experiment; see report.md for the exact tested scope and results.
"""

from functools import lru_cache

import torch


_ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
_ORIGINAL_ENCODE = None
_INSTALLED_ENCODE = None


def _interleave_xyz(x, y, z, depth):
    """Same bit positions as the official KeyLUT.xyz2key construction."""
    code = torch.zeros_like(x)
    for bit in range(depth):
        mask = 1 << bit
        code = (
            code
            | ((x & mask) << (2 * bit + 2))
            | ((y & mask) << (2 * bit + 1))
            | ((z & mask) << (2 * bit))
        )
    return code


def _hilbert_xyz(x, y, z, depth):
    """Exact packed equivalent of official MSB-first bit/dimension loops.

    The official dimension loop does, in order:
      1. If the selected coordinate bit is 1, XOR lower bits of coordinate 0.
      2. If it is 0, exchange lower bits of coordinate 0 and this coordinate.
    ``lower`` is exactly that bit-array suffix.  Dimension 0's exchange is a
    no-op, so it needs only step 1.  The least-significant bit has an empty
    suffix and therefore does not need an iteration.
    """
    for bit in range(depth - 1, 0, -1):
        lower = (1 << bit) - 1

        active = (x >> bit) & 1
        x = x ^ (active * lower)

        active = (y >> bit) & 1
        x = x ^ (active * lower)
        exchange = ((x ^ y) & lower) * (1 - active)
        y = y ^ exchange
        x = x ^ exchange

        active = (z >> bit) & 1
        x = x ^ (active * lower)
        exchange = ((x ^ z) & lower) * (1 - active)
        z = z ^ exchange
        x = x ^ exchange

    # Official gray.swapaxes(1, 2).reshape(...) packs bits as x_MSB, y_MSB,
    # z_MSB, x_next, ...; this is the same XYZ interleave as Morton encoding.
    code = _interleave_xyz(x, y, z, depth)
    # Match gray2binary's descending powers of two, without its CPU tensor
    # log2/ceil or tensor-controlled while loop.
    shift = 1 << ((3 * depth - 1).bit_length() - 1)
    while shift > 0:
        code = code ^ (code >> shift)
        shift //= 2
    return code


@lru_cache(maxsize=128)
def _kernel_for(depth, order, compile_kernel):
    transpose = order.endswith("-trans")
    hilbert = order.startswith("hilbert")

    def kernel(grid_coord, batch):
        # Both official Morton LUT lookup and Hilbert's final -num_bits slice
        # retain only the low depth bits, including for signed input integers.
        mask = (1 << depth) - 1
        x = grid_coord[:, 1 if transpose else 0].long() & mask
        y = grid_coord[:, 0 if transpose else 1].long() & mask
        z = grid_coord[:, 2].long() & mask
        if hilbert:
            code = _hilbert_xyz(x, y, z, depth)
            # Official Hilbert returns a scalar for N=1 before batch bits are
            # added.  Preserve that standalone helper behavior as well.
            code = code.squeeze()
        else:
            code = _interleave_xyz(x, y, z, depth)
        if batch is not None:
            code = (batch.long() << (depth * 3)) | code
        return code

    # In PyTorch 2.6 dynamic=True also turns closure integers such as depth into
    # SymInts; Python (1 << depth) then fails in symbolic shape arithmetic.
    # dynamic=False disables automatic scalar/shape generalization, while the
    # explicit mark_dynamic calls below still override it for tensor dimension
    # 0.  Thus every cached depth/order remains static and only N is symbolic.
    return torch.compile(kernel, fullgraph=True, dynamic=False) if compile_kernel else kernel


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z", *, compile_kernel=True):
    """Official Point encode contract, with exact packed integer operations.

    ``compile_kernel=False`` exposes the same packed implementation eagerly for
    a separate diagnostic; the official implementation remains the reference.
    Only the current Point contract [N,3], integer coordinates, depth 1..16 is
    supported.  This function intentionally does not change depth selection.
    """
    if order not in _ORDERS:
        raise ValueError(f"Unsupported serialization order: {order}")
    if not isinstance(depth, int) or not 1 <= depth <= 16:
        raise ValueError(f"Expected Python integer depth in [1,16], got {depth!r}")
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError("Expected grid_coord with shape [N,3]")
    if grid_coord.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected int32/int64 grid coordinates")
    if compile_kernel and grid_coord.shape[0] > 1:
        torch._dynamo.mark_dynamic(grid_coord, 0)
        if batch is not None and batch.ndim > 0 and batch.shape[0] > 1:
            torch._dynamo.mark_dynamic(batch, 0)
    return _kernel_for(depth, order, compile_kernel)(grid_coord, batch)


def patch(encoder=None, *, compile_kernel=True):
    """Install encoding fusion for this Python process; encoder is unchanged.

    Accepting ``encoder`` makes this usable by the experiment's patch dispatch,
    but the binding is process-wide and affects every Point.serialization call
    in this isolated worker.  Calling patch repeatedly updates compile mode.
    No model parameters/state_dict keys or RNG state are touched.
    """
    global _ORIGINAL_ENCODE, _INSTALLED_ENCODE
    import model.third_party.utonia_dsapt.structure as structure

    if _ORIGINAL_ENCODE is None:
        _ORIGINAL_ENCODE = structure.encode

    def installed(grid_coord, batch=None, depth=16, order="z"):
        return encode(
            grid_coord,
            batch=batch,
            depth=depth,
            order=order,
            compile_kernel=compile_kernel,
        )

    structure.encode = installed
    _INSTALLED_ENCODE = installed
    return {
        "target": "utonia.structure.encode",
        "compile_kernel": compile_kernel,
        "depths": [1, 16],
        "orders": list(_ORDERS),
        "sort_inverse_shuffle": "original Point.serialization",
    }


def unpatch():
    """Restore only this module's active encode binding; return whether restored."""
    global _INSTALLED_ENCODE
    import model.third_party.utonia_dsapt.structure as structure

    if _INSTALLED_ENCODE is not None and structure.encode is _INSTALLED_ENCODE:
        structure.encode = _ORIGINAL_ENCODE
        _INSTALLED_ENCODE = None
        return True
    return False
