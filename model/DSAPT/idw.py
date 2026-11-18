"""Exact geometry-only IDW with first-order gradients into token features.

All coordinates must share the encoder centering and scale. Query normals do
not enter the distance. CUDA sparse interpolation streams indexed features;
global interpolation recomputes bounded query chunks during backward. Neither
path retains [B,Q,N] weights or [B,Q,k,C] feature gathers in autograd.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@dataclass(frozen=True)
class NeighborMap:
    indices: torch.Tensor
    weights: torch.Tensor
    squared_distances: torch.Tensor
    num_tokens: int
    backend: str
    power: float
    eps: float


if triton is not None:
    @triton.jit
    def _distance(Q, R, LENS, OUT, NQ, NR, QB, QP, RB, RP,
                  BQ: tl.constexpr, BR: tl.constexpr):
        qi = tl.program_id(0) * BQ + tl.arange(0, BQ)
        ri = tl.program_id(1) * BR + tl.arange(0, BR)
        b = tl.program_id(2)
        n = tl.load(LENS + b)
        qp = b * QB + qi * QP
        rp = b * RB + ri * RP
        qx = tl.load(Q + qp, qi < NQ, other=0).to(tl.float32)
        qy = tl.load(Q + qp + 1, qi < NQ, other=0).to(tl.float32)
        qz = tl.load(Q + qp + 2, qi < NQ, other=0).to(tl.float32)
        rx = tl.load(R + rp, ri < n, other=0).to(tl.float32)
        ry = tl.load(R + rp + 1, ri < n, other=0).to(tl.float32)
        rz = tl.load(R + rp + 2, ri < n, other=0).to(tl.float32)
        dx, dy, dz = qx[:, None] - rx[None, :], qy[:, None] - ry[None, :], qz[:, None] - rz[None, :]
        d2 = dx * dx + dy * dy + dz * dz
        d2 = tl.where(ri[None, :] < n, d2, float("inf"))
        off = (b * NQ + qi[:, None]) * NR + ri[None, :]
        tl.store(OUT + off, d2, (qi[:, None] < NQ) & (ri[None, :] < NR))

    @triton.jit
    def _normalize(D2, W, ROWS, K, P: tl.constexpr,
                   EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        k = tl.arange(0, BLOCK)
        d2 = tl.load(D2 + row * K + k, k < K, other=float("inf"))
        if P == 2.0:
            dp = d2
        else:
            dp = tl.exp(tl.log(d2) * (P * 0.5))
        inverse = 1.0 / (dp + EPS)
        inverse = tl.where(k < K, inverse, 0.0)
        zero = (d2 == 0.0) & (k < K)
        nz = tl.sum(zero.to(tl.float32), 0)
        denom = tl.sum(inverse, 0)
        weight = tl.where(nz > 0, zero.to(tl.float32) / tl.maximum(nz, 1.0),
                          inverse / tl.maximum(denom, 1.0e-30))
        tl.store(W + row * K + k, weight, k < K)

    @triton.jit
    def _sparse_forward(F, IDX, W, OUT, Q, N, C: tl.constexpr,
                        K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
        q = tl.program_id(0)
        b = tl.program_id(1)
        cs = tl.program_id(2) * BC + tl.arange(0, BC)
        ks = tl.arange(0, BK)
        index = tl.load(IDX + (b * Q + q) * K + ks, ks < K, other=0)
        weight = tl.load(W + (b * Q + q) * K + ks, ks < K, other=0)
        value = tl.load(F + (b * N + index[:, None]) * C + cs[None, :],
                        (ks[:, None] < K) & (cs[None, :] < C), other=0).to(tl.float32)
        out = tl.sum(value * weight[:, None], axis=0)
        tl.store(OUT + (b * Q + q) * C + cs, out, cs < C)

    @triton.jit
    def _sparse_backward(GO, IDX, W, GF, Q, N, C: tl.constexpr,
                         K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
        q = tl.program_id(0)
        b = tl.program_id(1)
        cs = tl.program_id(2) * BC + tl.arange(0, BC)
        ks = tl.arange(0, BK)
        index = tl.load(IDX + (b * Q + q) * K + ks, ks < K, other=0)
        weight = tl.load(W + (b * Q + q) * K + ks, ks < K, other=0)
        grad = tl.load(GO + (b * Q + q) * C + cs, cs < C, other=0).to(tl.float32)
        tl.atomic_add(GF + (b * N + index[:, None]) * C + cs[None, :],
                      grad[None, :] * weight[:, None],
                      (ks[:, None] < K) & (cs[None, :] < C), sem="relaxed")


def _require_triton(tensor):
    if triton is None or not tensor.is_cuda:
        raise RuntimeError("backend='triton' requires installed Triton and CUDA tensors")


def _geometry(query_xyz, token_xyz, lengths, k=1):
    if query_xyz.ndim not in (2, 3) or token_xyz.ndim != query_xyz.ndim:
        raise ValueError("Use [Q,3]/[N,3] or [B,Q,3]/[B,N,3] geometry")
    unbatched = query_xyz.ndim == 2
    q = query_xyz.detach().unsqueeze(0) if unbatched else query_xyz.detach()
    r = token_xyz.detach().unsqueeze(0) if unbatched else token_xyz.detach()
    if q.shape[-1] != 3 or r.shape[-1] != 3 or q.shape[0] != r.shape[0]:
        raise ValueError("Coordinates must have XYZ channels and matching batch size")
    if q.device != r.device or q.shape[1] == 0 or r.shape[1] < k:
        raise ValueError("Geometry device/count mismatch or fewer tokens than k")
    if lengths is None:
        lens = torch.full((r.shape[0],), r.shape[1], device=r.device, dtype=torch.long)
    else:
        lens = torch.as_tensor(lengths, device=r.device, dtype=torch.long).reshape(-1)
        if lens.shape != (r.shape[0],):
            raise ValueError("lengths must contain one valid suffix-padded token count per batch")
        # Caller must check k <= lengths <= N before timed execution; avoiding a
        # device-to-host synchronization here is intentional.
    return q.float().contiguous(), r.float().contiguous(), lens, unbatched


def _parameters(power, eps, query_chunk):
    if not math.isfinite(power) or power <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("IDW power and eps must be positive finite scalars")
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")


@torch.no_grad()
def squared_distances(query, reference, lengths, backend="triton"):
    """Batched FP32 direct squared distances; padding columns are infinity."""
    if backend == "triton":
        _require_triton(query)
        b, nq, _ = query.shape
        nr = reference.shape[1]
        result = torch.empty((b, nq, nr), device=query.device, dtype=torch.float32)
        _distance[(triton.cdiv(nq, 32), triton.cdiv(nr, 128), b)](
            query, reference, lengths, result, nq, nr,
            query.stride(0), query.stride(1), reference.stride(0), reference.stride(1),
            BQ=32, BR=128, num_warps=4, enable_fp_fusion=False)
        return result
    if backend != "cdist":
        raise ValueError(f"Unknown distance backend: {backend}")
    d2 = torch.cdist(query, reference, compute_mode="donot_use_mm_for_euclid_dist").square_()
    invalid = torch.arange(reference.shape[1], device=query.device)[None, None, :] >= lengths[:, None, None]
    d2.masked_fill_(invalid, float("inf"))
    return d2


@torch.no_grad()
def weights_from_squared_distances(d2, power=2.0, eps=1e-12, *, backend="auto", inplace=False):
    """Uniform among exactly coincident selected points, inverse distance otherwise."""
    _parameters(power, eps, 1)
    d2 = d2.float().contiguous()
    if backend == "auto":
        backend = "triton" if d2.is_cuda and triton is not None else "torch"
    if backend == "triton":
        _require_triton(d2)
        k = d2.shape[-1]
        result = d2 if inplace else torch.empty_like(d2)
        block = triton.next_power_of_2(k)
        _normalize[(d2.numel() // k,)](d2, result, d2.numel() // k, k,
                                     float(power), float(eps), block,
                                     num_warps=4 if block <= 1024 else 16)
        return result
    if backend != "torch":
        raise ValueError(f"Unknown normalization backend: {backend}")
    inverse = (d2 if power == 2 else d2.pow(power / 2)).add(eps).reciprocal()
    zeros = d2 == 0
    zero_count = zeros.sum(-1, keepdim=True)
    return torch.where(zero_count > 0, zeros / zero_count.clamp_min(1),
                       inverse / inverse.sum(-1, keepdim=True).clamp_min(1e-30))


@torch.no_grad()
def build_neighbors(query_xyz, token_xyz, k, *, lengths=None, backend="triton",
                    query_chunk=2048, power=2.0, eps=1e-12):
    """Fresh exact kNN search + normalization, without feature gather.

    Backends: triton direct squared distance + torch.topk, cdist direct CUDA
    distance + torch.topk, scipy CPU cKDTree (including transfers), pytorch3d
    optional installed CUDA KNN. Tied distance identities are interchangeable.
    lengths validity (k <= lengths <= N) must be checked by the caller once.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be positive")
    _parameters(power, eps, query_chunk)
    q, r, lens, unbatched = _geometry(query_xyz, token_xyz, lengths, k)
    shape = (*q.shape[:2], k)
    if backend in ("triton", "cdist"):
        indices = torch.empty(shape, device=q.device, dtype=torch.int64)
        values = torch.empty(shape, device=q.device, dtype=torch.float32)
        for start in range(0, q.shape[1], query_chunk):
            stop = min(start + query_chunk, q.shape[1])
            dist = squared_distances(q[:, start:stop], r, lens, backend)
            d, i = dist.topk(k, dim=-1, largest=False, sorted=False)
            values[:, start:stop] = d
            indices[:, start:stop] = i
            del dist, d, i
    elif backend == "scipy":
        import numpy as np
        from scipy.spatial import cKDTree
        qc, rc, lc = q.cpu().numpy(), r.cpu().numpy(), lens.cpu().numpy()
        idx = np.empty(shape, dtype=np.int64)
        for b, n in enumerate(lc):
            if not k <= n <= r.shape[1]:
                raise ValueError("Invalid lengths")
            _, selected = cKDTree(rc[b, :n]).query(qc[b], k=k, eps=0, p=2, workers=4)
            idx[b] = selected.reshape(q.shape[1], k)
        indices = torch.from_numpy(idx).to(q.device)
        # Recompute selected distances in FP32 for identical IDW arithmetic.
        batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
        delta = q[:, :, None, :] - r[batch, indices]
        values = delta.square().sum(-1)
    elif backend == "pytorch3d":
        from pytorch3d.ops import knn_points
        result = knn_points(q, r, lengths2=lens, K=k, return_sorted=False)
        values, indices = result.dists, result.idx
    else:
        raise ValueError(f"Unknown kNN backend: {backend}")
    weights = weights_from_squared_distances(values, power, eps)
    if unbatched:
        indices, weights, values = indices[0], weights[0], values[0]
    return NeighborMap(indices, weights, values, r.shape[1], backend, float(power), float(eps))


class _SparseInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, weights, backend):
        b, n, c = features.shape
        q, k = indices.shape[1:]
        result = torch.empty((b, q, c), device=features.device, dtype=features.dtype)
        if backend == "triton":
            block_k = triton.next_power_of_2(k)
            block_c = 32 if k >= 128 else 64
            _sparse_forward[(q, b, triton.cdiv(c, block_c))](
                features, indices, weights, result, q, n, c, k, block_k, block_c,
                num_warps=4 if k <= 32 else 8)
        else:
            batch = torch.arange(b, device=features.device)[:, None, None]
            # The reference is query-chunked, but intentionally not fused.
            for start in range(0, q, 128):
                stop = min(start + 128, q)
                gathered = features[batch, indices[:, start:stop]].float()
                result[:, start:stop] = (gathered * weights[:, start:stop, :, None]).sum(-2).to(features.dtype)
        ctx.save_for_backward(indices, weights)
        ctx.input_shape, ctx.input_dtype, ctx.backend = features.shape, features.dtype, backend
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        indices, weights = ctx.saved_tensors
        b, n, c = ctx.input_shape
        q, k = indices.shape[1:]
        grad_output = grad_output.contiguous()
        grad_features = torch.zeros(ctx.input_shape, dtype=torch.float32, device=grad_output.device)
        if ctx.backend == "triton":
            block_c = 32 if k >= 128 else 64
            _sparse_backward[(q, b, triton.cdiv(c, block_c))](
                grad_output, indices, weights, grad_features, q, n, c, k,
                triton.next_power_of_2(k), block_c, num_warps=4 if k <= 32 else 8)
        else:
            for start in range(0, q, 128):
                stop = min(start + 128, q)
                contribution = grad_output[:, start:stop, None, :].float() * weights[:, start:stop, :, None]
                scatter_index = indices[:, start:stop, :, None].expand(-1, -1, -1, c).reshape(b, -1, c)
                grad_features.scatter_add_(1, scatter_index, contribution.reshape(b, -1, c))
        return grad_features.to(ctx.input_dtype), None, None, None


def interpolate_knn(features, neighbors, *, backend="triton"):
    """[B,N,C] -> [B,Q,C], or unbatched equivalents; feature gradient only."""
    unbatched = features.ndim == 2
    f = features.unsqueeze(0) if unbatched else features
    idx = neighbors.indices.unsqueeze(0) if unbatched else neighbors.indices
    weights = neighbors.weights.unsqueeze(0) if unbatched else neighbors.weights
    if f.ndim != 3 or idx.ndim != 3 or f.shape[0] != idx.shape[0] or f.shape[1] != neighbors.num_tokens:
        raise ValueError("Features and neighbor map batch/token dimensions do not match")
    if f.device != idx.device or f.device != weights.device:
        raise ValueError("Features and neighbor map must be on the same device")
    if backend == "triton":
        _require_triton(f)
    elif backend != "torch":
        raise ValueError(f"Unknown interpolation backend: {backend}")
    output = _SparseInterpolate.apply(f.contiguous(), idx.contiguous(), weights.contiguous(), backend)
    return output[0] if unbatched else output


class _GlobalInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, query, reference, lengths, power, eps, query_chunk, distance_backend, matmul_dtype):
        b, n, c = features.shape
        nq = query.shape[1]
        output = torch.empty((b, nq, c), device=features.device, dtype=features.dtype)
        with torch.autocast(device_type=features.device.type, enabled=False):
            fm = features.to(matmul_dtype)
            for start in range(0, nq, query_chunk):
                stop = min(start + query_chunk, nq)
                weights = squared_distances(query[:, start:stop], reference, lengths, distance_backend)
                weights = weights_from_squared_distances(weights, power, eps, inplace=True)
                output[:, start:stop] = torch.bmm(weights.to(matmul_dtype), fm).to(features.dtype)
                del weights
        ctx.save_for_backward(query, reference, lengths)
        ctx.input_shape, ctx.input_dtype = features.shape, features.dtype
        ctx.parameters = power, eps, query_chunk, distance_backend, matmul_dtype
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, reference, lengths = ctx.saved_tensors
        power, eps, query_chunk, distance_backend, matmul_dtype = ctx.parameters
        gradient = torch.zeros(ctx.input_shape, dtype=torch.float32, device=grad_output.device)
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            for start in range(0, query.shape[1], query_chunk):
                stop = min(start + query_chunk, query.shape[1])
                weights = squared_distances(query[:, start:stop], reference, lengths, distance_backend)
                weights = weights_from_squared_distances(weights, power, eps, inplace=True)
                if matmul_dtype == torch.float32:
                    # addbmm reduces the batch dimension; baddbmm preserves it.
                    torch.baddbmm(gradient, weights.transpose(1, 2), grad_output[:, start:stop].float(),
                                 beta=1, alpha=1, out=gradient)
                else:
                    # Tensor cores accumulate each GEMM internally in FP32,
                    # then return the selected low-precision dtype. Accumulate
                    # the query-chunk contributions in an FP32 feature buffer.
                    part = torch.bmm(weights.to(matmul_dtype).transpose(1, 2),
                                     grad_output[:, start:stop].to(matmul_dtype))
                    gradient.add_(part.float())
                del weights
        return gradient.to(ctx.input_dtype), None, None, None, None, None, None, None, None


def interpolate_global(features, query_xyz, token_xyz, *, lengths=None, query_chunk=512,
                       power=2.0, eps=1e-12, distance_backend="triton", matmul_dtype="float32"):
    """All-token IDW with bounded workspace and recomputed backward.

    Distances/normalization use FP32. Default FP32 GEMMs obey the process CUDA
    matmul TF32 setting, which benchmarks should record. Optional matmul_dtype
    'float16', 'bfloat16', or 'input' casts weights before tensor-core GEMMs;
    compare its output AND gradient error against 'float32' independently.
    Autocast is disabled internally, even for BF16 features.
    Only first-order feature gradients are implemented. No feature values are
    saved for backward because IDW is a linear map of the features.
    """
    _parameters(power, eps, query_chunk)
    q, r, lens, unbatched = _geometry(query_xyz, token_xyz, lengths)
    f = features.unsqueeze(0) if unbatched else features
    if f.ndim != 3 or f.shape[:2] != r.shape[:2] or f.device != r.device:
        raise ValueError("Features and token geometry must share batch, count and device")
    if distance_backend == "triton":
        _require_triton(f)
    dtypes = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16, "input": features.dtype}
    if matmul_dtype not in dtypes or dtypes[matmul_dtype] not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("matmul_dtype must select float32, float16, or bfloat16")
    output = _GlobalInterpolate.apply(f, q, r, lens, float(power), float(eps), int(query_chunk),
                                      distance_backend, dtypes[matmul_dtype])
    return output[0] if unbatched else output


__all__ = ["NeighborMap", "build_neighbors", "interpolate_knn", "interpolate_global",
           "squared_distances", "weights_from_squared_distances"]
