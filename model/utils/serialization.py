import torch
import torch.nn.functional as F


# --------------------------
# 2D Morton(Z-order) 工具函数
# --------------------------

def _part1by1_64(n: torch.Tensor) -> torch.Tensor:
    """
    64-bit helper: separate bits with 1 zero: abcdef -> a0b0c0d0e0f...
    n: int64 tensor, only low 32 bits used.
    """
    n = n & 0x00000000FFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2))  & 0x3333333333333333
    n = (n | (n << 1))  & 0x5555555555555555
    return n


def morton2d_codes(u: torch.Tensor, v: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """
    计算 2D Morton(Z-order) 编码，用于截面 (u, v) 上的序列化。
    u, v: int64 tensors, 同形状 (B, M) 或 (M,)
    bits: 每个维度使用的 bit 数（<= 31）
    返回：int64 Morton code, 同形状
    """
    assert bits <= 31, "2D Morton 支持最多 31 bits per dim"

    x = u.long()
    y = v.long()

    xx = _part1by1_64(x)
    yy = _part1by1_64(y)

    return (xx << 1) | yy   # interleave bits


# --------------------------
# 构造流向-截面正交基
# --------------------------

def _build_surface_basis(inflow_dir: torch.Tensor) -> torch.Tensor:
    """
    根据来流方向 inflow_dir 构造一个 3D 正交基 (e1,e2,e3)：
      e1: 流向方向 (单位向量)
      e2, e3: 与 e1 正交的一对单位向量
    inflow_dir: (3,) 或 (B, 3)
    返回: basis: (B, 3, 3)，每个 batch 一个 [e1; e2; e3]
    """
    if inflow_dir.dim() == 1:
        d = inflow_dir.unsqueeze(0)
        squeeze_back = True
    else:
        d = inflow_dir
        squeeze_back = False

    B = d.shape[0]

    # e1: 归一化来流方向
    e1 = F.normalize(d, dim=-1)        # (B, 3)

    # 选一个与 e1 不平行的向量作为辅助 r
    z = torch.tensor([0., 0., 1.], device=d.device, dtype=d.dtype).expand(B, 3)
    x = torch.tensor([1., 0., 0.], device=d.device, dtype=d.dtype).expand(B, 3)

    use_z = (e1.abs() @ torch.tensor([0., 0., 1.], device=d.device, dtype=d.dtype)) < 0.9
    r = torch.where(use_z.unsqueeze(-1), z, x)  # (B, 3)

    # e2: 将 r 在 e1 的正交补中归一化
    proj = (r * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(r - proj, dim=-1)         # (B, 3)

    # e3: e1 × e2
    e3 = torch.cross(e1, e2, dim=-1)           # (B, 3)
    e3 = F.normalize(e3, dim=-1)

    basis = torch.stack([e1, e2, e3], dim=1)   # (B, 3, 3)

    if squeeze_back:
        basis = basis.squeeze(0)               # (3, 3)
    return basis


# --------------------------
# Flow-aware Surface Filling Curve
# --------------------------

def flow_aware_sfc(
    points: torch.Tensor,
    inflow_dir: torch.Tensor,
    num_bins: int = 32,
    bits: int = 16,
    snake: bool = True,
):
    """
    Flow-Aware Surface Filling Curve (FA-SFC)

    输入:
      points: (N, d) 或 (B, N, d)，其中 d >= 3，points[..., :3] 为 xyz 坐标，
              points[..., 3:] 可以是任意流场属性（压力、速度等），会一起被序列化。
      inflow_dir: 来流方向向量 (3,) 或 (B, 3)，不一定是单位向量
      num_bins: 沿流向 s 分成多少个带 (streamwise bands)
      bits: 截面 (u,v) Morton 量化用的 bits 数
      snake: 是否对相邻 band 采用正反交替(蛇形)连接，减少跨带跳跃

    输出:
      sorted_points: 按 FA-SFC 排序后的点和属性: 形状同 points
      sorted_idx   : 全局索引的排序: (B, N) 或 (N,)
      s_norm       : 每个点归一化的流向坐标 s ∈ [0,1]: (B, N) 或 (N,)
    """
    original_dim = points.dim()
    assert original_dim in (2, 3), "points 必须是 (N,d) 或 (B,N,d)"
    assert points.shape[-1] >= 3, "points 最后维度 d 必须 >= 3（前三维是坐标）"

    if original_dim == 2:
        pts = points.unsqueeze(0)        # (1, N, d)
        batch_mode = False
    else:
        pts = points                     # (B, N, d)
        batch_mode = True

    B, N, D = pts.shape
    device = pts.device
    dtype = pts.dtype

    # 只用前三维作为坐标参与几何/流向计算
    xyz = pts[..., :3]                   # (B, N, 3)

    # 1) 构造基 (e1,e2,e3): e1=流向, e2,e3=截面方向
    basis = _build_surface_basis(inflow_dir.to(device))  # (3,3) 或 (B,3,3)
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)                        # (1,3,3)
    if basis.shape[0] == 1 and B > 1:
        basis = basis.expand(B, 3, 3)                     # 广播成每个 batch 一样的来流

    e1 = basis[:, 0, :]   # 流向 (B,3)
    e2 = basis[:, 1, :]   # 截面 u 方向 (B,3)
    e3 = basis[:, 2, :]   # 截面 v 方向 (B,3)

    # 2) 投影到基上：得到 s,u,v
    s = (xyz * e1[:, None, :]).sum(dim=-1)   # (B,N), 流向坐标
    u = (xyz * e2[:, None, :]).sum(dim=-1)   # (B,N), 截面 u
    v = (xyz * e3[:, None, :]).sum(dim=-1)   # (B,N), 截面 v

    # 3) 每个 batch 内对 s 做 min-max 归一化 -> s_norm ∈ [0,1]
    s_min = s.amin(dim=1, keepdim=True)
    s_max = s.amax(dim=1, keepdim=True)
    s_range = (s_max - s_min).clamp(min=1e-9)
    s_norm = (s - s_min) / s_range           # (B,N)

    # 4) 按 s_norm 分成 num_bins 个流向带
    bin_id = (s_norm * num_bins).long().clamp(0, num_bins - 1)  # (B,N)

    # 存储最终的排序索引
    sorted_idx_all = torch.empty(B, N, dtype=torch.long, device=device)

    # 5) 对每个 batch 分别处理
    for b in range(B):
        # 当前 batch 的全局 index 0..N-1
        global_idx = torch.arange(N, device=device, dtype=torch.long)

        # 这个 batch 的 bin_id, u, v
        bin_b = bin_id[b]     # (N,)
        u_b = u[b]            # (N,)
        v_b = v[b]            # (N,)

        local_idx_pieces = []

        for band in range(num_bins):
            mask = (bin_b == band)
            if not mask.any():
                continue

            idx_in_band = global_idx[mask]    # (M,)
            u_band = u_b[mask]
            v_band = v_b[mask]

            # 归一化 u,v 到 [0,1] 后量化成整数
            u_min = u_band.min()
            u_max = u_band.max()
            u_range = (u_max - u_min).clamp(min=1e-9)
            u_norm = (u_band - u_min) / u_range

            v_min = v_band.min()
            v_max = v_band.max()
            v_range = (v_max - v_min).clamp(min=1e-9)
            v_norm = (v_band - v_min) / v_range

            scale = (1 << bits) - 1
            u_q = (u_norm * scale).clamp(0, scale).long()
            v_q = (v_norm * scale).clamp(0, scale).long()

            # 计算 2D Morton code
            codes_band = morton2d_codes(u_q, v_q, bits=bits)  # (M,)

            # bin 内排序
            order_local = torch.argsort(codes_band)           # (M,)
            idx_sorted_band = idx_in_band[order_local]        # (M,)

            # 蛇形连接: 偶数 band 正向, 奇数 band 反向
            if snake and (band % 2 == 1):
                idx_sorted_band = idx_sorted_band.flip(0)

            local_idx_pieces.append(idx_sorted_band)

        if len(local_idx_pieces) == 0:
            sorted_idx_b = global_idx
        else:
            sorted_idx_b = torch.cat(local_idx_pieces, dim=0)  # (N,)

        sorted_idx_all[b] = sorted_idx_b

    # 6) 用 sorted_idx_all 对原始点云+属性重排
    # pts: (B,N,D), sorted_idx_all: (B,N)
    sorted_points = pts.gather(
        dim=1,
        index=sorted_idx_all.unsqueeze(-1).expand(-1, -1, D)
    )  # (B,N,D)

    # 如果输入是 (N,d)，去掉 batch 维
    if not batch_mode:
        sorted_points = sorted_points.squeeze(0)   # (N,D)
        sorted_idx_all = sorted_idx_all.squeeze(0) # (N,)
        s_norm = s_norm.squeeze(0)                 # (N,)

    return sorted_points, sorted_idx_all, s_norm


def _part1by2_64(n: torch.Tensor) -> torch.Tensor:
    """
    64-bit version of "separate bits by 2 zeros": abcdef -> a00b00c00d00e00f...
    n: int64 tensor, assumed to use only low 21 bits
    """
    # 下面这些常数都是 standard Morton 扩展，适用于 64-bit 的 bit-spread
    n = n & 0x1fffff  # 只保留低 21 bits
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8))  & 0x100f00f00f00f00f
    n = (n | (n << 4))  & 0x10c30c30c30c30c3
    n = (n | (n << 2))  & 0x1249249249249249
    return n


def morton3d_codes(points: torch.Tensor, bits: int = 20) -> torch.Tensor:
    """
    Compute 3D Morton(Z-order) codes for points.
    points: (N, 3) or (B, N, 3), float tensor
    bits: number of bits per dimension (max 21 for current _part1by2_64)
    return: int64 codes, shape (N,) or (B, N)
    """
    assert bits <= 21, "Current implementation supports up to 21 bits per dim."

    orig_shape = points.shape
    assert orig_shape[-1] == 3, "points last dim must be 3 (x, y, z)"

    # (B, N, 3) or (N, 3) -> (B, N, 3)
    if points.dim() == 2:
        pts = points.unsqueeze(0)  # (1, N, 3)
    elif points.dim() == 3:
        pts = points
    else:
        raise ValueError("points must be (N, 3) or (B, N, 3)")

    B, N, _ = pts.shape

    # 1) 每个 batch 内做 min-max 归一化到 [0, 1]
    mins = pts.amin(dim=1, keepdim=True)
    maxs = pts.amax(dim=1, keepdim=True)
    ranges = (maxs - mins).clamp(min=1e-9)
    pts_norm = (pts - mins) / ranges  # (B, N, 3) ∈ [0,1]

    # 2) 量化到 [0, 2^bits - 1]
    scale = (1 << bits) - 1
    coords = (pts_norm * scale).clamp(0, scale).long()  # (B, N, 3), int64

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    xx = _part1by2_64(x)
    yy = _part1by2_64(y)
    zz = _part1by2_64(z)

    morton = (xx << 2) | (yy << 1) | zz  # (B, N), int64

    if points.dim() == 2:
        morton = morton.squeeze(0)  # (N,)
    return morton


def morton3d_sort(points: torch.Tensor, bits: int = 20):
    """
    Sort points by 3D Morton code (Z-order).
    points: (N, 3) or (B, N, 3)
    return:
        idx:   indices that sort the points, same shape as codes
        codes: morton codes, shape (N,) or (B,N)
    """
    codes = morton3d_codes(points[..., :3], bits=bits)
    idx = torch.argsort(codes, dim=-1)

    if points.dim() == 2:
        sorted_points = points.gather(
            dim=0,
            index=idx.unsqueeze(-1).expand(-1, points.shape[-1]),
        )
    else:
        sorted_points = points.gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )  # (B,N,D)
    return sorted_points, idx, codes


def _build_flow_aligned_basis(
    inflow_dir: torch.Tensor,
    ground_aligned: bool = True,
) -> torch.Tensor:
    """
    Build a coordinate basis whose first axis follows the inflow direction.

    For vehicle/train surfaces, `ground_aligned=True` keeps the global z axis
    as vertical and rotates only the horizontal x-y plane. This gives a Morton
    grid that is slanted with the incoming flow while preserving the usual
    vertical interpretation of the geometry.
    """
    if inflow_dir.dim() == 1:
        flow = inflow_dir.unsqueeze(0)
        squeeze_back = True
    else:
        flow = inflow_dir
        squeeze_back = False

    B = flow.shape[0]
    device = flow.device
    dtype = flow.dtype

    if ground_aligned:
        z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(B, 3)
        stream = flow.clone()
        stream[:, 2] = 0.0
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(B, 3)
        stream_norm = stream.norm(dim=-1, keepdim=True)
        stream = torch.where(stream_norm > 1e-8, stream, fallback)
        stream = F.normalize(stream, dim=-1)
        cross = torch.cross(z, stream, dim=-1)
        cross = F.normalize(cross, dim=-1)
        basis = torch.stack([stream, cross, z], dim=1)
    else:
        basis = _build_surface_basis(flow)
        if basis.dim() == 2:
            basis = basis.unsqueeze(0)

    if squeeze_back:
        basis = basis.squeeze(0)
    return basis


def flow_aligned_coordinates(
    points: torch.Tensor,
    inflow_dir: torch.Tensor,
    ground_aligned: bool = True,
) -> torch.Tensor:
    """
    Rotate xyz coordinates into a flow-aligned frame.

    Output axes are `(streamwise, cross-flow, vertical/local-normal-frame)`.
    The function preserves any non-coordinate channels by returning only the
    transformed xyz coordinates; callers can use the sort indices to reorder
    full point features.
    """
    original_dim = points.dim()
    if original_dim == 2:
        pts = points.unsqueeze(0)
        batch_mode = False
    elif original_dim == 3:
        pts = points
        batch_mode = True
    else:
        raise ValueError("points must be (N, D) or (B, N, D).")

    xyz = pts[..., :3]
    basis = _build_flow_aligned_basis(inflow_dir.to(device=xyz.device, dtype=xyz.dtype), ground_aligned)
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)
    if basis.shape[0] == 1 and xyz.shape[0] > 1:
        basis = basis.expand(xyz.shape[0], 3, 3)

    coords = torch.einsum("bnc,bkc->bnk", xyz, basis)
    if not batch_mode:
        coords = coords.squeeze(0)
    return coords


def flow_aligned_morton3d_codes(
    points: torch.Tensor,
    inflow_dir: torch.Tensor,
    bits: int = 20,
    ground_aligned: bool = True,
) -> torch.Tensor:
    """
    Morton codes after rotating coordinates into a flow-aligned frame.

    Unlike `flow_aware_sfc`, this does not split the domain into streamwise
    bands. It keeps a single global 3D Morton grid, only rotated by inflow.
    """
    coords = flow_aligned_coordinates(points, inflow_dir, ground_aligned=ground_aligned)
    return morton3d_codes(coords, bits=bits)


def flow_aligned_morton3d_sort(
    points: torch.Tensor,
    inflow_dir: torch.Tensor,
    bits: int = 20,
    ground_aligned: bool = True,
):
    """
    Sort points with a global 3D Morton curve in flow-aligned coordinates.
    """
    codes = flow_aligned_morton3d_codes(
        points[..., :3],
        inflow_dir,
        bits=bits,
        ground_aligned=ground_aligned,
    )
    idx = torch.argsort(codes, dim=-1)
    if points.dim() == 2:
        sorted_points = points.gather(
            dim=0,
            index=idx.unsqueeze(-1).expand(-1, points.shape[-1]),
        )
    else:
        sorted_points = points.gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]),
        )
    return sorted_points, idx, codes


# 预留 Hilbert 接口，将来你可以把 morton3d_codes 换掉，保持签名不变
def hilbert3d_codes_placeholder(points: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """
    占位函数：目前先直接调用 Morton 作为近似。
    如果你后面实现了真正的 Hilbert 编码，只需要替换这里的实现即可。
    """
    # TODO: replace with real Hilbert 3D coding
    return morton3d_codes(points, bits=bits)


def hilbert3d_sort(points: torch.Tensor, bits: int = 16):
    """
    Hilbert-like sort: 目前用 Morton 占位，将来可替换为真正 Hilbert。
    """
    codes = hilbert3d_codes_placeholder(points, bits=bits)
    idx = torch.argsort(codes, dim=-1)
    return idx, codes


if __name__ == "__main__":
    # 小测试：在 GPU 上运行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N = 2, 4096
    pts = torch.randn(B, N, 3, device=device)

    idx, codes = morton3d_sort(pts, bits=20)
    print("idx shape:", idx.shape, "codes shape:", codes.shape)

    # 简单测试
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, N = 2, 4096
    pts = torch.randn(B, N, 3, device=device)
    inflow = torch.tensor([1.0, 0.0, 0.0], device=device)  # 来流方向

    sorted_pts, sorted_idx, s_norm = flow_aware_sfc(
        pts, inflow, num_bins=32, bits=12, snake=True
    )
    print("sorted_pts:", sorted_pts.shape)
    print("sorted_idx:", sorted_idx.shape)
    print("s_norm:", s_norm.shape)
