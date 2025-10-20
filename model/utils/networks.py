import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_func

from torch_cluster import knn as tc_knn

def balanced_patch_partition_batched(x, centers, k=32, f=None):
    """
    使用 torch-scatter 加速每个点分配到最近中心点。
    x: (B, N, D)
    centers: (B, M, D)
    f: (B, N, C), optional
    返回：
        grouped_x: (B, M, k, D)
        grouped_f: (B, M, k, C)
    """
    B, N, D = x.shape
    M = centers.shape[1]
    assert N == M * k

    grouped_x = torch.zeros(B, M, k, D, device=x.device, dtype=x.dtype)
    grouped_f = torch.zeros(B, M, k, f.shape[2], device=f.device, dtype=f.dtype) if f is not None else None

    for b in range(B):
        xb = x[b]            # (N, D)
        cb = centers[b]      # (M, D)
        fb = f[b] if f is not None else None

        # Step 1: 为每个点找到最近的中心点
        # (N, M) 距离矩阵
        dist = torch.cdist(xb.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)  # (N, M)
        nearest_center = torch.argmin(dist, dim=1)  # (N,) 每个点归属哪个中心

        # Step 2: 用 scatter 构建反向索引
        # grouped_x[b, cid] 将被这些点聚合
        # scatter 不支持变长 -> 用排序方式模拟 padding k 个点

        # 先按中心 id 排序（为了更容易分块）
        sorted_cid, sort_idx = torch.sort(nearest_center)
        xb_sorted = xb[sort_idx]              # (N, D)
        if f is not None:
            fb_sorted = fb[sort_idx]          # (N, C)

        # 然后按中心点分块：将每个中心对应的点按顺序填入
        for cid in range(M):
            idx_range = (sorted_cid == cid).nonzero(as_tuple=False).squeeze()
            idx_range = idx_range[:k]  # 截断最多 k 个点
            if idx_range.numel() < k:
                # padding with repeat
                pad_num = k - idx_range.numel()
                pad_idx = idx_range[torch.randint(0, idx_range.numel(), (pad_num,), device=x.device)]
                idx_range = torch.cat([idx_range, pad_idx], dim=0)

            grouped_x[b, cid] = xb_sorted[idx_range]
            if f is not None:
                grouped_f[b, cid] = fb_sorted[idx_range]

    return grouped_x, grouped_f

def voxel_hash(points, voxel_size):
    """将点投影到 voxel 网格上"""
    return torch.floor(points / voxel_size).int()

def voxel_knn(x, query, k=16, voxel_size=None, radius_factor=1.5):
    """
    使用 voxel 哈希加速的近似 knn 搜索

    x: (N, 3) - 数据点云
    query: (M, 3) - 查询点
    k: 每个查询点返回的近邻数
    voxel_size: optional float，如果为 None 自动估计
    radius_factor: 扩展邻域半径（单位是 voxel）
    """
    N, M = x.shape[0], query.shape[0]
    device = x.device

    # Step 0: 自动估计 voxel_size（如果没给）
    if voxel_size is None:
        min_coords = x.min(dim=0)[0]
        max_coords = x.max(dim=0)[0]
        span = max_coords - min_coords  # (3,)
        longest_axis = span.max()
        voxel_size = longest_axis / 16.0
        print(f"[Auto Voxel] voxel_size set to {voxel_size:.4f}")

    # Step 1: 哈希 voxel
    x_vox = voxel_hash(x, voxel_size)        # (N, 3)
    q_vox = voxel_hash(query, voxel_size)    # (M, 3)

    # Step 2: 构建哈希表（键 → 点索引列表）
    from collections import defaultdict
    voxel_dict = defaultdict(list)
    for i in range(N):
        key = tuple(x_vox[i].tolist())
        voxel_dict[key].append(i)

    # Step 3: 邻域 offset（3×3×3）
    neighbor_offsets = torch.stack(torch.meshgrid([
        torch.arange(-radius_factor, radius_factor + 1),
        torch.arange(-radius_factor, radius_factor + 1),
        torch.arange(-radius_factor, radius_factor + 1),
    ], indexing="ij"), dim=-1).reshape(-1, 3).to(device)  # (P, 3)

    knn_idx = torch.zeros((M, k), dtype=torch.long, device=device)

    for i in range(M):
        qkey = q_vox[i]
        candidate_idxs = []

        for offset in neighbor_offsets:
            neighbor_key = tuple((qkey + offset).tolist())
            candidate_idxs += voxel_dict.get(neighbor_key, [])

        if len(candidate_idxs) == 0:
            # fallback 全局
            dists = torch.norm(x - query[i], dim=1)
            idx = torch.topk(dists, k, largest=False)[1]
        else:
            candidates = x[candidate_idxs]  # (C, 3)
            dists = torch.norm(candidates - query[i], dim=1)  # (C,)
            topk = torch.topk(dists, k=min(k, len(dists)), largest=False)[1]
            idx = torch.tensor([candidate_idxs[j.item()] for j in topk], device=device)
            if len(idx) < k:
                pad = idx[torch.randint(0, len(idx), (k - len(idx),))]
                idx = torch.cat([idx, pad], dim=0)

        knn_idx[i] = idx

    return knn_idx  # (M, k)

def knn(x, query, k):
    """
    x: (B, N, 3)       输入点云
    query: (B, S, 3)   查询点（slots）
    return: idx (B, S, k)  每个查询点的 kNN 索引
    """
    B, N, _ = x.shape
    S = query.shape[1]
    dists = torch.cdist(query, x)  # (B, S, N)
    idx = dists.topk(k, largest=False)[1]  # (B, S, k)
    return idx

def knn_interpolation(dst_xyz, src_xyz, src_feat, k=3):
    """
    k-NN 插值：将稀疏点云上的特征插值到目标点集上

    Args:
        dst_xyz: (B, N, 3) - 目标点集坐标
        src_xyz: (B, M, 3) - 源点集坐标（有特征）
        src_feat: (B, C, M) - 源点集特征
        k: int - 使用的最近邻数量
    
    Returns:
        interpolated_feat: (B, C, N) - 插值后的特征
    """
    src_feat = src_feat.permute(0,2,1).contiguous()
    B, N, _ = dst_xyz.shape
    _, M, _ = src_xyz.shape
    _, C, _ = src_feat.shape

    # 计算 pairwise 距离: (B, N, M)
    dist = torch.cdist(dst_xyz, src_xyz, p=2)  # 欧几里得距离

    # 找到 k 个最近邻：索引和距离 (B, N, k)
    dists, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)

    # 避免除以 0：最小为 1e-10
    dists = torch.clamp(dists, min=1e-10)
    weights = 1.0 / dists  # (B, N, k)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # 归一化权重

    # 获取对应的特征值 (B, C, N, k)
    idx_expand = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, k)
    src_feat_expand = src_feat.unsqueeze(2).expand(-1, -1, N, -1)  # (B, C, N, M)
    neighbor_feats = torch.gather(src_feat_expand, dim=3, index=idx_expand)  # (B, C, N, k)

    # 插值结果 (B, C, N)
    weights = weights.unsqueeze(1)  # (B, 1, N, k)
    interpolated_feat = torch.sum(neighbor_feats * weights, dim=-1)

    return interpolated_feat.permute(0,2,1).contiguous()

# def knn_interpolation(dst_xyz, src_xyz, src_feat, k=3):
#     """
#     使用 knn_torch_cluster 插值，将 src_feat 从 src_xyz 插值到 dst_xyz 上。

#     参数:
#         dst_xyz: (B, N, 3) - 目标点坐标
#         src_xyz: (B, M, 3) - 源点坐标
#         src_feat: (B, M, C) - 源点特征 (注意：特征在最后一维)
#         k: int - 邻居数量

#     返回:
#         interpolated_feat: (B, N, C) - 插值后的特征
#     """
#     B, N, _ = dst_xyz.shape
#     _, M, _ = src_xyz.shape
#     _, Mf, C = src_feat.shape
#     assert M == Mf, "源点数量不匹配"

#     # knn 查找：在 src 中查找每个 dst 点的 k 个邻居 (B, N, k)
#     idx = knn(src_xyz, dst_xyz, k=k)  # (B, N, k)

#     # 取出邻居特征 (B, N, k, C)
#     idx_feat = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, k, C)
#     neighbor_feats = torch.gather(src_feat.unsqueeze(1).expand(-1, N, -1, -1), dim=2, index=idx_feat)  # (B, N, k, C)

#     # 取出邻居坐标 (B, N, k, 3)
#     idx_coord = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
#     neighbor_coords = torch.gather(src_xyz.unsqueeze(1).expand(-1, N, -1, -1), dim=2, index=idx_coord)

#     # 目标坐标扩展为 (B, N, 1, 3)
#     dst_coords = dst_xyz.unsqueeze(2)

#     # 欧氏距离 (B, N, k)
#     dists = torch.norm(dst_coords - neighbor_coords, dim=-1)
#     dists = torch.clamp(dists, min=1e-10)

#     # 权重归一化 (B, N, k)
#     weights = 1.0 / dists
#     weights = weights / weights.sum(dim=-1, keepdim=True)

#     # 加权求和：输出 (B, N, C)
#     weights = weights.unsqueeze(-1)  # (B, N, k, 1)
#     interpolated_feat = torch.sum(neighbor_feats * weights, dim=2)

#     return interpolated_feat.contiguous()



def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu', weights_only=True)
    model.load_state_dict(dict, strict=True)
    stop_grad(model)

def stop_grad(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
        
class LayerNorm2D(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 dim,
                 affine=True):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=affine)

    def forward(self, x: torch.Tensor): # x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(b,c,h,w).contiguous()
        return x

class RMSNorm2D(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 dim,
                 affine=True):
        super(RMSNorm2D, self).__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=affine)

    def forward(self, x: torch.Tensor): # x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(b,c,h,w).contiguous()
        return x

class ConvNorm(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel=1,
                 stride=1,
                 padding=0,
                 affine=True,
                 act=True):
        super(ConvNorm, self).__init__()
        self.norm = LayerNorm2D(outplanes, affine=affine)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel, stride, padding)
        self.relu = nn.GELU()
        self.act = act

    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.relu(x) if self.act else x

class LinearNorm(nn.Module):
    def __init__(self, inplanes, planes, act=True, affine=True):
        super().__init__()
        self.norm = nn.RMSNorm(planes, elementwise_affine=affine)
        self.conv = nn.Linear(inplanes, planes)
        self.act = act
    def forward(self, x):
        x = self.norm(self.conv(x))
        return F.gelu(x) if self.act else x


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: [B, N, 3] - point cloud
        npoint: int - number of points to sample
    Return:
        centroids: [B, npoint] - sampled point indices
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), float('inf'), device=xyz.device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest  # 更新采样点
        centroid_xyz = xyz[batch_indices, farthest, :].unsqueeze(1)  # [B, 1, 3]
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]

    return centroids  # 每个batch中的采样点索引

class FFN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, dpr=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )
        self.dp = DropPath(dpr)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(x + self.dp(self.encoder(x)))
    
def QuietSoftmax(x, dim=-1):
    x = x - torch.max(x)
    x = torch.exp(x)
    return x / (1 + torch.sum(x, dim=dim, keepdim=True))

def HardSoftmax(x, dim=-1):
    y_soft = F.softmax(x, dim=dim)
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(x).scatter_(dim, index, 1.)
    return (y_hard - y_soft).detach() + y_soft

def HardMax(x: torch.Tensor, dim=-1):
    y_soft = x
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(x, device=x.device).scatter_(dim, index, 1.)
    return (y_hard - y_soft).detach() + y_soft

############################################# Transformer #############################################
# -----------------------------------------------------------------------------------------------------

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size).permute(0, 2, 1, 3).contiguous() # [B, N, H, S]
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob + 1e-8) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiheadAttention(torch.nn.Module):
    def __init__(self,
            output_dim,           
            input_dim,             # The from/to tensors dimensions
            num_heads           = 1,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.to_q = nn.Linear(output_dim, output_dim)
        self.to_k = nn.Linear(input_dim, output_dim)
        self.to_v = nn.Linear(input_dim, output_dim)

        self.dim = output_dim
        self.output_cap_dim = output_dim
        self.to_dim = input_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_dim / num_heads)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm = nn.LayerNorm(output_dim) 
        self.dropout = DropPath(attention_dropout)

        self.proj = nn.Linear(output_dim, output_dim)


    def forward(self, input, output, mask=None): # mask shape [B, output_num, input_num]
        # queries, keys and values
        i = self.norm_input(input)
        o = self.norm(output)
        queries = self.to_q(o)
        keys    = self.to_k(i)
        values  = self.to_v(i)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input.shape
        b, n2, c2 = output.shape
        values  = transpose_for_scores(values,  self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        queries = transpose_for_scores(queries, self.num_heads, n2,   self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]
        #[B, H, 16*pred_len, 16*context_len+12action]
        if mask is not None:
            att_scores = torch.masked_fill(att_scores, mask, float('-inf'))
        
        att_probs = F.softmax(att_scores, dim=-1, dtype=att_scores.dtype)

        # Compute weighted-sum of the values using the attention distribution
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output = output + self.dropout(self.proj(control))
        return att_probs, output


class SimplexAttention(torch.nn.Module):
    def __init__(self,
            output_dim,           
            input_dim,
            # Additional options
            num_heads           = 6,              # Number of attention heads
            attention_dropout   = 0,              # Attention dropout rate
            direction = 0
        ):

        super().__init__()
        self.to_q = nn.Linear(output_dim, output_dim)
        self.to_k = nn.Linear(input_dim, output_dim)
        self.to_v = nn.Linear(input_dim, output_dim)

        self.dim = output_dim
        self.output_dim = output_dim
        self.to_dim = input_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_dim / num_heads)

        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        self.dropout = DropPath(attention_dropout)

        self.modulation = nn.Sequential(
            nn.Linear(output_dim, 2*output_dim),
            nn.GELU(),
            nn.Linear(2*output_dim, 2*output_dim)
        )

        self.proj = nn.Linear(output_dim, output_dim)
        self.norm_direction = direction
    

    def integrate(self, tensor, control): # integration, norm
        # Normalize tensor
        tensor = self.norm(tensor)
        # Compute gain/bias
        control = self.modulation(control)
        gain, bias = torch.split(control, [self.output_dim, self.output_dim], dim = -1)
        tensor = tensor + self.dropout(tensor * gain + bias)
        return tensor


    def forward(self, input, output): # mask shape [B, input_num]
        # queries, keys and values
        queries = self.to_q(output)
        keys    = self.to_k(input)
        values  = self.to_v(input)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input.shape
        b, n2, c2 = output.shape
        queries = transpose_for_scores(queries, self.num_heads, n2, self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1, self.size_head)  # [B, N, T, H]
        values  = transpose_for_scores(values,  self.num_heads, n1, self.size_head)  # [B, N, T, H]

        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]  

        if self.norm_direction == 0:
            att_probs = F.softmax(att_scores, dim=-1)
        else:
            att_probs = F.softmax(att_scores, dim=-2)
            att_probs = att_probs / (att_probs.sum(dim=-1, keepdim=True) + 1)
            
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output = self.integrate(output, self.proj(control))
        return att_probs, output


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,           input_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
            self_attn=True,
            attn_type='vanilla'
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        assert attn_type in ['vanilla', 'simplex']
        self.self_attn = self_attn
        if self_attn:
            self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout)
        self.multihead_attn = \
            MultiheadAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout) \
            if attn_type == 'vanilla' else SimplexAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.LayerNorm(output_cap_dim),
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim))

    def forward(self, input_cap, output_cap, mask_output=None):
        if self.self_attn:
            _, output_cap = self.self_attn(output_cap, output_cap, mask=mask_output)
            
        attn, output_cap = self.multihead_attn(input_cap, output_cap)
        output_cap = output_cap + self.droppath(self.FFN(output_cap))
        return attn, output_cap

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.LayerNorm(4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim),
            nn.LayerNorm(output_cap_dim))


    def forward(self, x, mask=None):
        x, _ = self.attn(x, mask=mask)
        x = self.ffn(x)
        return x
    
    def attn(self, x, mask=None):
        attn, x = self.self_attn(x, x, mask=mask)
        return x, attn
    
    def ffn(self, x):
        x = x + self.droppath(self.FFN(x))
        return x
