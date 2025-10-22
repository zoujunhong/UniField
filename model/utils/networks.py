import torch
import torch.nn as nn
import torch.nn.functional as F

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

