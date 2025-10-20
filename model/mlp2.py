import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.networks import HardMax, HardSoftmax

class SelentionLinear(nn.Module):
    def __init__(self, known_points=8, num_points=56564, hidden_dim=256):
        super().__init__()
        self.known_points = known_points
        self.point_selector = PointSelector(known_points)
        
        self.weight = nn.Parameter(torch.randn(1, num_points, hidden_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, tau=1.0):
        B = x.shape[0]
        point_weight = self.point_selector(pos, tau)
        point_weight = HardMax(point_weight, dim=-1).sum(dim=1).clamp(0,1)
        # point_weight = self.softmax_point_selection(x, tau)
        weight = torch.repeat_interleave(self.weight, B, 0) * point_weight.unsqueeze(-1)
        bias = torch.repeat_interleave(self.bias, B, 0)
        x = torch.matmul(x.unsqueeze(1), weight).squeeze(1) + bias
        return x
        
class MLP(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, known_points=8, num_points=56564, hidden_dim=256):
        super().__init__()
        self.known_points = known_points
        self.selection_linear = SelentionLinear(known_points, num_points, hidden_dim)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 56564))
    
    def forward(self, x, pos, tau=1.0):
        x = self.selection_linear(x, pos, tau)
        x = self.net(x)
        return x
    

class PointSelector(nn.Module):
    def __init__(self, x):
        """
        :param x:  选取的点数
        """
        super(PointSelector, self).__init__()
        self.x = x
        # 可学习参数：x 个 6 维向量 (均值3维 + 标准差3维)
        self.learnable_params = nn.Parameter(torch.randn(x, 3))

    def forward(self, point_cloud, std=1.0):
        """
        :param point_cloud: [B, N, d]  输入的点云
        :return: [B, x, d]  采样得到的点
        """
        B, N, _ = point_cloud.shape

        # 获取均值和标准差
        mean = self.learnable_params  # [x, 3]
        # std = F.softplus(self.learnable_params[:, 3:]) + 1e-6  # [x, 3], 通过 softplus 保证 std > 0

        # 计算每个点相对于 x 个高斯分布的概率密度
        point_cloud = point_cloud.unsqueeze(1)  # [B, 1, N, 3]
        mean = mean.unsqueeze(0).unsqueeze(2)  # [1, x, 1, 3]
        # std = std.unsqueeze(0).unsqueeze(2)  # [1, x, 1, 3]

        # 计算多维正态分布的概率密度 (未归一化)
        exponent = -0.5 * (((point_cloud - mean) / std) ** 2).mean(dim=-1)  # [B, x, N]
        pdf = torch.exp(exponent)  # [B, x, N]

        # 归一化概率
        probs = pdf / (torch.sum(pdf, dim=-1, keepdim=True))

        return probs

# 示例
# B, N, x = 2, 100, 5  # 批量大小2, 点云大小100, 选取5个点
# point_cloud = torch.randn(B, N, 3)

# model = PointSelector(x)
# selected_points = model(point_cloud)
# print(selected_points.shape)  # [B, x, 3]