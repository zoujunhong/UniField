import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.networks import HardMax, HardSoftmax, FFN

class SelentionLinear(nn.Module):
    def __init__(self, known_points=8, num_points=56564, hidden_dim=256, stage=1):
        super().__init__()
        self.known_points = known_points
        self.stage = stage
        if stage == 2:
            self.point_weight = nn.Parameter(torch.randn(1, num_points), requires_grad=True)
        
        self.weight = nn.Parameter(torch.randn(1, num_points, hidden_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)
        
    def softmax_point_selection(self, point_cloud, tau=1.0):
        B, N = point_cloud.shape
        if self.stage == 2:
            weights = torch.repeat_interleave(self.point_weight, B, 0)
        else:
            weights = torch.randn_like(point_cloud, device=point_cloud.device)
            
        minimum = torch.min(weights).detach()
        for idx in range(self.known_points):
            one_hot = HardSoftmax(weights/tau, dim=1)
            total_weights = one_hot if idx == 0 else total_weights + one_hot
            weights = torch.masked_fill(weights, one_hot.bool(), minimum-10000)

        return total_weights
    
    def forward(self, x: torch.Tensor, tau=1.0):
        B = x.shape[0]
        point_weight = self.softmax_point_selection(x, tau)
        weight = torch.repeat_interleave(self.weight, B, 0) * point_weight.unsqueeze(-1)
        bias = torch.repeat_interleave(self.bias, B, 0)
        x = torch.matmul(x.unsqueeze(1), weight).squeeze(1) + bias
        return x
        
class MLP(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, known_points=8, num_points=56564, hidden_dim=256, stage=1):
        super().__init__()
        self.known_points = known_points
        self.selection_linear = SelentionLinear(known_points, num_points, hidden_dim, stage)
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points))
    
    def forward(self, x, tau=1.0):
        x = self.selection_linear(x, tau)
        x = self.net(x)
        return x


# class MLP(nn.Module):
#     """ Masked Autoencoder with VisionTransformer backbone
#     """
#     def __init__(self, known_points=8, num_points=56564, hidden_dim=256):
#         super().__init__()
#         self.known_points = known_points
#         self.selection_linear = SelentionLinear(known_points, num_points, hidden_dim)
#         self.net = nn.Sequential(
#             nn.ReLU(),
#             FFN(hidden_dim, hidden_dim*4),
#             FFN(hidden_dim, hidden_dim*4),
#             FFN(hidden_dim, hidden_dim*4),
#             FFN(hidden_dim, hidden_dim*4),
#             FFN(hidden_dim, hidden_dim*4),
#             FFN(hidden_dim, hidden_dim*4),
#             nn.Linear(hidden_dim, num_points))
    
#     def forward(self, x, tau=1.0):
#         x = self.selection_linear(x, tau)
#         x = self.net(x)
#         return x
    