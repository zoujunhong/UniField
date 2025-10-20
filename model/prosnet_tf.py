import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.networks import HardMax, HardSoftmax, FFN, TransformerEncoderLayer

class Selection(nn.Module):
    def __init__(self, known_points=8, num_points=56564):
        super().__init__()
        self.known_points = known_points
        self.point_weight = nn.Parameter(torch.randn(1, known_points, num_points, 1), requires_grad=True)
    
    def forward(self, x: torch.Tensor, tau=1.0): # x shape [B, N, D]
        B = x.shape[0]
        point_weight = torch.repeat_interleave(self.point_weight, B, 0)
        one_hot = HardSoftmax(point_weight/tau, dim=1) # weight shape [B, M, N, 1]
        x = torch.sum(x.unsqueeze(1) * one_hot, dim=2)
        return x
        
class PROSNet_tf(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, known_points=8, num_points=56564, hidden_dim=256):
        super().__init__()
        self.known_points = known_points
        self.proj_in = nn.Linear(4, hidden_dim)
        # self.selection = Selection(known_points, num_points)
        
        self.tf_layers = nn.Sequential(
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64),
            TransformerEncoderLayer(hidden_dim, hidden_dim//64))
        
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points))
    
    def forward(self, x, tau=1.0):
        # x = self.selection(x, tau)
        x = self.proj_in(x)
        x = self.tf_layers(x)
        x = torch.mean(x, dim=1)
        x = self.net(x)
        return x
