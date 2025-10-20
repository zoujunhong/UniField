# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.networks import farthest_point_sample, knn_interpolation, LinearNorm, knn
from .utils.SlotAttention import SlotAttention
from timm.models.vision_transformer import PatchEmbed, Block
from .utils.GansformerGenerator import Generator_pointcloud

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def index_points(points, idx):
    # 从 points 中提取指定索引的点
    # points: [B, N, C], idx: [B, N, k] => out: [B, N, k, C]
    B = points.shape[0]
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1)
    return points[batch_indices, idx, :]

class MultiHeadPointAttention(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, num_heads=4):
        super().__init__()
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.k = k
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_kv = nn.Linear(in_channels, out_channels * 2)  # For key and value

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.fc_out = nn.Linear(out_channels, out_channels)

    def forward(self, x, pos):
        # x: [B, N, C], pos: [B, N, 3]
        B, N, _ = x.shape

        q = self.fc_q(x)  # [B, N, out_dim]
        kv = self.fc_kv(x)  # [B, N, out_dim * 2]
        k, v = kv.chunk(2, dim=-1)  # each [B, N, out_dim]

        # Reshape for multi-head: [B, N, H, D]
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        with torch.no_grad():
            idx = knn(pos, pos, self.k)  # [B, N, k]
        
        pos_neighbors = index_points(pos, idx)  # [B, N, k, 3]
        # q_neighbors = index_points(q, idx)  # [B, N, k, H, D]
        k_neighbors = index_points(k, idx)  # [B, N, k, H, D]
        v_neighbors = index_points(v, idx)  # [B, N, k, H, D]
        pos_diff = pos.unsqueeze(2) - pos_neighbors  # [B, N, k, 3]

        pos_enc = self.pos_mlp(pos_diff)  # [B, N, k, H*D]
        pos_enc = pos_enc.view(B, N, self.k, self.num_heads, self.head_dim)

        q_exp = q.unsqueeze(2)  # [B, N, 1, H, D]
        rel = k_neighbors - q_exp + pos_enc  # [B, N, k, H, D]

        attn = self.attn_mlp(rel.view(B, N, self.k, -1))  # [B, N, k, H*D]
        attn = attn.view(B, N, self.k, self.num_heads, self.head_dim)
        attn = F.softmax(attn, dim=2)

        agg = torch.sum(attn * (v_neighbors + pos_enc), dim=2)  # [B, N, H, D]
        agg = agg.reshape(B, N, -1)  # [B, N, out_dim]

        return self.fc_out(agg)


class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, k=16, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0, "channels must be divisible by num_heads"
        self.k = k
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.attention = MultiHeadPointAttention(in_channels, in_channels, k, num_heads)
        
        self.FFN = nn.Sequential(
            nn.Linear(in_channels,4*in_channels),
            nn.LayerNorm(4*in_channels),
            nn.GELU(),
            nn.Linear(4*in_channels,in_channels),
            nn.LayerNorm(in_channels))

    def forward(self, x, pos):
        x = x + self.attention(x, pos)
        x = x + self.FFN(x)
        return x

class PointTransformer(nn.Module):
    def __init__(self,
                 depth=[4, 4, 6, 12, 8],
                 channels=[64, 128, 256, 512, 1024],
                 num_points=[1024, 256, 64, 16],
                 out_channels=1,
                 k=16):
        super().__init__()
        self.depth = depth
        self.num_points = num_points
        self.channels = channels
        self.k = k

        self.proj_in = nn.Linear(3, channels[0])
        self.proj_out = nn.Linear(channels[0], out_channels)
        
        self.proj_down_layers = nn.ModuleList()
        self.proj_up_layers = nn.ModuleList()
        self.tf_down_layers = nn.ModuleList()
        self.tf_up_layers = nn.ModuleList()
        
        for i in range(len(depth)):
            layers = nn.ModuleList()
            for j in range(depth[i]):
                layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//64, 1), k=k))
                
            self.tf_down_layers.append(layers)
            
            if i < len(depth) - 1:
                self.proj_down_layers.append(nn.Linear(channels[i], channels[i+1]))
                self.proj_up_layers.append(nn.Linear(channels[-(i+1)], channels[-(i+2)]))
                
                layers = nn.ModuleList()
                for j in range(depth[i]):
                    layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//64, 1), k=k))
                
                self.tf_up_layers.append(layers)
        
            #     self.downsample_layers.append(PointSlotAttention(channels[i+1], channels[i+1]*4, channels[i], num_points[i]))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, pos): # [B, N, 3], [B, 2]
        B, N, _ = pos.shape
        x = self.proj_in(pos)
        x_list = []
        pos_list = []
        
        for i in range(len(self.depth)):
            
            for j in range(self.depth[i]):
                x = self.tf_down_layers[i][j](x, pos)
            
            x_list.append(x)
            pos_list.append(pos)
            
            if i < len(self.depth) - 1:
                if i == 0:
                    for j in range(B):
                        point_idx = torch.randperm(N, device=x.device)[:self.num_points[i]][None] if j == 0 else torch.cat([point_idx, torch.randperm(N, device=x.device)[:self.num_points[i]][None]], dim=0)
                else:
                    point_idx = farthest_point_sample(pos, self.num_points[i])

                pos = pos.gather(1, point_idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, k, 3]
                x = self.proj_down_layers[i](x.gather(1, point_idx.unsqueeze(-1).expand(-1, -1, self.channels[i])))  # [B, k, D]

        for i in range(len(self.depth) - 1):
            x = self.proj_up_layers[i](knn_interpolation(pos_list[-(i+2)], pos_list[-(i+1)], x, k=self.k)) + x_list[-(i+2)]
            
            for j in range(self.depth[-(i+2)]):
                x = self.tf_up_layers[-(i+1)][j](x, pos_list[-(i+2)])
        
        return self.proj_out(x)
    

    
if __name__ == '__main__':
    B, N, C = 5, 10000, 3
    pos = torch.randn(B, N, 3).cuda()

    pt_layer = PointTransformer(k=16).cuda()
    
    from thop import profile
    Flops, Params = profile(pt_layer,(pos,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))