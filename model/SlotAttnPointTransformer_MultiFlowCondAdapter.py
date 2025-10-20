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
from .utils.networks import knn_interpolation, LinearNorm, knn
from .utils.SlotAttention import PointSlotAttention

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
            # print(idx.shape)
        
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


class AdaptLayer(nn.Module):
    def __init__(self, in_channels, adapt_channels, cond_channels):
        super().__init__()
        self.in_proj = LinearNorm(in_channels, adapt_channels)
        self.out_proj = LinearNorm(adapt_channels, in_channels)
        self.cond_proj = nn.Sequential(
                            LinearNorm(cond_channels, 2*adapt_channels),
                            nn.Linear(2*adapt_channels, 2*adapt_channels))

    def forward(self, x, cond):
        shift, scale = self.cond_proj(cond).chunk(2, dim=1)
        x = x + self.out_proj(modulate(self.in_proj(x), shift, scale))    
        return x

class MultiFlowCondAdaptRouter(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x, cond, route):
        # x: (B, D_in)
        # route: (B,) integer tensor (e.g., [0, 2, 1, 0, ...])

        B = x.size(0)
        branch_outputs = []  # List of (B, D_out)
        for branch in self.branches:
            branch_outputs.append(branch(x, cond))  # 所有支路同时处理输入

        # Stack: (n_branch, B, D_out)
        stacked = torch.stack(branch_outputs, dim=0)

        # route -> one-hot: (B, n_branch)
        one_hot = F.one_hot(route, num_classes=len(self.branches)).float()

        # (n_branch, B, D_out) * (B, n_branch) → (B, D_out)
        # 利用 einsum 或广播实现选择
        selected_output = torch.einsum('mbnd,bm->bnd', stacked, one_hot)

        return selected_output

class AdaptClassifier(nn.Module):
    def __init__(self, in_channels, num_coeff):
        super().__init__()
        self.coefficient_predictor = nn.Sequential(
            LinearNorm(in_channels, 256),
            nn.Linear(256, num_coeff)
        )

    def forward(self, x):  
        return self.coefficient_predictor(x)

class MultiFlowClassifierRouter(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x, route):
        # x: (B, D_in)
        # route: (B,) integer tensor (e.g., [0, 2, 1, 0, ...])

        B = x.size(0)
        branch_outputs = []  # List of (B, D_out)
        for branch in self.branches:
            branch_outputs.append(branch(x))  # 所有支路同时处理输入

        # Stack: (n_branch, B, D_out)
        stacked = torch.stack(branch_outputs, dim=0)

        # route -> one-hot: (B, n_branch)
        one_hot = F.one_hot(route, num_classes=len(self.branches)).float()

        # (n_branch, B, D_out) * (B, n_branch) → (B, D_out)
        # 利用 einsum 或广播实现选择
        selected_output = torch.einsum('mbd,bm->bd', stacked, one_hot)

        return selected_output
    
class PointTransformer_MultiFlowCondAdapter(nn.Module):
    def __init__(self,
                 depth=[8, 8, 8, 8, 8],
                 channels=[192, 384, 768, 1536, 3072],
                 num_points=[1024, 256, 64, 16],
                 adapter_dim=64,
                 head_dim=64,
                 out_channels=1,
                 cond_dims=2,
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
        
        self.downsample_layers = nn.ModuleList()
        
        self.adapt_down_layers = nn.ModuleList()
        self.adapt_up_layers = nn.ModuleList()
        
        for i in range(len(depth)):
            layers = nn.ModuleList()
            for j in range(depth[i]):
                layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//head_dim, 1), k=k))
                
                self.adapt_down_layers.append(MultiFlowCondAdaptRouter(
                        [AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims)]))
                
            self.tf_down_layers.append(layers)
            
            if i < len(depth) - 1:
                self.proj_down_layers.append(nn.Linear(channels[i], channels[i+1]))
                self.proj_up_layers.append(nn.Linear(channels[-(i+1)], channels[-(i+2)]))
                
                layers = nn.ModuleList()
                for j in range(depth[i]):
                    layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//head_dim, 1), k=k))
                    
                    self.adapt_up_layers.append(MultiFlowCondAdaptRouter(
                        [AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims)]))
                
                self.tf_up_layers.append(layers)
        
                self.downsample_layers.append(PointSlotAttention(num_points[i], channels[i], hidden_dim=channels[i]*4, k=k))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, pos, cond, route): # [B, N, 3], [B, 2]
        B, N, _ = pos.shape
        x = self.proj_in(pos)
        x_list = []
        pos_list = []
        
        count = 0
        for i in range(len(self.depth)):
            
            for j in range(self.depth[i]):
                x = self.adapt_down_layers[count](x, cond, route)
                count = count + 1
                x = self.tf_down_layers[i][j](x, pos)
            
            x_list.append(x)
            pos_list.append(pos)
            
            if i < len(self.depth) - 1:
                x, pos = self.downsample_layers[i](x, pos)
                x = self.proj_down_layers[i](x)

        count = 0
        for i in range(len(self.depth) - 1):
            x = knn_interpolation(pos_list[-(i+2)], pos_list[-(i+1)], self.proj_up_layers[i](x), k=self.k) + x_list[-(i+2)]
            
            for j in range(self.depth[-(i+2)]):
                x = self.adapt_up_layers[-(count+1)](x, cond, route)
                count = count + 1
                x = self.tf_up_layers[-(i+1)][j](x, pos_list[-(i+2)])
        
        return self.proj_out(x)
    
    
    def forward_test(self, pos, cond, route): # [B, N, 3], [B, 2]
        B, N, _ = pos.shape
        x = self.proj_in(pos)
        x_list = []
        pos_list = []
        attn_list = []
        
        count = 0
        for i in range(len(self.depth)):
            
            for j in range(self.depth[i]):
                x = self.adapt_down_layers[count](x, cond, route)
                count = count + 1
                x = self.tf_down_layers[i][j](x, pos)
            
            x_list.append(x)
            pos_list.append(pos)
            
            if i < len(self.depth) - 1:
                x, pos, attn = self.downsample_layers[i](x, pos, return_attn=True)
                x = self.proj_down_layers[i](x)
                attn_list.append(attn)

        count = 0
        for i in range(len(self.depth) - 1):
            x = knn_interpolation(pos_list[-(i+2)], pos_list[-(i+1)], self.proj_up_layers[i](x), k=self.k) + x_list[-(i+2)]
            
            for j in range(self.depth[-(i+2)]):
                x = self.adapt_up_layers[-(count+1)](x, cond, route)
                count = count + 1
                x = self.tf_up_layers[-(i+1)][j](x, pos_list[-(i+2)])
        
        return self.proj_out(x), attn_list
    


class PointTransformer_MultiFlowCondAdapter_CoeffPred(nn.Module):
    def __init__(self,
                 depth=[8, 8, 8, 8, 8],
                 channels=[192, 384, 768, 1536, 3072],
                 num_points=[1024, 256, 64, 16],
                 adapter_dim=64,
                 out_channels=1,
                 cond_dims=2,
                 num_coeff=1,
                 k=16):
        super().__init__()
        self.depth = depth
        self.num_points = num_points
        self.channels = channels
        self.k = k

        self.proj_in = nn.Linear(3, channels[0])
        self.proj_out = nn.Linear(channels[0], out_channels)
        
        self.coefficient_predictor = MultiFlowClassifierRouter(
            [AdaptClassifier(channels[-1], num_coeff),
             AdaptClassifier(channels[-1], num_coeff),
             AdaptClassifier(channels[-1], num_coeff)]
        )
        
        self.proj_down_layers = nn.ModuleList()
        self.proj_up_layers = nn.ModuleList()
        
        self.tf_down_layers = nn.ModuleList()
        self.tf_up_layers = nn.ModuleList()
        
        self.downsample_layers = nn.ModuleList()
        
        self.adapt_down_layers = nn.ModuleList()
        self.adapt_up_layers = nn.ModuleList()
        
        for i in range(len(depth)):
            layers = nn.ModuleList()
            for j in range(depth[i]):
                layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//64, 1), k=k))
                
                self.adapt_down_layers.append(MultiFlowCondAdaptRouter(
                        [AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims)]))
                
            self.tf_down_layers.append(layers)
            
            if i < len(depth) - 1:
                self.proj_down_layers.append(nn.Linear(channels[i], channels[i+1]))
                self.proj_up_layers.append(nn.Linear(channels[-(i+1)], channels[-(i+2)]))
                
                layers = nn.ModuleList()
                for j in range(depth[i]):
                    layers.append(PointTransformerLayer(channels[i], num_heads=max(channels[i]//64, 1), k=k))
                    
                    self.adapt_up_layers.append(MultiFlowCondAdaptRouter(
                        [AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims),
                         AdaptLayer(channels[i], adapter_dim, cond_dims)]))
                
                self.tf_up_layers.append(layers)
        
                self.downsample_layers.append(PointSlotAttention(num_points[i], channels[i], hidden_dim=channels[i]*4, k=k))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, pos, cond, route): # [B, N, 3], [B, 2]
        B, N, _ = pos.shape
        x = self.proj_in(pos)
        x_list = []
        pos_list = []
        
        count = 0
        for i in range(len(self.depth)):
            
            for j in range(self.depth[i]):
                x = self.adapt_down_layers[count](x, cond, route)
                count = count + 1
                x = self.tf_down_layers[i][j](x, pos)
            
            x_list.append(x)
            pos_list.append(pos)
            
            if i < len(self.depth) - 1:
                x, pos = self.downsample_layers[i](x, pos)
                x = self.proj_down_layers[i](x)

        coeff_pred = self.coefficient_predictor(torch.mean(x, dim=1), route)
        
        count = 0
        for i in range(len(self.depth) - 1):
            x = knn_interpolation(pos_list[-(i+2)], pos_list[-(i+1)], self.proj_up_layers[i](x), k=self.k) + x_list[-(i+2)]
            
            for j in range(self.depth[-(i+2)]):
                x = self.adapt_up_layers[-(count+1)](x, cond, route)
                count = count + 1
                x = self.tf_up_layers[-(i+1)][j](x, pos_list[-(i+2)])
        
        return self.proj_out(x), coeff_pred
    
if __name__ == '__main__':
    B, N, C = 1, 8192, 3
    pos = torch.randn(B, N, 3).cuda()
    cond = torch.randn(B, 2).cuda()
    route = torch.randint(0, 2, (B,)).cuda()
    pt_layer = PointTransformer_MultiFlowCondAdapter(
                 depth=[16, 16, 16, 16, 16],
                 channels=[256, 512, 1024, 2048, 4096],
                 num_points=[1024, 256, 64, 16],
                 k=16).cuda()
    
    from thop import profile
    Flops, Params = profile(pt_layer,(pos,cond,route,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))