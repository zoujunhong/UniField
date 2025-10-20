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
from timm.models.vision_transformer import PatchEmbed, Block
from .utils.networks import HardMax, HardSoftmax
from .utils.ARDecoder import AutoregressivePredictor


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, known_points=12, depth_pcblock=4, in_chans=4,
                 embed_dim=1024, depth=24, num_heads=16, dpr=0.2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.num_points = 1024
        self.known_points = known_points
        self.in_chans = in_chans
        self.pos_embed = nn.Linear(3, embed_dim)
        
        self.pchoose_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=0)
            for i in range(depth_pcblock)])
        self.norm_pchoose = norm_layer(embed_dim)
        self.pchoose_pred = nn.Linear(embed_dim, 1, bias=True)

        self.ARpredictor = AutoregressivePredictor(hid_dim=embed_dim, depth=depth,drop_path_rate=dpr)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def softmax_point_selection_batch(self, weights, point_cloud):
        B, N = weights.shape
        D = point_cloud.shape[2]
        
        points = []
        # 进行 known_points 次采样
        for _ in range(self.num_points):
            one_hot = HardMax(weights, dim=1)
            # 通过独热码选择点云数据
            points.append(torch.sum(one_hot.unsqueeze(-1) * point_cloud, dim=1))
            weights = torch.masked_fill(weights, one_hot.bool(), 0.)

        points = torch.stack(points, dim=1)  # [B, known_points, D]

        return points

    def forward(self, x): # N*D, D=4, xyz+p
        # embed patches
        xyz = x[:,:,:3]
        pchoose = self.pos_embed(xyz)
            
        # select n points
        for blk in self.pchoose_blocks:
            pchoose = blk(pchoose)
        pchoose = self.norm_pchoose(pchoose)
        pchoose = self.pchoose_pred(pchoose).squeeze(-1)
        pchoose = F.softmax(pchoose, dim=1)
        points = self.softmax_point_selection_batch(pchoose, x)
        
        xyz2 = points[:,:,:3]
        p = points[:,:,3:]
        tf_output = self.ARpredictor(xyz2, p)
        return tf_output, points[:,1:,3:]

    def forward_test(self, x): # N*D, D=4, xyz+p
        # embed patches
        xyz = x[:,:,:3]
        pchoose = self.pos_embed(xyz)
            
        # select n points
        for blk in self.pchoose_blocks:
            pchoose = blk(pchoose)
        pchoose = self.norm_pchoose(pchoose)
        pchoose = self.pchoose_pred(pchoose).squeeze(-1)
        pchoose = F.softmax(pchoose, dim=1)
        points = self.softmax_point_selection_batch(pchoose, x)
        
        xyz2 = points[:,:,:3]
        p = points[:,:,3:]
        tf_output = self.ARpredictor(xyz2, p)
        
        rec_points = torch.cat([xyz2[:,self.known_points:], tf_output], dim=-1)
        total_points = torch.cat([points[:,:self.known_points], rec_points], dim=1)
        return tf_output, points[:,self.known_points:,3:], total_points

def mae_vit_large_patch4():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=12, in_chans=4,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_base():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=12, in_chans=4,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_large():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=12, in_chans=4,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model
