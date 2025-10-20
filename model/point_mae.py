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
from .utils.networks import HardMax, HardSoftmax, TransformerEncoderLayer


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, known_points=8, in_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.known_points = known_points
        self.in_chans = in_chans
        self.pos_embed = nn.Linear(3, embed_dim)
        self.pressure_embed = nn.Linear(1, embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding
        # self.pchoose_blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=0)
        #     for i in range(4)])
        # self.norm_pchoose = norm_layer(embed_dim)
        # self.pchoose_pred = nn.Linear(embed_dim, 1, bias=True)

        dpr = [x.item() for x in torch.linspace(0, 0.3, depth)]
        self.blocks = nn.ModuleList([
            # TransformerEncoderLayer(embed_dim, num_heads, dpr[i])
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.pred = nn.Linear(embed_dim, in_chans-3, bias=True) # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        torch.nn.init.constant_(self.pred.weight, 0.)
        torch.nn.init.constant_(self.pred.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gumbel_softmax_point_selection_batch(self, weights, point_cloud):
        B, N = weights.shape
        D = point_cloud.shape[2]
        
        points = []
        # 进行 known_points 次采样
        for _ in range(self.known_points):
            one_hot = F.gumbel_softmax(weights, tau=1.0, hard=True, dim=1)
            # 通过独热码选择点云数据
            points.append(torch.sum(one_hot.unsqueeze(-1) * point_cloud, dim=1))
            weights = torch.masked_fill(weights, one_hot.bool(), float('-inf'))

        points = torch.stack(points, dim=1)  # [B, known_points, D]

        # 获取未选中的点云
        remaining_weights = torch.isinf(weights)
        unselected_points = self.select_points(point_cloud, remaining_weights)      

        return points, unselected_points

    def softmax_point_selection_batch(self, weights, point_cloud):
        B, N = weights.shape
        D = point_cloud.shape[2]
        
        points = []
        # 进行 known_points 次采样
        for _ in range(self.known_points):
            one_hot = HardSoftmax(weights, dim=1) # 测试时不再进行随机采样，而是每次直接取权重最大的点，保证同样的模型选择同样的点
            # 通过独热码选择点云数据
            print(one_hot.argmax(dim=1))
            points.append(torch.sum(one_hot.unsqueeze(-1) * point_cloud, dim=1))
            weights = torch.masked_fill(weights, one_hot.bool(), float('-inf'))

        points = torch.stack(points, dim=1)  # [B, known_points, D]

        # 获取未选中的点云
        remaining_weights = torch.isinf(weights)
        # print(torch.sum(remaining_weights))
        unselected_points = self.select_points(point_cloud, remaining_weights)

        return points, unselected_points
    
    def select_points(self, point_cloud: torch.Tensor, remaining_weights: torch.Tensor) -> torch.Tensor:
        """
        Select points from point_cloud based on remaining_weights.
        
        Args:
            point_cloud (torch.Tensor): Tensor of shape (B, N, D) representing the point cloud.
            remaining_weights (torch.Tensor): Boolean tensor of shape (B, N), where False indicates selected points.
        
        Returns:
            torch.Tensor: Tensor of shape (B, M, D) containing the selected points.
        """
        B, N, D = point_cloud.shape
        selected_points = point_cloud[~remaining_weights].reshape(B, N - self.known_points, D)
        return selected_points
    
    def point_select(self, x): # N*D, D=4, xyz+p
        # embed patches
        b = x.shape[0]
        xyz = x[:,:,:3]
        pos_embed = self.pos_embed(xyz)
            
        # select n points
        pchoose = pos_embed.clone()
        for blk in self.pchoose_blocks:
            pchoose = blk(pchoose)
        pchoose = self.norm_pchoose(pchoose)
        pchoose = self.pchoose_pred(pchoose).squeeze(-1)
        selected_points, unselected_points = self.softmax_point_selection_batch(pchoose, x)
        points = torch.cat([selected_points, unselected_points], dim=1)
        return points

    def forward(self, x): # N*D, D=4, xyz+p
        # embed patches
        b = x.shape[0]
        # xyz = x[:,:,:3]
        # pos_embed = self.pos_embed(xyz)
            
        # # select n points
        # pchoose = pos_embed.clone()
        # for blk in self.pchoose_blocks:
        #     pchoose = blk(pchoose)
        # pchoose = self.norm_pchoose(pchoose)
        # pchoose = self.pchoose_pred(pchoose).squeeze(-1)
        # selected_points, unselected_points = self.gumbel_softmax_point_selection_batch(pchoose, x)
        # points = torch.cat([selected_points, unselected_points], dim=1)
        points = x
        xyz2 = points[:,:,:3]
        p = points[:,:,3:]
        pos_embed = self.pos_embed(xyz2)
        pressure_embed = torch.cat([self.pressure_embed(p[:,:self.known_points]), self.mask_token.repeat(b, p.shape[1] - self.known_points, 1)], dim=1)

        tf_input = pos_embed + pressure_embed
        # apply Transformer blocks
        for blk in self.blocks:
            tf_input = blk(tf_input)
        tf_output = self.norm(tf_input[:,self.known_points:])
        tf_output = self.pred(tf_output)
        return tf_output, points[:,self.known_points:,3:]

    def forward_test(self, x): # N*D, D=4, xyz+p
        # embed patches
        with torch.no_grad():
            b = x.shape[0]
            xyz = x[:,:,:3]
            pos_embed = self.pos_embed(xyz)
            
            # select n points
            pchoose = pos_embed.clone()
            for blk in self.pchoose_blocks:
                pchoose = blk(pchoose)
            pchoose = self.norm_pchoose(pchoose)
            pchoose = self.pchoose_pred(pchoose).squeeze(-1)
            selected_points, unselected_points = self.softmax_point_selection_batch(pchoose, x)
            points = torch.cat([selected_points, unselected_points], dim=1)
            print(points.shape)
            xyz2 = points[:,:,:3]
            p = points[:,:,3:]
            pos_embed = self.pos_embed(xyz2)
            pressure_embed = torch.cat([self.pressure_embed(p[:,:self.known_points]), self.mask_token.repeat(b, p.shape[1] - self.known_points, 1)], dim=1)

            tf_input = pos_embed + pressure_embed
            # apply Transformer blocks
            for blk in self.blocks:
                tf_input = blk(tf_input)
            tf_output = self.norm(tf_input[:,self.known_points:])
            tf_output = self.pred(tf_output)
        
            rec_points = torch.cat([xyz2[:,self.known_points:], tf_output], dim=-1)
            total_points = torch.cat([points[:,:self.known_points], rec_points], dim=1)
        return tf_output, points[:,self.known_points:,3:], total_points

def mae_vit_large_patch4():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=8, in_chans=4,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_base():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=8, in_chans=4,
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_large():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=8, in_chans=4,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_XXL():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=8, in_chans=4,
        embed_dim=1280, depth=32, num_heads=10,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model

def pointmae_vit_XXXL():
    Model = MaskedAutoencoderViT
        
    model = Model(
        known_points=8, in_chans=4,
        embed_dim=2048, depth=32, num_heads=16,
        mlp_ratio=3., norm_layer=nn.LayerNorm)
    return model
