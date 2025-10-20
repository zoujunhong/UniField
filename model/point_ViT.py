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

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PressureFieldViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, in_chans=3, cond_channels=2,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.in_chans = in_chans
        self.depth = depth
        self.in_proj = nn.Linear(3, embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, 0.3, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        
        self.cond_proj = nn.ModuleList([
            nn.Linear(cond_channels, 2 * embed_dim)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.out_proj = nn.Linear(embed_dim, 1, bias=True) # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

    def forward(self, x, cond): # N*D, D=4, xyz+p
        # embed patches
        b = x.shape[0]
        x = self.in_proj(x)
        x[:, :1024] = x[:, :1024] + self.mask_token
        
        # apply Transformer blocks
        for i in range(self.depth):
            shift, scale = self.cond_proj[i](cond).chunk(2, dim=1)
            x = modulate(x, shift, scale)
            x, _ = self.blocks[i](x)
            
        tf_output = self.norm(x[:,:1024])
        tf_output = self.out_proj(tf_output)
        return tf_output



def pointmae_vit_large():
    model = PressureFieldViT(
        in_chans=3, cond_channels=2,
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm)
    return model
