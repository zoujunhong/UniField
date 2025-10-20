import numpy as np
from torch import nn
import torch
from .networks import SimplexAttention, TransformerDecoderLayer, LinearNorm

class ResLayer(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 planes,
                 upsample=False):
        super(ResLayer, self).__init__()
        self.planes = planes
        self.norm1 = nn.GroupNorm(1, planes, eps=1e-4)
        self.norm2 = nn.GroupNorm(1, planes, eps=1e-4)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)) if upsample else nn.Identity()
        

    def forward(self, x): # slot shape [b*n,c], x shape [b*n,c,h,w]
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        return self.upsample(self.relu(out + x))


class SynthesisBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 outplanes,
                 num_layer,
                 upsample=False):
        super(SynthesisBlock, self).__init__()
        self.attn = SimplexAttention(outplanes, inplanes, max(outplanes//64, 1), 0)
        self.forward_layers = nn.ModuleList()
        for i in range(num_layer):
            self.forward_layers.append(ResLayer(outplanes, upsample=False if i < num_layer-1 else upsample))

    def forward(self, x, slots): # slot shape [b,k,c], x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        attn, x = self.attn(slots, x)
        x = x.permute(0,2,1).reshape(b,c,h,w).contiguous()
        for i in range(len(self.forward_layers)):
            x = self.forward_layers[i](x)
        return x, attn

class TFBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 outplanes,
                 num_layer):
        super(TFBlock, self).__init__()
        self.blocks = nn.ModuleList
        self.attn = TransformerDecoderLayer(outplanes, inplanes, outplanes//64, self_attn=False, attn_type='simplex')

    def forward(self, x, slots): # slot shape [b,k,c], x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        attn, x = self.attn(slots, x)
        x = x.permute(0,2,1).reshape(b,c,h,w).contiguous()
        for i in range(len(self.forward_layers)):
            x = self.forward_layers[i](x)
        return x, attn

class Generator(nn.Module):
    def __init__(self, slot_dim=512, base_dim=512, out_dim=3, base_resolution=7, target_resolution=224, block_num=4):
        super().__init__()
        self.grid = nn.Parameter(torch.randn([1,base_dim, base_resolution,base_resolution]), requires_grad=True) # [1,4,h,w]
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            upsample = base_resolution < target_resolution
            # self.generator_blocks.append(SynthesisBlock(slot_dim, base_dim, 2, upsample=False))
            self.generator_blocks.append(SynthesisBlock(slot_dim, base_dim, 2, upsample=upsample))
            base_dim = base_dim//2 if upsample else base_dim
            base_resolution = base_resolution * 2 if upsample else base_resolution

        
        self.end_cnn = nn.Sequential(
            ResLayer(base_dim, False),
            nn.Conv2d(base_dim, out_dim, 1, 1, 0))
        

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.grid,b,0)
        for i in range(len(self.generator_blocks)):
            init_grid, attn = self.generator_blocks[i](init_grid, slots)

        return self.end_cnn(init_grid), attn


class Generator_pointcloud(nn.Module):
    def __init__(self, slot_dim=512, base_dim=512, out_dim=3, block_num=4):
        super().__init__()
        self.proj_pos = nn.Linear(3, base_dim)
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            self.generator_blocks.append(TransformerDecoderLayer(base_dim, slot_dim, base_dim//64, self_attn=False, attn_type='simplex'))

        self.proj_out = nn.Sequential(
            LinearNorm(base_dim, base_dim),
            nn.Linear(base_dim, out_dim))
        

    def forward(self, pos, slots):
        x = self.proj_pos(pos)
        b = slots.shape[0]
        for i in range(len(self.generator_blocks)):
            attn, x = self.generator_blocks[i](slots, x)

        return self.proj_out(x)
