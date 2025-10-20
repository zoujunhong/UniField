from .networks import FFN
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

class FFN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim))
    
    def forward(self, x):
        return F.gelu(x + self.ffn(x))

class LinearNorm(nn.Module):
    def __init__(self, inplanes, planes, act=True, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(planes, elementwise_affine=affine)
        self.conv = nn.Linear(inplanes, planes)
        self.act = act
    def forward(self, x):
        x = self.norm(self.conv(x))
        return F.gelu(x) if self.act else x

"""Slot Attention-based auto-encoder for object discovery."""
class VAE(nn.Module):
    def __init__(self, input_dim=768, mid_dim=1152, hidden_dim=128):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            LinearNorm(input_dim, mid_dim),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            nn.Linear(mid_dim, hidden_dim*2))
        
        self.decoder = nn.Sequential(
            LinearNorm(hidden_dim, mid_dim),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            FFN(mid_dim, mid_dim*2),
            nn.Linear(mid_dim, input_dim))


    def forward(self, x): # x shape [B, D]
        y = self.encoder(x)
        posterior = DiagonalGaussianDistribution(y[:,:,None,None])
        
        loss_kl = posterior.kl().mean()
        z = posterior.sample().squeeze()
        rec_x = self.decoder(z)
        loss_rec = F.mse_loss(rec_x, x)
        return loss_rec, loss_kl
    
    def encode(self, x): # x shape [B, D]
        y = self.encoder(x)
        posterior = DiagonalGaussianDistribution(y[:,:,None,None])
        z = posterior.mode().squeeze()
        return z
    
    def decode(self, z):
        return self.decoder(z)
    