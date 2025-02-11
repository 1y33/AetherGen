import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import math


class UNetConfig:
    def __init__(self, in_channels, time_dim, features):
        self.in_channels = in_channels
        self.time_dim = time_dim
        self.features = features

class ModifiedUNet(nn.Module):
    def __init__(self, UNetConfig):
        super().__init__()
        
        self.time_dim = UNetConfig.time_dim
        self.features = UNetConfig.features
        self.in_channels = UNetConfig.in_channels
        
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Initial conv
        self.conv_in = nn.Conv2d(self.in_channels, self.features[0], kernel_size=3, padding=1)
        
        # Down path
        self.downs = nn.ModuleList([])
        for i in range(len(self.features)-1):
            self.downs.append(nn.ModuleList([
                ResBlock(self.features[i], self.features[i], self.time_dim),
                ResBlock(self.features[i], self.features[i]),
                nn.Conv2d(self.features[i], self.features[i+1], 4, 2, 1),
            ]))
        
        # Middle
        mid_features = self.features[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_features, mid_features, self.time_dim),
            AttentionBlock(mid_features),
            ResBlock(mid_features, mid_features, self.time_dim),
        ])
        
        # Up path
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.features)-1)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(self.features[i+1], self.features[i], 4, 2, 1),
                ResBlock(self.features[i] * 2, self.features[i], self.time_dim),
                ResBlock(self.features[i], self.features[i], self.time_dim),
            ]))
            
        self.conv_out = nn.Conv2d(self.features[0], self.in_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        # Time embedding    
        t = self.time_mlp(time)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Store residual connections
        residuals = []
        
        # Down path
        for down_block in self.downs:
            x = down_block[0](x, t)
            x = down_block[1](x)
            residuals.append(x)
            x = down_block[2](x)
            
        # Middle
        x = self.mid[0](x, t)
        x = self.mid[1](x)
        x = self.mid[2](x, t)
        
        # Up path
        for up_block in self.ups:
            x = up_block[0](x)
            x = torch.cat([x, residuals.pop()], dim=1)
            x = up_block[1](x, t)
            x = up_block[2](x, t)
            
        return self.conv_out(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = nn.SiLU()(h)

        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb[..., None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = nn.SiLU()(h)

        return h + self.shortcut(x)
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)
        
        self.scale = channels ** -0.5  

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = self.proj_out(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out + x.permute(0, 2, 1).reshape(B, C, H, W)
