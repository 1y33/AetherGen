import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class DiffusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @torch.no_grad()
    def sample(self, model, n):
        model.eval()
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x

class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256, features=[64, 128, 256, 512]):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1)
        
        # Down path
        self.downs = nn.ModuleList([])
        for i in range(len(features)-1):
            self.downs.append(nn.ModuleList([
                ResBlock(features[i], features[i], time_dim),
                ResBlock(features[i], features[i]),
                nn.Conv2d(features[i], features[i+1], 4, 2, 1),
            ]))
        
        # Middle
        mid_features = features[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_features, mid_features, time_dim),
            AttentionBlock(mid_features),
            ResBlock(mid_features, mid_features, time_dim),
        ])
        
        # Up path
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(features)-1)):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(features[i+1], features[i], 4, 2, 1),
                ResBlock(features[i] * 2, features[i], time_dim),
                ResBlock(features[i], features[i], time_dim),
            ]))
            
        self.conv_out = nn.Conv2d(features[0], in_channels, kernel_size=3, padding=1)

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

def train(args):
    device = args.device
    model = ModifiedUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionModel(device=device)
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE=loss.item())
            
        # Sampling
        if epoch % args.sample_epoch == 0:
            sampled_images = diffusion.sample(model, n=8)
            # Save or display images here

if __name__ == "__main__":
    import argparse
    import math
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=500)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--lr", type=float, default=3e-4)
    # parser.add_argument("--sample_epoch", type=int, default=10)
    # args = parser.parse_args()
    
    # train(args)
    device = "cpu"
    model = ModifiedUNet().to(device)

    print(model)