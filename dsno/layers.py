import torch
import torch.nn as nn
from typing import List, Optional

def get_sinusoidal_embeddings(timesteps: torch.Tensor, dim: int):
    device = timesteps.device
    half = dim // 2
    
    idx_scale = -torch.log(torch.tensor(10000.0)) / (half - 1)
    freq = torch.exp(torch.arange(half, device=device).float() * idx_scale) # [half]
    
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0) # [M, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb[:, :1]], dim=1)
    
    return emb # [M, dim]



class TCBlock(nn.Module):
    """
    Temporal block using Fourier modes for convolution along time axis.
    """
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = channels
        self.modes = modes
        # [in_dim, out_dim, modes]
        self.weight = nn.Parameter(torch.randn(dim, dim, modes, 2))
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # x: [batch_flat, dim, time]
        Bf, C, M = x.shape
        # scale = self.FiLM_scale(t_emb).transpose(0, 1).unsqueeze(0) # [1, dim, M]
        # shift = self.FiLM_shift(t_emb).transpose(0, 1).unsqueeze(0) # [1, dim, M]
        # x = x * (1 + scale) + shift
        
        # x_ft: [batch, dim, freq]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        for i in range(self.modes):
            w = torch.complex(self.weight[:, :, i, 0], self.weight[:, :, i, 1])  # [dim, dim]
            # x_ft[..., i]: [batch, dim]
            out_ft[..., i] = x_ft[..., i] @ w.T
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x + self.activation(x_ifft)
    
    
class ResNetBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.FiLM_scale = nn.Linear(emb_dim, channels)
        self.FiLM_shift = nn.Linear(emb_dim, channels)
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.norm1(self.act(x))
        h = self.conv1(h)
        
        scale = self.FiLM_scale(t_emb).view(-1, h.shape[1], 1, 1)
        shift = self.FiLM_shift(t_emb).view(-1, h.shape[1], 1, 1)
        
        h = h * (1 + scale) + shift
        h = self.norm2(self.act(h))
        h = self.conv2(h)
        
        return x + h
    
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        emb_dim: int,
        num_layers_per_block: List[int]=[1,1,1,1,1],
        channels_per_block: Optional[List[int]]=None
    ):
        super().__init__()
        
        if channels_per_block is None:
            c0 = base_channels
            channels = [c0, c0*2, c0*4, c0*2, c0]
        else:
            channels = channels_per_block
        self.channels = channels
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        for i in range(2):
            layers = [ResNetBlock(channels[i], emb_dim)
                      for _ in range(num_layers_per_block[i])]
            
            self.blocks.append(nn.Sequential(*layers))
            self.pools.append(nn.AvgPool2d(2))
            
            self.down_convs.append(nn.Conv2d(channels[i], channels[i+1], 1))
        
        layers = [ResNetBlock(channels[2], emb_dim)
                  for _ in range(num_layers_per_block[2])]
        self.blocks.append(nn.Sequential(*layers))
        
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest') for _ in range(2)])
        self.up_convs = nn.ModuleList([
            nn.Conv2d(channels[2], channels[3], 1),
            nn.Conv2d(channels[3], channels[4], 1)
        ])
        for i in range(3, 5):
            layers = [ResnetBlock(channels[i], time_emb_dim)
                      for _ in range(num_layers_per_block[i])]
            self.blocks.append(nn.Sequential(*layers))
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        skips = []
        h = self.init_conv(x)
        idx = 0
        
        for i, pool in enumerate(self.pools):
            h = self.blocks[idx](h, t)
            skips.append(h)
            h = pool(h)
            h = self.down_convs[i](h)
            idx += 1

        h = self.blocks[idx](h, t)
        idx += 1
        
        for i, up in enumerate(self.ups):
            h = up(h)
            h = self.up_convs[i](h)
            skip = skips[-(i+1)]
            h = h + skip
            h = self.blocks[idx](h, t)
            idx += 1
        
        return skips + [h]