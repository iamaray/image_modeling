import torch
import torch.nn as nn
import torch.fft

def get_sinusoidal_embeddings(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freq = torch.exp(-torch.arange(half, device=device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1)))
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, emb[:, :1]], dim=1)
    return emb  # [M, dim]

class TCBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, channels, modes, 2))
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        Bf, C, M = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        max_modes = min(self.modes, x_ft.size(-1))
        for i in range(max_modes):
            w = torch.complex(self.weight[:, :, i, 0], self.weight[:, :, i, 1])
            out_ft[..., i] = x_ft[..., i] @ w.T
        x_ifft = torch.fft.irfft(out_ft, n=M, dim=-1)
        return x + self.activation(x_ifft)

class ResnetBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film_scale = nn.Linear(time_emb_dim, channels)
        self.film_shift = nn.Linear(time_emb_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        scale = self.film_scale(t).view(-1, h.shape[1], 1, 1)
        shift = self.film_shift(t).view(-1, h.shape[1], 1, 1)
        h = h * (1 + scale) + shift
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return x + h

class ResNetBlockSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x, t):
        h = x
        for block in self.blocks:
            h = block(h, t)
        return h

class UNetBackbone(nn.Module):
    """
    U-Net spatial backbone with skip connections.
    Supports custom channel widths per level.
    Returns feature maps at each resolution: [down1, down2, bottom, up1, up2].
    """
    def __init__(
        self,
        in_channels,
        base_channels,
        time_emb_dim,
        num_layers_per_block=(1, 1, 1, 1, 1),
        channels_per_block=None
    ):
        super().__init__()
        
        if channels_per_block is None:
            c0 = base_channels
            channels = [c0, c0 * 2, c0 * 4, c0 * 2, c0]
        else:
            channels = list(channels_per_block)
        self.channels = channels
        
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        for i in range(2):
            seq = [ResnetBlock(channels[i], time_emb_dim)
                   for _ in range(num_layers_per_block[i])]
            self.blocks.append(ResNetBlockSequence(seq))
            
            self.pools.append(nn.AvgPool2d(2))
            self.down_convs.append(nn.Conv2d(channels[i], channels[i + 1], 1))
        
        seq = [ResnetBlock(channels[2], time_emb_dim)
               for _ in range(num_layers_per_block[2])]
        self.blocks.append(ResNetBlockSequence(seq))
        
        self.ups = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest') for _ in range(2)
        ])
        self.up_convs = nn.ModuleList([
            nn.Conv2d(channels[2], channels[3], 1),
            nn.Conv2d(channels[3], channels[4], 1)
        ])
        
        for i in range(3, 5):
            seq = [ResnetBlock(channels[i], time_emb_dim)
                   for _ in range(num_layers_per_block[i])]
            self.blocks.append(ResNetBlockSequence(seq))

    def forward(self, x, t):
        feats = []
        
        h = self.init_conv(x)
        h = self.blocks[0](h, t)
        feats.append(h)  # down1
        
        h = self.pools[0](h)
        h = self.down_convs[0](h)
        h = self.blocks[1](h, t)
        feats.append(h)  # down2
        
        h = self.pools[1](h)
        h = self.down_convs[1](h)
        
        h = self.blocks[2](h, t)
        feats.append(h)  # bottom
        
        h = self.ups[0](h)
        h = self.up_convs[0](h)
        h = h + feats[1]  # skip connect from down2
        h = self.blocks[3](h, t)
        feats.append(h)  # up1
        
        h = self.ups[1](h)
        h = self.up_convs[1](h)
        h = h + feats[0]  # skip connect from down1
        h = self.blocks[4](h, t)
        feats.append(h)  # up2
        
        return feats