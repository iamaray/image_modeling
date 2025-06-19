import torch
import torch.nn as nn
import torch.fft

class TCBlock(nn.Module):
    """
    Temporal block using Fourier modes for convolution along time axis.
    """
    def __init__(self, dim, modes):
        super().__init__()
        self.dim = dim
        self.modes = modes
        # [in_dim, out_dim, modes]
        self.weight = nn.Parameter(torch.randn(dim, dim, modes, 2))

    def forward(self, x):
        # x: [batch, dim, time]
        # x_ft: [batch, dim, freq]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        for i in range(self.modes):
            w = torch.complex(self.weight[:, :, i, 0], self.weight[:, :, i, 1])  # [dim, dim]
            # x_ft[..., i]: [batch, dim]
            out_ft[..., i] = x_ft[..., i] @ w.T
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class DSNO(nn.Module):
    """
    Bare-bones DSNO model: U-Net backbone + Fourier temporal conv block + final projection.
    """
    def __init__(self, in_channels=3, hidden_channels=64, modes=16):
        super().__init__()
        self.backbone = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.tc = TCBlock(hidden_channels, modes)
        self.proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, xT):
        # xT: [B, C, H, W]
        B, C, H, W = xT.shape
        feat = self.backbone(xT)                        # [B, hidden, H, W]
        feat_flat = feat.view(B, feat.shape[1], -1)     # [B, hidden, H*W]
        feat_time = self.tc(feat_flat)                  # [B, hidden, H*W]
        feat = feat_time.view(B, feat.shape[1], H, W)
        return self.proj(feat)                          # [B, C, H, W]