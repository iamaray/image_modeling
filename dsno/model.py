import torch
import torch.nn as nn
import torch.fft

from .layers import TCBlock

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
    
    