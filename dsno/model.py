import torch
import torch.nn as nn
import torch.fft
from typing import List, Optional

from .layers import TCBlock, UNetBackbone, get_sinusoidal_embeddings


class DSNO(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        modes=16,
        time_emb_dim=128,
        num_layers_per_block=(1,1,1,1,1),
        channels_per_block=None
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.backbone = UNetBackbone(
            in_channels,
            base_channels,
            time_emb_dim,
            num_layers_per_block=num_layers_per_block,
            channels_per_block=channels_per_block
        )
        self.tc_blocks = nn.ModuleList([
            TCBlock(self.backbone.channels[i], modes)
            for i in range(len(self.backbone.channels))
        ])
        out_ch = self.backbone.channels[-1]
        self.out_norm = nn.GroupNorm(8, out_ch)
        self.out_act = nn.LeakyReLU(0.2)
        self.out_conv = nn.Conv2d(out_ch, in_channels, 3, padding=1)

    def forward(self, xT, timesteps):
        B, C, H, W = xT.shape
        M = timesteps.shape[0]
        t_emb = get_sinusoidal_embeddings(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)
        x = xT.unsqueeze(1).expand(B, M, C, H, W).reshape(B*M, C, H, W)
        t = t_emb.unsqueeze(0).expand(B, M, t_emb.size(1)).reshape(B*M, -1)
        feats = self.backbone(x, t)
        h = None
        for feat, tc in zip(feats, self.tc_blocks):
            Bf, Cf, Hf, Wf = feat.shape
            h_flat = feat.view(Bf, Cf, Hf * Wf)
            h_tc = tc(h_flat).view(Bf, Cf, Hf, Wf)
            h = h_tc
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)
        return out.view(B, M, C, H, W)