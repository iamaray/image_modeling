import torch
import torch.nn as nn

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
    def __init__(self, dim: int, modes: int):
        super().__init__()
        self.dim = dim
        self.modes = modes
        # [in_dim, out_dim, modes]
        self.weight = nn.Parameter(torch.randn(dim, dim, modes, 2))

        self.FiLM_scale = nn.Linear(tim)

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