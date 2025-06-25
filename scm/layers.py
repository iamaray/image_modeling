import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import RAdam
from functools import partial
from functorch import jvp 

IMG_SIZE       = 32
SIGMA_D        = 1.0
C_T_NORM_C     = 0.1           
T_WARMUP_H     = 10_000        
PROPOSAL_Pmu   = -1.0          
PROPOSAL_Psig  =  1.4 



def add_noise(x0, t):
    z = torch.randn_like(x0)
    return torch.cos(t)[:, None, None, None] * x0 + torch.sin(t)[:, None, None, None]*z, z

class TimeEmbed(nn.Module):
    def __init__(self, d_model=256, max_freq=0.02):
        super().__init__()
        self.freq = max_freq
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Linear(d_model*4, d_model*4)
        )
    def forward(self, t):
        half = t.unsqueeze(-1) * torch.exp(
            torch.arange(0, d_model//2, device=t.device) * math.log(self.freq))
        emb = torch.cat([torch.sin(half), torch.cos(half)], dim=-1)
        return self.proj(emb)

class TimeWeight(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        
        self.freqs = torch.exp(torch.linspace(0, math.log(0.02), emb_dim // 2))
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.SiLU(),
            nn.Linear(emb_dim*2, 1)
        )
        
    def forward(self, t):
        # t shape: (B, )
        half = t.unsqueez(-1) * self.freqs.to(t.device)
        emb = torch.cat([torch.sin(half), torch.cos(half)], dim=-1)
        return self.mlp(emb).squeeze(-1) # (B, )
    
