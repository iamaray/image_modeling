import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMPipeline, DDIMScheduler
from dsno.model import DSNO, get_sinusoidal_embeddings
import os

def compute_snr_weights(scheduler, timesteps):
    sigmas = scheduler.sigmas[timesteps]
    alphas = scheduler.alphas_cumprod[timesteps]
    alpha_t = torch.sqrt(alphas)
    return (alpha_t / sigmas).view(-1, 1, 1, 1, 1)  # shape [M,1,1,1,1]

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.x_T = torch.stack([d['x_T'] for d in data])  # [N, C, H, W]
        self.traj = torch.stack([d['trajectory'] for d in data])  # [N, M, C, H, W]
    def __len__(self):
        return len(self.x_T)
    def __getitem__(self, idx):
        return self.x_T[idx], self.traj[idx]

def train(
    teacher_data_path: str,
    model: DSNO,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda'
):
    dataset = TrajectoryDataset(teacher_data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.L1Loss(reduction='none')

    # dataset.traj: [N, M, C, H, W]
    M = dataset.traj.size(1)
    timesteps = torch.linspace(
        scheduler.config.num_train_timesteps,
        0,
        steps=M,
        dtype=torch.long,
        device=device
    )
    weights = compute_snr_weights(scheduler, timesteps)  # [M,1,1,1,1]

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_T, trajectory in loader:
            x_T = x_T.to(device)
            trajectory = trajectory.to(device)      # [B, M, C, H, W]
            B = x_T.size(0)

            pred = model(x_T, timesteps)            # [B, M, C, H, W]
            loss_tensor = mse(pred, trajectory)     # [B, M, C, H, W]
            loss = (weights * loss_tensor).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
