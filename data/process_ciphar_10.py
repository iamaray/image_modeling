import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List
from diffusers import DDPMPipeline, DDIMScheduler

from .trajectories import make_ddim_teacher_trajectories


def process_cifar10(output_path: str = 'data/cifar10_teacher.pt', batch_size: int = 64, timesteps: List[int] = [128, 256, 384, 512], teacher_model: str = 'ddpm'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=False).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
    data = []
    for x0, _ in loader:
        x0 = x0.to(device)
        x_T = torch.randn_like(x0)
        trajectory = make_ddim_teacher_trajectories(x_T, timesteps, pipe, scheduler)
        data.append({'x_T': x_T.cpu(), 'trajectory': trajectory.cpu()})
        del x0, x_T, trajectory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)