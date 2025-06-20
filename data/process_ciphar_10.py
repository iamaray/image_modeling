import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List
from diffusers import DDPMPipeline, DDIMScheduler

from .trajectories import make_ddim_teacher_trajectories


def process_cifar10(output_path: str = 'data/cifar10_teacher.pt', batch_size: int = 16, timesteps: List[int] = [128, 256, 384, 512], teacher_model: str = 'ddpm'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = datasets.CIFAR10(root='data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]), download=True)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)
    torch.save(loader, 'data/train_loader.pt')
    
    test_set = datasets.CIFAR10(root='data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]), download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    torch.save(test_loader, 'data/test_loader.pt')
    
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=False).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
    
    data = []
    
    num_batches = 1_000_000 // batch_size
    batches = num_batches * batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # x0 = x0.to(device)
            # print(x0.shape)
            x_T = torch.randn((batch_size, 3, 32, 32))
            
            trajectory = make_ddim_teacher_trajectories(x_T, timesteps, pipe, scheduler)
            
            data.append({
                'x_T': x_T.detach().cpu(), 
                'trajectory': trajectory.detach().cpu()
            })
            
            del x_T, trajectory
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{batches}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)