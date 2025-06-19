import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List
from diffusers import DDPMPipeline, DDIMScheduler

from .trajectories import make_ddim_teacher_trajectories


def process_ciphar10(
    output_path: str = 'data/ciphar10_ddpm_teacher_data.pt',
    batch_size: int = 64,
    timesteps: List[int] = [128, 256, 384, 512],
    teacher_model: str = 'ddpm'
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(
        root='data',
        train=True,
        transform=transform,
        download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    pipe = None
    scheduler = None
    teacher_funct = None

    match teacher_model:
        case 'ddpm':
            pipe = DDPMPipeline.from_pretrained(
                "google/ddpm-cifar10-32", use_safetensors=False)
            pipe.unet.to(device)
            scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            scheduler.set_timesteps(
                scheduler.config.num_train_timesteps, device=device)

            teacher_funct = make_ddim_teacher_trajectories
        case _:
            raise ValueError(
                f"Unsupported teacher model: {teacher_model}. Supported models: ['ddpm']")

    data = []
    for idx, (x0, _) in enumerate(train_loader):
        x0 = x0.to(device)
        x_T = torch.randn_like(x0)
        trajectory = teacher_funct(x_T, timesteps, pipe, scheduler)

        data.append({
            'x_T': x_T.cpu(),
            'trajectory': trajectory.cpu()
        })

        del x0, x_T, trajectory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} batches")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved teacher {teacher_model} trajectories to {output_path}.")
