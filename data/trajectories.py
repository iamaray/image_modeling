import torch
from diffusers import DDPMPipeline, DDIMScheduler
from typing import List, Optional

def make_ddim_teacher_trajectories(x_T: torch.Tensor, timesteps: List[int], pipe: DDPMPipeline, scheduler: DDIMScheduler) -> torch.Tensor:
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=x_T.device)
    latents = x_T
    traj = []
    for step in sorted(timesteps, reverse=True):
        noise_pred = pipe.unet(latents, step).sample
        latents = scheduler.step(noise_pred, step, latents).prev_sample
        traj.append(latents)
    return torch.stack(traj, dim=1)