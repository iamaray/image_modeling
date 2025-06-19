import torch
from diffusers import DDPMPipeline, DDIMScheduler
from typing import List, Optional

def make_ddim_teacher_trajectories(
    x_T : torch.Tensor, 
    timesteps: List[int], 
    pipe: DDPMPipeline, 
    scheduler: DDIMScheduler) -> torch.Tensor :
    
    """
        x_T: [B, C, H, W] noise.
        timesteps: length-M list of reverse process timesteps to record.
        pipe: pretrained DDPM + UNet pipeine.
        scheduler: DDIMScheduler configured with .set_timesteps

        
        Returns: tensor [B, M, C, H, W] of latents at timesteps.
    """
    
    scheduler.set_timesteps(max(timesteps), device=x_T.device)
    
    traj = []
    ts_set = set(timesteps)
    
    for step in scheduler.timesteps:
        noise_pred = pipe.unet(latents, step).sample
        latents = scheduler.step(noise_pred, step, latents).prev_sample
        
        if step in ts_set:
            traj.append(latents)
            
    return torch.stack(traj, dim=1) # [B, M, C, H, W]