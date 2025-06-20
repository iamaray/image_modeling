import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMPipeline, DDIMScheduler
from dsno.model import DSNO, get_sinusoidal_embeddings
import os
import time
import matplotlib.pyplot as plt

def compute_snr_weights(scheduler, timesteps):
    # Move timesteps to CPU for indexing, then back to original device
    device = timesteps.device
    timesteps_cpu = timesteps.cpu()
    alphas_cumprod = scheduler.alphas_cumprod[timesteps_cpu].to(device)
    sqrt_alpha_prod = alphas_cumprod ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5
    # SNR = alpha / (1 - alpha)
    snr = sqrt_alpha_prod / sqrt_one_minus_alpha_prod
    return snr.view(1, -1, 1, 1, 1)  # shape [1, M, 1, 1, 1] for broadcasting with [B, M, C, H, W]

class TrajectoryDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        # Flatten all batches into individual samples
        self.x_T = []
        self.traj = []
        
        for batch in data:
            x_T_batch = batch['x_T']  # [B, C, H, W]
            traj_batch = batch['trajectory']  # [B, M, C, H, W]
            
            # Split batch into individual samples
            for i in range(x_T_batch.size(0)):
                self.x_T.append(x_T_batch[i])  # [C, H, W]
                self.traj.append(traj_batch[i])  # [M, C, H, W]
        
        # Now stack all individual samples
        self.x_T = torch.stack(self.x_T)  # [N, C, H, W]
        self.traj = torch.stack(self.traj)  # [N, M, C, H, W]
    
    def __len__(self):
        return len(self.x_T)
    
    def __getitem__(self, idx):
        return self.x_T[idx], self.traj[idx]

def train(
    teacher_data_path: str,
    model: DSNO,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_every: int = 20,
    checkpoint_dir: str = 'checkpoints'
):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    dataset = TrajectoryDataset(teacher_data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=False)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.L1Loss(reduction='none')

    # dataset.traj: [N, M, C, H, W]
    M = dataset.traj.size(1)
    timesteps = torch.linspace(
        scheduler.config.num_train_timesteps - 1,
        0,
        steps=M,
        dtype=torch.long,
        device=device
    )
    weights = compute_snr_weights(scheduler, timesteps)  # [1, M, 1, 1, 1]

    # Training tracking
    train_losses = []
    log_file = 'training_log.txt'
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(loader)}")
    
    model.train()
    start_time = time.time()
    
    # Open log file
    with open(log_file, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x_T, trajectory) in enumerate(loader):
            x_T = x_T.to(device)
            trajectory = trajectory.to(device)      # [B, M, C, H, W]
            B = x_T.size(0)

            pred = model(x_T, timesteps)            # [B, M, C, H, W]
            loss_tensor = mse(pred, trajectory)     # [B, M, C, H, W]
            loss = (weights * loss_tensor).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # Log to file and console
        log_line = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s, Total: {total_time:.1f}s"
        print(log_line)
        
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'dsno_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'dsno_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    
    # Plot and save training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)')
    plt.title('DSNO Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining completed!")
    print(f"Final loss: {train_losses[-1]:.6f}")
    print(f"Best loss: {min(train_losses):.6f} at epoch {train_losses.index(min(train_losses)) + 1}")
    print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
    
    return train_losses