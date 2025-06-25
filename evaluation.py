import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from diffusers import DDIMScheduler
import os
from typing import Optional
from torchvision.models import inception_v3

from dsno.model import DSNO

def compute_fid(real_features, fake_features):
    """Compute Frechet Inception Distance between real and fake features"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def extract_inception_features(images, inception_model, device):
    """Extract features using InceptionV3"""
    inception_model.eval()
    features = []
    
    with torch.no_grad():
        for batch in DataLoader(images, batch_size=32, shuffle=False):
            batch = batch.to(device)
            if batch.size(1) == 3 and batch.size(-1) == 32:
                batch = F.interpolate(batch, size=299, mode='bilinear', align_corners=False)
            feat = inception_model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def generate_samples(model, num_samples, device, scheduler=None):
    """Generate samples using the DSNO model"""
    model.eval()
    
    if scheduler is None:
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(50)
    
    # Create timesteps for generation
    timesteps = torch.linspace(999, 0, steps=4, dtype=torch.long, device=device)
    
    samples = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Start with noise
            x_T = torch.randn(current_batch_size, 3, 32, 32, device=device)
            
            # Generate trajectory
            trajectory = model(x_T, timesteps)  # [B, M, C, H, W]
            
            # Take the last timestep as the final sample
            final_samples = trajectory[:, -1]  # [B, C, H, W]
            samples.append(final_samples.cpu())
    
    return torch.cat(samples, dim=0)

def evaluate_model(model_path, device='cuda', num_samples=1000):
    """Comprehensive model evaluation"""
    print(f"Loading model from {model_path}")
    
    model = DSNO()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Load real CIFAR-10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.CIFAR10(root='data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Generating {num_samples} samples...")
    fake_samples = generate_samples(model, num_samples, device)
    
    # Get real samples
    real_samples = []
    for i, (images, _) in enumerate(test_loader):
        real_samples.append(images)
        if len(real_samples) * 64 >= num_samples:
            break
    real_samples = torch.cat(real_samples, dim=0)[:num_samples]
    
    # Compute basic statistics
    fake_mean = fake_samples.mean().item()
    fake_std = fake_samples.std().item()
    real_mean = real_samples.mean().item()
    real_std = real_samples.std().item()
    
    print(f"Real samples - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
    print(f"Fake samples - Mean: {fake_mean:.4f}, Std: {fake_std:.4f}")
    
    # Try to compute FID (requires torchvision inception model)
    try:
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove final classification layer
        inception = inception.to(device)
        
        print("Computing FID...")
        real_features = extract_inception_features(real_samples, inception, device)
        fake_features = extract_inception_features(fake_samples, inception, device)
        
        fid_score = compute_fid(real_features, fake_features)
        print(f"FID Score: {fid_score:.2f}")
        
    except Exception as e:
        print(f"Could not compute FID: {e}")
        fid_score = None
    
    # Save sample images
    save_sample_grid(fake_samples[:64], 'generated_samples.png')
    save_sample_grid(real_samples[:64], 'real_samples.png')
    
    results = {
        'fake_mean': fake_mean,
        'fake_std': fake_std,
        'real_mean': real_mean,
        'real_std': real_std,
        'fid_score': fid_score
    }
    
    return results

def save_sample_grid(samples, filename, nrow=8):
    """Save a grid of sample images"""
    from torchvision.utils import save_image
    
    # Clamp to [0, 1] range
    samples = torch.clamp(samples, 0, 1)
    
    save_image(samples, filename, nrow=nrow, normalize=False)
    print(f"Saved sample grid to {filename}")

def plot_training_curves(log_file='training_log.txt'):
    """Plot training curves from log file"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return
    
    epochs = []
    losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss' in line:
                parts = line.strip().split(',')
                epoch_part = parts[0].split('/')
                epoch = int(epoch_part[0].split()[-1])
                loss_part = parts[1].split(':')
                loss = float(loss_part[-1].strip())
                
                epochs.append(epoch)
                losses.append(loss)
    
    if epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Best loss: {min(losses):.6f} at epoch {epochs[losses.index(min(losses))]}")
    else:
        print("No training data found in log file")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DSNO model')
    parser.add_argument('--model_path', type=str, default='checkpoints/dsno_final.pth',
                       help='Path to trained model')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--plot_curves', action='store_true',
                       help='Plot training curves')
    
    args = parser.parse_args()
    
    if args.plot_curves:
        plot_training_curves()
    
    if os.path.exists(args.model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = evaluate_model(args.model_path, device, args.num_samples)
        print("\nEvaluation Results:")
        for key, value in results.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
    else:
        print(f"Model file {args.model_path} not found")