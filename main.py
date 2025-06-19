import torch
from data.process_ciphar_10 import process_cifar10
import argparse

from dsno.model import DSNO
from dsno.trainer import train

def process():
    process_cifar10(batch_size=64)
    
def train_model(epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSNO()
    train(
        teacher_data_path='data/ciphar10_ddpm_teacher_data.pt',
        model=model,
        batch_size=64,
        epochs=epochs,
        device=device)


def main():
    parser = argparse.ArgumentParser(description='DSNO Training Pipeline')
    parser.add_argument('command', choices=['process', 'train'], 
                       help='Command to run: process or train')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        print("Processing CIFAR-10 data...")
        process()
        print("Data processing complete!")
    elif args.command == 'train':
        print(f"Training model for {args.epochs} epochs...")
        train_model(epochs=args.epochs)
        print("Training complete!")

if __name__ == "__main__":
    main()